use crate::chat::tools::{ToolCall, ToolConfig, ToolEvent, ToolOutputParser};
use anyhow::{Result, ensure};

#[derive(Clone, Debug)]
pub(super) enum Delta {
    Content(String),
    Reasoning(String),
    Tool(ToolCall),
}

#[derive(Default, Clone, Debug)]
pub(super) struct Reply {
    pub content: String,
    pub reasoning: String,
    pub calls: Vec<ToolCall>,
}

#[derive(Default)]
struct StopFilter {
    stops: Vec<String>,
    pending: String,
    stopped: bool,
}
impl StopFilter {
    fn push(&mut self, text: &str) -> String {
        if self.stopped {
            return String::new();
        }
        self.pending.push_str(text);
        if let Some(index) = self.stops.iter().filter_map(|s| self.pending.find(s)).min() {
            self.stopped = true;
            let result = self.pending[..index].to_owned();
            self.pending.clear();
            return result;
        }
        let keep = self
            .pending
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| self.stops.iter().any(|s| s.starts_with(&self.pending[i..])))
            .unwrap_or(self.pending.len());
        self.pending.drain(..keep).collect()
    }
    fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

struct ReasoningSplitter {
    thinking: bool,
    pending: String,
}
impl ReasoningSplitter {
    fn push(&mut self, text: &str) -> Vec<Delta> {
        self.pending.push_str(text);
        let mut result = Vec::new();
        loop {
            // Only the checkpoint's preopened reasoning phase is structural.
            // Tags inside the answer (including JSON/code strings) are literal.
            if !self.thinking {
                let text = std::mem::take(&mut self.pending);
                if !text.is_empty() {
                    result.push(Delta::Content(text));
                }
                break;
            }
            let marker = "</think>";
            if let Some(index) = self.pending.find(marker) {
                let text = self.pending[..index].to_owned();
                if !text.is_empty() {
                    result.push(self.delta(text));
                }
                self.pending.drain(..index + marker.len());
                self.thinking = !self.thinking;
            } else {
                let keep = self
                    .pending
                    .char_indices()
                    .map(|(i, _)| i)
                    .find(|&i| marker.starts_with(&self.pending[i..]))
                    .unwrap_or(self.pending.len());
                let text = self.pending.drain(..keep).collect::<String>();
                if !text.is_empty() {
                    result.push(self.delta(text));
                }
                break;
            }
        }
        result
    }
    fn delta(&self, text: String) -> Delta {
        if self.thinking {
            Delta::Reasoning(text)
        } else {
            Delta::Content(text)
        }
    }
    fn finish(&mut self) -> Vec<Delta> {
        let rest = std::mem::take(&mut self.pending);
        if rest.is_empty() {
            Vec::new()
        } else {
            vec![self.delta(rest)]
        }
    }
}

pub(super) struct OutputProcessor {
    stop: StopFilter,
    parser: Option<ToolOutputParser>,
    reasoning: ReasoningSplitter,
    raw: String,
    pub reply: Reply,
}
impl OutputProcessor {
    pub fn new(stops: Vec<String>, config: ToolConfig, id: String, thinking: bool) -> Self {
        let parser = if config.definitions.is_empty() {
            None
        } else {
            Some(ToolOutputParser::new(config, id, thinking))
        };
        Self {
            stop: StopFilter {
                stops,
                ..Default::default()
            },
            parser,
            reasoning: ReasoningSplitter {
                thinking,
                pending: String::new(),
            },
            raw: String::new(),
            reply: Reply::default(),
        }
    }
    pub fn stopped(&self) -> bool {
        self.stop.stopped
    }
    pub fn push(&mut self, text: &str) -> Result<Vec<Delta>> {
        if self.stopped() {
            return Ok(Vec::new());
        }
        self.raw.push_str(text);
        let visible = self.stop.push(text);
        self.process_visible(&visible)
    }
    fn process_visible(&mut self, text: &str) -> Result<Vec<Delta>> {
        let events = match &mut self.parser {
            Some(parser) => parser.push(text)?,
            None => vec![ToolEvent::Text(text.into())],
        };
        self.events(events)
    }
    fn events(&mut self, events: Vec<ToolEvent>) -> Result<Vec<Delta>> {
        let mut result = Vec::new();
        for event in events {
            match event {
                ToolEvent::Text(text) => result.extend(self.reasoning.push(&text)),
                ToolEvent::Call(call) => {
                    result.extend(self.reasoning.finish());
                    result.push(Delta::Tool(call));
                }
            }
        }
        self.record(&result);
        Ok(result)
    }
    fn record(&mut self, deltas: &[Delta]) {
        for delta in deltas {
            match delta {
                Delta::Content(s) => self.reply.content.push_str(s),
                Delta::Reasoning(s) => self.reply.reasoning.push_str(s),
                Delta::Tool(call) => self.reply.calls.push(call.clone()),
            }
        }
    }
    pub fn finish(&mut self, final_text: &str) -> Result<Vec<Delta>> {
        let mut result = Vec::new();
        if !self.stopped() {
            ensure!(
                final_text.starts_with(&self.raw),
                "generator final text differs from emitted prefixes"
            );
            let tail = final_text[self.raw.len()..].to_owned();
            result.extend(self.push(&tail)?);
        }
        let tail = self.stop.finish();
        result.extend(self.process_visible(&tail)?);
        if let Some(parser) = &mut self.parser {
            let events = parser.finish()?;
            result.extend(self.events(events)?);
        }
        let tail = self.reasoning.finish();
        self.record(&tail);
        result.extend(tail);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stop_never_leaks_a_sequence_across_unicode_chunks() {
        let full = "Łódź: koniec🛑NEXTsecret";
        for split in full.char_indices().map(|(i, _)| i) {
            let mut filter = StopFilter {
                stops: vec!["🛑NEXT".into()],
                ..Default::default()
            };
            let mut text = filter.push(&full[..split]);
            text.push_str(&filter.push(&full[split..]));
            text.push_str(&filter.finish());
            assert_eq!(text, "Łódź: koniec");
            assert!(filter.stopped);
        }
    }
    #[test]
    fn incomplete_stop_prefix_is_flushed_at_normal_finish() {
        let mut filter = StopFilter {
            stops: vec!["END".into()],
            ..Default::default()
        };
        assert_eq!(filter.push("hello E"), "hello ");
        assert_eq!(filter.finish(), "E");
        assert!(!filter.stopped);
    }
    #[test]
    fn reasoning_and_stop_processing_preserve_visible_text() {
        let mut p = OutputProcessor::new(
            vec!["STOP".into()],
            ToolConfig::default(),
            "call_x".into(),
            true,
        );
        for s in ["my ", "thought</thi", "nk>\nanswerST", "OPother"] {
            p.push(s).unwrap();
        }
        // Cancellation can leave the engine's final text behind the callback that stopped it.
        p.finish("my thought</think>\n").unwrap();
        assert_eq!(p.reply.reasoning, "my thought");
        assert_eq!(p.reply.content, "\nanswer");
        assert!(p.stopped());
    }
    #[test]
    fn final_unstreamed_suffix_is_consumed_once() {
        let mut p = OutputProcessor::new(vec![], ToolConfig::default(), "call_x".into(), false);
        p.push("hel").unwrap();
        p.finish("hello").unwrap();
        assert_eq!(p.reply.content, "hello");
    }
}
