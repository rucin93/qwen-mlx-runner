use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

use serde_json::{Value, json};

use crate::chat::GenerationRequest;

// A sink is supplied once at server startup; disabled tracing has no metadata
// allocation and tests can observe real request events without global env races.
pub(super) type RequestLog = Arc<dyn Fn(Value) + Send + Sync>;

#[derive(Clone)]
pub(super) struct RequestTrace(Arc<TraceState>);
struct TraceState {
    sink: RequestLog,
    metadata: Value,
    started: Instant,
    terminal: AtomicBool,
}
impl RequestTrace {
    pub(super) fn new(
        sink: RequestLog,
        id: &str,
        model: &str,
        request: &GenerationRequest,
        started: Instant,
    ) -> Self {
        Self(Arc::new(TraceState {
            sink,
            metadata: json!({
                "kind":"request_timing", "id":id, "model":model,
                "engine_version":env!("CARGO_PKG_VERSION"),
                "message_count":request.messages.len(),
                "tool_count":request.tools.definitions.len(),
                "message_content_bytes":request.messages.iter().map(|m| m.content.len()).sum::<usize>(),
                "message_bytes":serde_json::to_vec(&request.messages).map_or(0, |v| v.len()),
                "tool_bytes":serde_json::to_vec(&request.tools.definitions).map_or(0, |v| v.len()),
                "thinking":request.enable_thinking, "reasoning_effort":request.reasoning_effort,
            }),
            started,
            terminal: AtomicBool::new(false),
        }))
    }
    pub(super) fn metadata(&self) -> Value {
        self.0.metadata.clone()
    }
    pub(super) fn log(&self, value: Value) {
        (self.0.sink)(value);
    }
    fn lifecycle(&self, event: &str, fields: Value) {
        let mut value = self.metadata();
        value["kind"] = json!("request_lifecycle");
        value["event"] = json!(event);
        value["elapsed_seconds"] = json!(self.0.started.elapsed().as_secs_f64());
        value
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        self.log(value);
    }
    pub(super) fn event(&self, event: &str, fields: Value) {
        if !self.0.terminal.load(Ordering::Relaxed) {
            self.lifecycle(event, fields);
        }
    }
    pub(super) fn terminal(&self, status: &str, stage: &str, code: &str, fields: Value) {
        // The worker and HTTP drop guard can observe the same disconnect. Only
        // the first terminal observation emits a line, including during prepare
        // or a generator that has not yet returned to its cancellation callback.
        if self.0.terminal.swap(true, Ordering::Relaxed) {
            return;
        }
        let mut fields = fields;
        fields["status"] = json!(status);
        fields["stage"] = json!(stage);
        fields["code"] = json!(code);
        self.lifecycle("terminal", fields);
    }
}
