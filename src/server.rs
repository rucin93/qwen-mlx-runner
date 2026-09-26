use std::net::SocketAddr;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc::{self, SyncSender},
};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Result, bail};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use serde_json::{Value, json};
use tokio::sync::mpsc as async_mpsc;

use crate::chat::{GenerationOutput, GenerationRequest, Message, TextGenerator};

mod output;
mod request;
use output::{Delta, OutputProcessor, Reply};
use request::{ParsedRequest, RequestError, ResponseFormat, StreamOptions};

const QUEUE_CAPACITY: usize = 4;
const EVENT_CAPACITY: usize = 16;
static COMPLETION_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
struct ServerState {
    sender: SyncSender<Job>,
    model: String,
    vocab_size: Option<usize>,
}

struct Job {
    queued_at: Instant,
    request: GenerationRequest,
    stop: Vec<String>,
    response_format: ResponseFormat,
    id: String,
    events: async_mpsc::Sender<WorkerEvent>,
    cancelled: Arc<AtomicBool>,
}

enum WorkerEvent {
    Ready,
    InvalidRequest(RequestError),
    Delta(Delta),
    Done(GenerationOutput, Reply, RequestTiming),
    Error(String),
}

#[derive(Default, Debug, serde::Serialize)]
struct RequestTiming {
    queue_seconds: f64,
    prepare_seconds: f64,
    first_model_text_seconds: Option<f64>,
    first_reasoning_seconds: Option<f64>,
    first_content_seconds: Option<f64>,
    first_tool_call_seconds: Option<f64>,
    total_seconds: f64,
}
impl RequestTiming {
    fn ready(&mut self, delta: &Delta, start: Instant) {
        let field = match delta {
            Delta::Reasoning(text) if !text.trim().is_empty() => &mut self.first_reasoning_seconds,
            Delta::Content(text) if !text.trim().is_empty() => &mut self.first_content_seconds,
            Delta::Tool(_) => &mut self.first_tool_call_seconds,
            _ => return,
        };
        field.get_or_insert_with(|| start.elapsed().as_secs_f64());
    }
}

struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

pub async fn serve(engine: Box<dyn TextGenerator>, address: SocketAddr) -> Result<()> {
    if !address.ip().is_loopback() {
        bail!("HTTP server must bind to a loopback address");
    }
    let model = engine.model_id().to_owned();
    let vocab_size = engine.vocab_size();
    let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
    std::thread::Builder::new()
        .name("qwen-generation".into())
        .spawn(move || worker(engine, receiver))?;
    let app = router(ServerState {
        sender,
        model,
        vocab_size,
    });
    let listener = tokio::net::TcpListener::bind(address).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn router(state: ServerState) -> Router {
    Router::new()
        .route("/health", get(|| async { Json(json!({"status":"ok"})) }))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(completions))
        .layer(DefaultBodyLimit::max(1024 * 1024))
        .with_state(state)
}

fn worker(mut engine: Box<dyn TextGenerator>, receiver: mpsc::Receiver<Job>) {
    let log_requests = std::env::var("QWEN_METAL_LOG_REQUESTS").as_deref() == Ok("1");
    while let Ok(mut job) = receiver.recv() {
        if job.cancelled.load(Ordering::Relaxed) || job.events.is_closed() {
            continue;
        }
        let mut timing = RequestTiming {
            queue_seconds: job.queued_at.elapsed().as_secs_f64(),
            ..Default::default()
        };
        let mut metadata = log_requests.then(|| json!({"kind":"request_timing","id":job.id,"model":engine.model_id(),"message_count":job.request.messages.len(),"tool_count":job.request.tools.definitions.len(),"thinking":job.request.enable_thinking,"reasoning_effort":job.request.reasoning_effort}));
        let prepare_started = Instant::now();
        if job.response_format == ResponseFormat::JsonObject {
            let instruction = "Respond with one valid JSON object only. Do not use Markdown code fences or text outside the JSON object.";
            if let Some(first) = job
                .request
                .messages
                .first_mut()
                .filter(|m| m.role == "system")
            {
                first.content.push_str("\n\n");
                first.content.push_str(instruction);
            } else {
                job.request.messages.insert(
                    0,
                    Message {
                        role: "system".into(),
                        content: instruction.into(),
                        ..Default::default()
                    },
                );
            }
        }
        if let Err(error) = engine.prepare_request(&mut job.request) {
            let code = if error
                .downcast_ref::<crate::chat::ContextLengthExceeded>()
                .is_some()
            {
                "context_length_exceeded"
            } else {
                "invalid_value"
            };
            let _ = job
                .events
                .blocking_send(WorkerEvent::InvalidRequest(RequestError {
                    message: error.to_string(),
                    param: "messages".into(),
                    code,
                }));
            continue;
        }
        timing.prepare_seconds = prepare_started.elapsed().as_secs_f64();
        if let Some(metadata) = &mut metadata {
            metadata["max_tokens"] = json!(job.request.max_tokens);
        }
        if job.events.blocking_send(WorkerEvent::Ready).is_err() {
            continue;
        }
        let mut processor = OutputProcessor::new(
            job.stop,
            job.request.tools.clone(),
            format!("call_{}", job.id),
            job.request.enable_thinking,
        );
        let buffered = job.response_format == ResponseFormat::JsonObject;
        let mut processing_error = None;
        let result = {
            let mut on_text = |part: &str| -> bool {
                if job.cancelled.load(Ordering::Relaxed) || job.events.is_closed() {
                    return false;
                }
                if part.is_empty() {
                    return !processor.stopped() && processing_error.is_none();
                }
                timing
                    .first_model_text_seconds
                    .get_or_insert_with(|| job.queued_at.elapsed().as_secs_f64());
                match processor.push(part) {
                    Ok(deltas) => {
                        if !buffered {
                            for delta in deltas {
                                timing.ready(&delta, job.queued_at);
                                if job.events.blocking_send(WorkerEvent::Delta(delta)).is_err() {
                                    return false;
                                }
                            }
                        }
                        !processor.stopped()
                    }
                    Err(error) => {
                        processing_error = Some(error.to_string());
                        false
                    }
                }
            };
            engine.generate(&job.request, &mut on_text)
        };
        if job.cancelled.load(Ordering::Relaxed) || job.events.is_closed() {
            continue;
        }
        let result = result.and_then(|mut output| {
            if let Some(error) = processing_error {
                bail!("{error}");
            }
            if !output.text.is_empty() {
                timing
                    .first_model_text_seconds
                    .get_or_insert_with(|| job.queued_at.elapsed().as_secs_f64());
            }
            let tail = processor.finish(&output.text)?;
            if buffered {
                let json: Value = serde_json::from_str(&processor.reply.content).map_err(|e| {
                    anyhow::anyhow!("model did not produce the requested JSON object: {e}")
                })?;
                if !json.is_object() {
                    bail!("model did not produce the requested JSON object");
                }
                if !processor.reply.reasoning.is_empty() {
                    timing.ready(
                        &Delta::Reasoning(processor.reply.reasoning.clone()),
                        job.queued_at,
                    );
                    let _ = job
                        .events
                        .blocking_send(WorkerEvent::Delta(Delta::Reasoning(
                            processor.reply.reasoning.clone(),
                        )));
                }
                timing.ready(
                    &Delta::Content(processor.reply.content.clone()),
                    job.queued_at,
                );
                let _ = job.events.blocking_send(WorkerEvent::Delta(Delta::Content(
                    processor.reply.content.clone(),
                )));
            } else {
                for delta in tail {
                    timing.ready(&delta, job.queued_at);
                    if job.events.blocking_send(WorkerEvent::Delta(delta)).is_err() {
                        bail!("client disconnected");
                    }
                }
            }
            if processor.stopped() {
                output.finish_reason = "stop".into();
            } else if !processor.reply.calls.is_empty() && output.finish_reason == "stop" {
                output.finish_reason = "tool_calls".into();
            }
            output.text = processor.reply.content.clone();
            Ok((output, processor.reply))
        });
        timing.total_seconds = job.queued_at.elapsed().as_secs_f64();
        if let Some(mut metadata) = metadata {
            match &result {
                Ok((output, _)) => {
                    metadata["prompt_tokens"] = json!(output.prompt_tokens);
                    metadata["completion_tokens"] = json!(output.completion_tokens);
                    metadata["finish_reason"] = json!(output.finish_reason);
                    metadata["timings"] = timings(output, &timing);
                }
                Err(_) => {
                    metadata["finish_reason"] = json!("error");
                    metadata["timings"] = json!(timing);
                }
            }
            eprintln!("{metadata}");
        }
        let event = match result {
            Ok((output, reply)) => WorkerEvent::Done(output, reply, timing),
            Err(error) => WorkerEvent::Error(error.to_string()),
        };
        let _ = job.events.blocking_send(event);
    }
}

async fn models(State(state): State<ServerState>) -> Json<Value> {
    Json(json!({"object":"list","data":[{"id":state.model,"object":"model","owned_by":"local"}]}))
}

fn error_body(message: impl Into<String>, kind: &str, param: Option<&str>, code: &str) -> Value {
    json!({"error":{"message":message.into(),"type":kind,"param":param,"code":code}})
}
fn api_error(status: StatusCode, message: impl Into<String>) -> Response {
    let kind = if status.is_server_error() {
        "server_error"
    } else {
        "invalid_request_error"
    };
    (
        status,
        Json(error_body(
            message,
            kind,
            None,
            if status.is_server_error() {
                "generation_error"
            } else {
                "invalid_request"
            },
        )),
    )
        .into_response()
}
fn request_error(error: RequestError) -> Response {
    let status = if error.code == "model_not_found" {
        StatusCode::NOT_FOUND
    } else {
        StatusCode::BAD_REQUEST
    };
    (
        status,
        Json(error_body(
            error.message,
            "invalid_request_error",
            Some(&error.param),
            error.code,
        )),
    )
        .into_response()
}

struct ChunkEncoder {
    id: String,
    model: String,
    created: u64,
    options: StreamOptions,
    service_tier: bool,
    random: Option<std::fs::File>,
}
impl ChunkEncoder {
    fn new(id: String, model: String, options: StreamOptions, service_tier: bool) -> Result<Self> {
        Ok(Self {
            id,
            model,
            created: now(),
            options,
            service_tier,
            random: if options.include_obfuscation {
                Some(std::fs::File::open("/dev/urandom")?)
            } else {
                None
            },
        })
    }
    fn event(
        &mut self,
        choices: Value,
        usage: Option<Value>,
        timings: Option<Value>,
    ) -> Result<Event> {
        let mut value = json!({"id":self.id,"object":"chat.completion.chunk","created":self.created,"model":self.model,"choices":choices});
        if self.options.include_usage {
            value["usage"] = usage.unwrap_or(Value::Null);
        }
        if let Some(timings) = timings {
            value["timings"] = timings;
        }
        if self.service_tier {
            value["service_tier"] = json!("default");
        }
        if let Some(random) = &mut self.random {
            if value["choices"].as_array().is_some_and(|a| !a.is_empty()) {
                use std::io::Read;
                value["obfuscation"] = json!("");
                let bytes = serde_json::to_vec(&value)?.len();
                let length = bytes.div_ceil(256) * 256 - bytes;
                let mut padding = vec![0u8; length];
                random.read_exact(&mut padding)?;
                const ALPHABET: &[u8; 64] =
                    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
                value["obfuscation"] = Value::String(
                    padding
                        .iter()
                        .map(|b| ALPHABET[(b & 63) as usize] as char)
                        .collect(),
                );
            }
        }
        Ok(Event::default().data(serde_json::to_string(&value)?))
    }
}
fn choice(delta: Value, finish: Option<&str>) -> Value {
    json!([{"index":0,"delta":delta,"finish_reason":finish,"logprobs":null}])
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

async fn completions(
    State(state): State<ServerState>,
    body: std::result::Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let value = match body {
        Ok(Json(v)) => v,
        Err(error) => return api_error(error.status(), error.body_text()),
    };
    let ParsedRequest {
        generation,
        stream,
        stream_options,
        stop,
        response_format,
        service_tier,
    } = match request::parse(value, &state.model) {
        Ok(r) => r,
        Err(error) => return request_error(error),
    };
    if let Some(vocab_size) = state.vocab_size {
        if let Err(error) = generation.sampling.validate(vocab_size) {
            return request_error(RequestError {
                message: error.to_string(),
                param: "logit_bias".into(),
                code: "invalid_value",
            });
        }
    }
    let id = format!(
        "chatcmpl-{}-{}",
        now(),
        COMPLETION_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    );
    let encoder = if stream {
        match ChunkEncoder::new(
            id.clone(),
            state.model.clone(),
            stream_options,
            service_tier,
        ) {
            Ok(e) => Some(e),
            Err(e) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        }
    } else {
        None
    };
    let (events, mut receiver) = async_mpsc::channel(EVENT_CAPACITY);
    let cancelled = Arc::new(AtomicBool::new(false));
    let guard = CancelOnDrop(cancelled.clone());
    if let Err(error) = state.sender.try_send(Job {
        queued_at: Instant::now(),
        request: generation,
        stop,
        response_format,
        id: id.clone(),
        events,
        cancelled,
    }) {
        return match error {
            mpsc::TrySendError::Full(_) => {
                api_error(StatusCode::SERVICE_UNAVAILABLE, "generation queue is full")
            }
            mpsc::TrySendError::Disconnected(_) => {
                api_error(StatusCode::SERVICE_UNAVAILABLE, "generation worker stopped")
            }
        };
    }
    // Validate and tokenize on the model worker before committing successful
    // SSE headers. OpenCode uses the 400 context error to compact long histories.
    match receiver.recv().await {
        Some(WorkerEvent::Ready) => {}
        Some(WorkerEvent::InvalidRequest(error)) => return request_error(error),
        Some(WorkerEvent::Error(message)) => {
            return api_error(StatusCode::INTERNAL_SERVER_ERROR, message);
        }
        _ => {
            return api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "generation worker stopped before request validation",
            );
        }
    }
    if let Some(mut encoder) = encoder {
        let stream = async_stream::stream! {
            let _guard=guard;
            let mut terminal=false;
            let mut tool_index=0;
            match encoder.event(choice(json!({"role":"assistant","content":""}),None),None,None) {
                Ok(event)=>yield Ok::<Event,std::convert::Infallible>(event),
                Err(error)=>{yield Ok(Event::default().data(error_body(error.to_string(),"server_error",None,"stream_error").to_string()));yield Ok(Event::default().data("[DONE]"));return;}
            }
            while let Some(event)=receiver.recv().await {
                let encoded=match event {
                    WorkerEvent::Ready=>continue,
                    WorkerEvent::InvalidRequest(error)=>{terminal=true;yield Ok(Event::default().data(error_body(error.message,"invalid_request_error",Some(&error.param),error.code).to_string()));yield Ok(Event::default().data("[DONE]"));break;}
                    WorkerEvent::Delta(delta)=>{
                        let delta=match delta {
                            Delta::Content(text)=>json!({"content":text}),
                            Delta::Reasoning(text)=>json!({"reasoning_content":text}),
                            Delta::Tool(call)=>{let index=tool_index;tool_index+=1;json!({"tool_calls":[{"index":index,"id":call.id,"type":call.kind,"function":call.function}]})}
                        };
                        encoder.event(choice(delta,None),None,None)
                    }
                    WorkerEvent::Done(output,_,timing)=>{
                        terminal=true;
                        match encoder.event(choice(json!({}),Some(&output.finish_reason)),None,Some(timings(&output,&timing))) {
                            Ok(event)=>yield Ok(event),Err(error)=>{yield Ok(Event::default().data(error_body(error.to_string(),"server_error",None,"stream_error").to_string()));yield Ok(Event::default().data("[DONE]"));break;}
                        }
                        if stream_options.include_usage {
                            match encoder.event(json!([]),Some(usage(&output)),None) {
                                Ok(event)=>yield Ok(event),Err(error)=>yield Ok(Event::default().data(error_body(error.to_string(),"server_error",None,"stream_error").to_string())),
                            }
                        }
                        yield Ok(Event::default().data("[DONE]"));break;
                    }
                    WorkerEvent::Error(message)=>{terminal=true;yield Ok(Event::default().data(error_body(message,"server_error",None,"generation_error").to_string()));yield Ok(Event::default().data("[DONE]"));break;}
                };
                match encoded {Ok(event)=>yield Ok(event),Err(error)=>{terminal=true;yield Ok(Event::default().data(error_body(error.to_string(),"server_error",None,"stream_error").to_string()));yield Ok(Event::default().data("[DONE]"));break;}}
            }
            if !terminal {yield Ok(Event::default().data(error_body("generation worker stopped","server_error",None,"generation_error").to_string()));yield Ok(Event::default().data("[DONE]"));}
        };
        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let _guard = guard;
        while let Some(event) = receiver.recv().await {
            match event {
                WorkerEvent::Ready => {}
                WorkerEvent::InvalidRequest(error) => return request_error(error),
                WorkerEvent::Delta(_) => {}
                WorkerEvent::Error(message) => {
                    return api_error(StatusCode::INTERNAL_SERVER_ERROR, message);
                }
                WorkerEvent::Done(output, reply, timing) => {
                    let mut message = json!({"role":"assistant","content":reply.content});
                    if !reply.reasoning.is_empty() {
                        message["reasoning_content"] = json!(reply.reasoning);
                    }
                    if !reply.calls.is_empty() {
                        message["tool_calls"] = json!(reply.calls);
                        if reply.content.is_empty() {
                            message["content"] = Value::Null;
                        }
                    }
                    let mut value = json!({"id":id,"object":"chat.completion","created":now(),"model":state.model,"choices":[{"index":0,"message":message,"finish_reason":output.finish_reason,"logprobs":null}],"usage":usage(&output),"timings":timings(&output,&timing)});
                    if service_tier {
                        value["service_tier"] = json!("default");
                    }
                    return Json(value).into_response();
                }
            }
        }
        api_error(StatusCode::SERVICE_UNAVAILABLE, "generation worker stopped")
    }
}

fn usage(output: &GenerationOutput) -> Value {
    json!({"prompt_tokens":output.prompt_tokens,"completion_tokens":output.completion_tokens,"total_tokens":output.prompt_tokens + output.completion_tokens})
}

fn timings(output: &GenerationOutput, timing: &RequestTiming) -> Value {
    let mut result = json!(timing);
    result["prefill_seconds"] = json!(output.prefill_seconds);
    result["decode_seconds"] = json!(output.decode_seconds);
    result
}

#[cfg(test)]
mod tests;
