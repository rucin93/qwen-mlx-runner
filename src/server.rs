use std::net::SocketAddr;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc::{self, SyncSender},
};
use std::time::{SystemTime, UNIX_EPOCH};

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

const QUEUE_CAPACITY: usize = 4;
const EVENT_CAPACITY: usize = 16;
const MAX_TOKENS: usize = 4096;
static COMPLETION_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
struct ServerState {
    sender: SyncSender<Job>,
    model: String,
}

struct Job {
    request: GenerationRequest,
    events: async_mpsc::Sender<WorkerEvent>,
    cancelled: Arc<AtomicBool>,
}

enum WorkerEvent {
    Text(String),
    Done(GenerationOutput),
    Error(String),
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
    let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
    std::thread::Builder::new()
        .name("qwen-generation".into())
        .spawn(move || worker(engine, receiver))?;
    let app = router(ServerState { sender, model });
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
    while let Ok(job) = receiver.recv() {
        if job.cancelled.load(Ordering::Relaxed) {
            continue;
        }
        let mut on_text = |part: &str| -> bool {
            if job.cancelled.load(Ordering::Relaxed) {
                return false;
            }
            if part.is_empty() {
                return !job.events.is_closed();
            }
            job.events
                .blocking_send(WorkerEvent::Text(part.to_owned()))
                .is_ok()
        };
        let result = engine.generate(&job.request, &mut on_text);
        if job.cancelled.load(Ordering::Relaxed) {
            continue;
        }
        let event = match result {
            Ok(output) => WorkerEvent::Done(output),
            Err(error) => WorkerEvent::Error(error.to_string()),
        };
        let _ = job.events.blocking_send(event);
    }
}

async fn models(State(state): State<ServerState>) -> Json<Value> {
    Json(json!({"object":"list","data":[{"id":state.model,"object":"model","owned_by":"local"}]}))
}

fn api_error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({"error":{"message":message.into(),"type":"invalid_request_error"}})),
    )
        .into_response()
}

fn parse_request(
    value: Value,
    model_id: &str,
) -> std::result::Result<(GenerationRequest, bool), String> {
    let object = value.as_object().ok_or("request must be a JSON object")?;
    for key in object.keys() {
        if !matches!(
            key.as_str(),
            "model"
                | "messages"
                | "max_tokens"
                | "max_completion_tokens"
                | "temperature"
                | "top_p"
                | "top_k"
                | "seed"
                | "stream"
                | "enable_thinking"
        ) {
            return Err(format!("unsupported request field: {key}"));
        }
    }
    let model = object
        .get("model")
        .and_then(Value::as_str)
        .ok_or("model must be a string")?;
    if model != model_id {
        return Err(format!("unknown model: {model}"));
    }
    let raw_messages = object
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("messages must be an array")?;
    if raw_messages.is_empty() {
        return Err("messages must not be empty".into());
    }
    let mut messages = Vec::with_capacity(raw_messages.len());
    for (index, item) in raw_messages.iter().enumerate() {
        let item = item
            .as_object()
            .ok_or_else(|| format!("messages[{index}] must be an object"))?;
        for key in item.keys() {
            if !matches!(key.as_str(), "role" | "content") {
                return Err(format!("unsupported messages[{index}] field: {key}"));
            }
        }
        let role = item
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("messages[{index}].role must be a string"))?;
        if !matches!(role, "system" | "user" | "assistant") {
            return Err(format!("unsupported role: {role}"));
        }
        let content = item
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("messages[{index}].content must be text"))?;
        messages.push(Message {
            role: role.to_owned(),
            content: content.to_owned(),
        });
    }
    if object.contains_key("max_tokens") && object.contains_key("max_completion_tokens") {
        return Err("set only one of max_tokens or max_completion_tokens".into());
    }
    let max_tokens = match object
        .get("max_tokens")
        .or_else(|| object.get("max_completion_tokens"))
    {
        None => 256,
        Some(v) => usize::try_from(v.as_u64().ok_or("max_tokens must be a positive integer")?)
            .map_err(|_| "max_tokens is too large")?,
    };
    if max_tokens == 0 || max_tokens > MAX_TOKENS {
        return Err(format!("max_tokens must be in 1..={MAX_TOKENS}"));
    }
    let float = |key: &str, default: f64| -> std::result::Result<f32, String> {
        let value = match object.get(key) {
            Some(v) => v
                .as_f64()
                .ok_or_else(|| format!("{key} must be a number"))?,
            None => default,
        };
        if !value.is_finite() || value > f32::MAX as f64 || value < -(f32::MAX as f64) {
            return Err(format!("{key} must be finite"));
        }
        Ok(value as f32)
    };
    let temperature = float("temperature", 1.0)?;
    let top_p = float("top_p", 1.0)?;
    if temperature < 0.0 || temperature > 2.0 {
        return Err("temperature must be in [0, 2]".into());
    }
    if top_p <= 0.0 || top_p > 1.0 {
        return Err("top_p must be in (0, 1]".into());
    }
    let top_k = match object.get("top_k") {
        Some(v) => usize::try_from(v.as_u64().ok_or("top_k must be a nonnegative integer")?)
            .map_err(|_| "top_k is too large")?,
        None => 0,
    };
    let seed = match object.get("seed") {
        Some(v) => v.as_u64().ok_or("seed must be a nonnegative integer")?,
        None => 0,
    };
    let stream = match object.get("stream") {
        Some(v) => v.as_bool().ok_or("stream must be a boolean")?,
        None => false,
    };
    let enable_thinking = match object.get("enable_thinking") {
        Some(v) => v.as_bool().ok_or("enable_thinking must be a boolean")?,
        None => false,
    };
    Ok((
        GenerationRequest {
            messages,
            max_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            enable_thinking,
        },
        stream,
    ))
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

async fn completions(State(state): State<ServerState>, Json(value): Json<Value>) -> Response {
    let (request, stream) = match parse_request(value, &state.model) {
        Ok(parsed) => parsed,
        Err(message) => return api_error(StatusCode::BAD_REQUEST, message),
    };
    let enable_thinking = request.enable_thinking;
    let (events, receiver) = async_mpsc::channel(EVENT_CAPACITY);
    let cancelled = Arc::new(AtomicBool::new(false));
    let guard = CancelOnDrop(cancelled.clone());
    if let Err(error) = state.sender.try_send(Job {
        request,
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
    let id = format!(
        "chatcmpl-{}-{}",
        now(),
        COMPLETION_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    );
    if stream {
        let model = state.model;
        let created = now();
        let mut receiver = receiver;
        let stream = async_stream::stream! {
            let _guard = guard;
            let mut terminal = false;
            yield Ok::<Event, std::convert::Infallible>(Event::default().data(json!({"id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}).to_string()));
            if enable_thinking {
                // The checkpoint prompt ends with this opening tag. Restore it in visible output.
                yield Ok(Event::default().data(json!({"id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[{"index":0,"delta":{"content":"<think>\n"},"finish_reason":null}]}).to_string()));
            }
            while let Some(event) = receiver.recv().await {
                match event {
                    WorkerEvent::Text(text) => yield Ok(Event::default().data(json!({"id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[{"index":0,"delta":{"content":text},"finish_reason":null}]}).to_string())),
                    WorkerEvent::Done(output) => {
                        terminal = true;
                        yield Ok(Event::default().data(json!({"id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[{"index":0,"delta":{},"finish_reason":output.finish_reason}],"usage":usage(&output),"timings":timings(&output)}).to_string()));
                        yield Ok(Event::default().data("[DONE]"));
                        break;
                    }
                    WorkerEvent::Error(message) => {
                        terminal = true;
                        yield Ok(Event::default().data(json!({"error":{"message":message,"type":"generation_error"}}).to_string()));
                        yield Ok(Event::default().data("[DONE]"));
                        break;
                    }
                }
            }
            if !terminal {
                yield Ok(Event::default().data(json!({"error":{"message":"generation worker stopped","type":"generation_error"}}).to_string()));
                yield Ok(Event::default().data("[DONE]"));
            }
        };
        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let _guard = guard;
        let mut receiver = receiver;
        let mut streamed = String::new();
        while let Some(event) = receiver.recv().await {
            match event {
                WorkerEvent::Text(part) => streamed.push_str(&part),
                WorkerEvent::Error(message) => {
                    return api_error(StatusCode::INTERNAL_SERVER_ERROR, message);
                }
                WorkerEvent::Done(output) => {
                    // The engine's final text is authoritative; callbacks may deliver partial UTF-8-safe prefixes.
                    let _ = streamed;
                    let content = if enable_thinking {
                        format!("<think>\n{}", output.text)
                    } else {
                        output.text.clone()
                    };
                    return Json(json!({"id":id,"object":"chat.completion","created":now(),"model":state.model,"choices":[{"index":0,"message":{"role":"assistant","content":content},"finish_reason":output.finish_reason}],"usage":usage(&output),"timings":timings(&output)})).into_response();
                }
            }
        }
        api_error(StatusCode::SERVICE_UNAVAILABLE, "generation worker stopped")
    }
}

fn usage(output: &GenerationOutput) -> Value {
    json!({"prompt_tokens":output.prompt_tokens,"completion_tokens":output.completion_tokens,"total_tokens":output.prompt_tokens + output.completion_tokens})
}

fn timings(output: &GenerationOutput) -> Value {
    json!({"prefill_seconds":output.prefill_seconds,"decode_seconds":output.decode_seconds})
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::{Request, header},
    };
    use tower::ServiceExt;

    struct TestEngine;
    impl TextGenerator for TestEngine {
        fn model_id(&self) -> &str {
            "test-model"
        }
        fn generate(
            &mut self,
            _: &GenerationRequest,
            on_text: &mut dyn FnMut(&str) -> bool,
        ) -> Result<GenerationOutput> {
            if !on_text("") {
                bail!("cancelled");
            }
            if !on_text("hello") {
                bail!("cancelled");
            }
            Ok(GenerationOutput {
                text: "hello".into(),
                prompt_tokens: 2,
                completion_tokens: 1,
                finish_reason: "stop".into(),
                prefill_seconds: 0.1,
                decode_seconds: 0.2,
            })
        }
    }

    fn app() -> Router {
        let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
        std::thread::spawn(move || worker(Box::new(TestEngine), receiver));
        router(ServerState {
            sender,
            model: "test-model".into(),
        })
    }

    fn body(stream: bool) -> String {
        json!({"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":stream})
            .to_string()
    }

    #[tokio::test]
    async fn non_stream_returns_engine_text_and_usage() {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body(false)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 100_000).await.unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["choices"][0]["message"]["content"], "hello");
        assert_eq!(value["usage"]["total_tokens"], 3);
    }

    #[tokio::test]
    async fn stream_has_delta_finish_and_done() {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body(true)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 100_000).await.unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("hello"));
        assert!(text.contains("\"finish_reason\":\"stop\""));
        assert!(text.contains("[DONE]"));
    }

    #[tokio::test]
    async fn rejects_unsupported_payloads() {
        for payload in [
            json!({"model":"test-model","messages":[{"role":"user","content":[{"type":"image_url"}]}]}),
            json!({"model":"test-model","messages":[{"role":"user","content":"hi"}],"tools":[]}),
            json!({"model":"test-model","messages":[{"role":"user","content":"hi"}],"n":2}),
        ] {
            let response = app()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/chat/completions")
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(Body::from(payload.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
    }

    #[tokio::test]
    async fn thinking_output_has_opening_boundary() {
        let payload = json!({"model":"test-model","messages":[{"role":"user","content":"hi"}],"enable_thinking":true});
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = to_bytes(response.into_body(), 100_000).await.unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["choices"][0]["message"]["content"], "<think>\nhello");
    }

    #[test]
    fn dropped_client_cancels_generation_at_empty_poll() {
        struct ProbeEngine(Arc<AtomicBool>);
        impl TextGenerator for ProbeEngine {
            fn model_id(&self) -> &str {
                "probe"
            }
            fn generate(
                &mut self,
                _: &GenerationRequest,
                on_text: &mut dyn FnMut(&str) -> bool,
            ) -> Result<GenerationOutput> {
                self.0.store(!on_text(""), Ordering::Relaxed);
                bail!("cancelled")
            }
        }
        let observed = Arc::new(AtomicBool::new(false));
        let (sender, receiver) = mpsc::sync_channel(1);
        let handle = std::thread::spawn({
            let observed = observed.clone();
            move || worker(Box::new(ProbeEngine(observed)), receiver)
        });
        let (events, client) = async_mpsc::channel(1);
        drop(client);
        sender
            .send(Job {
                request: GenerationRequest {
                    messages: vec![Message {
                        role: "user".into(),
                        content: "hi".into(),
                    }],
                    max_tokens: 1,
                    temperature: 0.0,
                    top_p: 1.0,
                    top_k: 0,
                    seed: 0,
                    enable_thinking: false,
                },
                events,
                cancelled: Arc::new(AtomicBool::new(false)),
            })
            .unwrap();
        drop(sender);
        handle.join().unwrap();
        assert!(observed.load(Ordering::Relaxed));
    }
}
