use super::*;
use axum::{
    body::{Body, to_bytes},
    http::{Request, header},
};
use tower::ServiceExt;

struct TestEngine;
impl TextGenerator for TestEngine {
    fn prepare_request(&self, _: &mut GenerationRequest) -> Result<()> {
        // Fixed scripted output; this fixture never allocates from the budget.
        Ok(())
    }
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
        vocab_size: Some(1000),
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
async fn opencode_stream_options_returns_separate_usage_chunk() {
    let payload = json!({"model":"test-model","messages":[{"role":"user","content":"hi"}],
            "stream":true,"stream_options":{"include_usage":true,"include_obfuscation":false}});
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
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 100_000).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let chunks: Vec<Value> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    let last = chunks.last().unwrap();
    assert_eq!(last["choices"], json!([]));
    assert_eq!(last["usage"]["total_tokens"], 3);
    assert!(
        chunks
            .iter()
            .all(|chunk| chunk.get("obfuscation").is_none())
    );
    assert!(
        chunks[..chunks.len() - 1]
            .iter()
            .all(|chunk| chunk["usage"].is_null())
    );
}

#[tokio::test]
async fn default_stream_obfuscation_pads_choices_without_changing_content() {
    let payload = json!({"model":"test-model","messages":[{"role":"user","content":"hi"}],
        "stream":true,"stream_options":{"include_usage":true}});
    let (status, body) = post(app(), payload).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let mut content = String::new();
    for data in body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
    {
        let chunk: Value = serde_json::from_str(data).unwrap();
        if chunk["choices"].as_array().unwrap().is_empty() {
            assert!(chunk.get("obfuscation").is_none());
        } else {
            assert_eq!(data.len() % 256, 0);
            let padding = chunk["obfuscation"].as_str().unwrap();
            assert!(
                padding
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
            );
            if let Some(text) = chunk["choices"][0]["delta"]["content"].as_str() {
                content.push_str(text);
            }
        }
    }
    assert_eq!(content, "hello");
}

#[tokio::test]
async fn rejects_unsupported_payloads() {
    for payload in [
        json!({"model":"test-model","messages":[{"role":"user","content":[{"type":"image_url"}]}]}),
        json!({"model":"test-model","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"custom"}]}),
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
    assert_eq!(value["choices"][0]["message"]["reasoning_content"], "hello");
    assert_eq!(value["choices"][0]["message"]["content"], "");
}

#[test]
fn dropped_client_cancels_generation_at_empty_poll() {
    struct ProbeEngine(Arc<AtomicBool>, Arc<std::sync::Barrier>);
    impl TextGenerator for ProbeEngine {
        fn prepare_request(&self, _: &mut GenerationRequest) -> Result<()> {
            Ok(())
        }
        fn model_id(&self) -> &str {
            "probe"
        }
        fn generate(
            &mut self,
            _: &GenerationRequest,
            on_text: &mut dyn FnMut(&str) -> bool,
        ) -> Result<GenerationOutput> {
            self.1.wait();
            self.1.wait();
            self.0.store(!on_text(""), Ordering::Relaxed);
            bail!("cancelled")
        }
    }
    let observed = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let (sender, receiver) = mpsc::sync_channel(1);
    let handle = std::thread::spawn({
        let observed = observed.clone();
        let barrier = barrier.clone();
        move || worker(Box::new(ProbeEngine(observed, barrier)), receiver)
    });
    let (events, client) = async_mpsc::channel(1);
    sender
        .send(Job {
            request: GenerationRequest {
                messages: vec![Message {
                    role: "user".into(),
                    content: "hi".into(),
                    ..Default::default()
                }],
                max_tokens: 1,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                seed: 0,
                enable_thinking: false,
                ..Default::default()
            },
            stop: vec![],
            response_format: ResponseFormat::Text,
            id: "test".into(),
            events,
            cancelled: Arc::new(AtomicBool::new(false)),
        })
        .unwrap();
    barrier.wait();
    drop(client);
    barrier.wait();
    drop(sender);
    handle.join().unwrap();
    assert!(observed.load(Ordering::Relaxed));
}

struct ScriptEngine {
    text: String,
    inspect: Option<std::sync::mpsc::Sender<GenerationRequest>>,
}
impl TextGenerator for ScriptEngine {
    fn prepare_request(&self, _: &mut GenerationRequest) -> Result<()> {
        Ok(())
    }
    fn model_id(&self) -> &str {
        "test-model"
    }
    fn vocab_size(&self) -> Option<usize> {
        Some(1000)
    }
    fn generate(
        &mut self,
        r: &GenerationRequest,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<GenerationOutput> {
        if let Some(inspect) = &self.inspect {
            let _ = inspect.send(r.clone());
        }
        let mut text = String::new();
        let mut finish = "stop";
        for c in self.text.chars() {
            if !on_text("") {
                finish = "cancelled";
                break;
            }
            let part = c.to_string();
            if !on_text(&part) {
                finish = "cancelled";
                break;
            }
            text.push(c);
        }
        Ok(GenerationOutput {
            text,
            prompt_tokens: 12,
            completion_tokens: 7,
            finish_reason: finish.into(),
            prefill_seconds: 0.0,
            decode_seconds: 0.01,
        })
    }
}
fn script_app(text: &str) -> (Router, std::sync::mpsc::Receiver<GenerationRequest>) {
    let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
    let (inspect, requests) = mpsc::channel();
    let engine = ScriptEngine {
        text: text.into(),
        inspect: Some(inspect),
    };
    std::thread::spawn(move || worker(Box::new(engine), receiver));
    (
        router(ServerState {
            sender,
            model: "test-model".into(),
            vocab_size: Some(1000),
        }),
        requests,
    )
}
fn tool_definition() -> Value {
    json!({"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}})
}
fn tool_payload(stream: bool) -> Value {
    json!({"model":"test-model","messages":[{"role":"system","content":"Be helpful."},{"role":"system","content":"Use tools."},{"role":"user","content":[{"type":"text","text":"Read "},{"type":"text","text":"README.md"}]}],"tools":[tool_definition()],"tool_choice":"auto","stream":stream,"max_tokens":2048})
}
async fn post(app: Router, payload: Value) -> (StatusCode, String) {
    let response = app
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
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1_000_000).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}
fn chunks(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|l| *l != "[DONE]")
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[tokio::test]
async fn opencode_tool_call_stream_has_complete_function_delta_and_usage() {
    let (app, requests) = script_app(
        "<tool_call>\n<function=read_file>\n<parameter=path>README.md</parameter>\n</function>\n</tool_call>",
    );
    let mut payload = tool_payload(true);
    payload["stream_options"] = json!({"include_usage":true,"include_obfuscation":false});
    let (status, body) = post(app, payload).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let request = requests.recv().unwrap();
    assert_eq!(request.messages[2].content, "Read README.md");
    assert!(request.tools.enabled());
    let c = chunks(&body);
    assert!(c.iter().all(|x| x.get("error").is_none()), "{body}");
    let calls: Vec<_> = c
        .iter()
        .filter_map(|x| x["choices"][0]["delta"]["tool_calls"].as_array())
        .flatten()
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["index"], 0);
    assert!(calls[0]["id"].as_str().unwrap().starts_with("call_"));
    assert_eq!(calls[0]["function"]["name"], "read_file");
    assert_eq!(
        serde_json::from_str::<Value>(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap(),
        json!({"path":"README.md"})
    );
    assert_eq!(c[c.len() - 2]["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(c.last().unwrap()["choices"], json!([]));
    assert_eq!(c.last().unwrap()["usage"]["total_tokens"], 19);
    assert!(
        c[..c.len() - 1]
            .iter()
            .all(|x| x.get("usage") == Some(&Value::Null))
    );
    assert_eq!(body.matches("data: [DONE]").count(), 1);
    assert!(!body.contains("<tool_call>"));
}

#[tokio::test]
async fn opencode_tool_round_trip_preserves_id_and_arguments() {
    let (app, _) = script_app(
        "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"README.md\"}}</tool_call>",
    );
    let (status, body) = post(app, tool_payload(false)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).unwrap();
    let message = &value["choices"][0]["message"];
    assert!(message["content"].is_null());
    assert_eq!(value["choices"][0]["finish_reason"], "tool_calls");
    let id = message["tool_calls"][0]["id"].as_str().unwrap();
    let mut payload = tool_payload(false);
    payload["messages"]
        .as_array_mut()
        .unwrap()
        .push(message.clone());
    payload["messages"]
        .as_array_mut()
        .unwrap()
        .push(json!({"role":"tool","tool_call_id":id,"content":"Project documentation."}));
    let (app, requests) = script_app("The file describes the project.");
    let (status, body) = post(app, payload).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let request = requests.recv().unwrap();
    assert_eq!(request.messages[3].tool_calls[0].id, id);
    assert_eq!(request.messages[4].tool_call_id.as_deref(), Some(id));
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["choices"][0]["message"]["content"],
        "The file describes the project."
    );
}

#[tokio::test]
async fn stop_across_callbacks_omits_stop_suffix_and_keeps_real_token_usage() {
    for stream in [false, true] {
        let (app, _) = script_app("Answer 🛑END hidden");
        let payload = json!({"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":stream,"stop":"🛑END"});
        let (status, body) = post(app, payload).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        if stream {
            let c = chunks(&body);
            let text = c
                .iter()
                .filter_map(|v| v["choices"][0]["delta"]["content"].as_str())
                .collect::<String>();
            assert_eq!(text, "Answer ");
            assert_eq!(c.last().unwrap()["choices"][0]["finish_reason"], "stop");
            assert!(c.iter().all(|v| v.get("usage").is_none()));
        } else {
            let v: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(v["choices"][0]["message"]["content"], "Answer ");
            assert_eq!(v["choices"][0]["finish_reason"], "stop");
            assert_eq!(v["usage"]["completion_tokens"], 7);
        }
    }
}

#[tokio::test]
async fn required_and_named_tools_report_generation_errors_on_model_violation() {
    for choice in [
        json!("required"),
        json!({"type":"function","function":{"name":"read_file"}}),
    ] {
        for stream in [false, true] {
            let (app, _) = script_app("I will answer without calling a tool.");
            let mut payload = tool_payload(stream);
            payload["tool_choice"] = choice.clone();
            let (status, body) = post(app, payload).await;
            if stream {
                assert_eq!(status, StatusCode::OK);
                assert!(
                    chunks(&body)
                        .iter()
                        .any(|v| v["error"]["code"] == "generation_error")
                );
                assert!(!body.contains("\"finish_reason\":\"tool_calls\""));
            } else {
                assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
                assert_eq!(
                    serde_json::from_str::<Value>(&body).unwrap()["error"]["code"],
                    "generation_error"
                );
            }
        }
    }
}

#[tokio::test]
async fn malformed_tool_output_is_not_exposed_as_an_executable_call() {
    for text in [
        "<tool_call><function=read_file>",
        "<tool_call>{\"name\":\"unknown\",\"arguments\":{}}</tool_call>",
    ] {
        let (app, _) = script_app(text);
        let (status, body) = post(app, tool_payload(true)).await;
        assert_eq!(status, StatusCode::OK);
        let c = chunks(&body);
        assert!(c.iter().any(|v| v.get("error").is_some()));
        assert!(
            c.iter()
                .all(|v| v["choices"][0]["delta"].get("tool_calls").is_none())
        );
    }
}

#[tokio::test]
async fn explicit_token_budgets_above_4096_are_accepted() {
    for field in ["max_tokens", "max_completion_tokens"] {
        let (app, requests) = script_app("okay");
        let mut payload = json!({"model":"test-model","messages":[{"role":"user","content":"hi"}]});
        payload[field] = json!(8192);
        let (status, body) = post(app, payload).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(requests.recv().unwrap().max_tokens, 8192);
    }
}

#[tokio::test]
#[ignore = "requires a real Apple Metal GPU"]
async fn uncapped_http_output_uses_remaining_context_in_both_engines() {
    use crate::{engine::ChatEngine, mtp_chat::MtpChatEngine};
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let target = tempfile::tempdir().unwrap();
    for file in ["model.safetensors", "tokenizer.json", "chat_template.jinja"] {
        std::fs::copy(
            fixtures.join("tiny-q4").join(file),
            target.path().join(file),
        )
        .unwrap();
    }
    let mut config: Value =
        serde_json::from_slice(&std::fs::read(fixtures.join("tiny-q4/config.json")).unwrap())
            .unwrap();
    config["text_config"]["max_position_embeddings"] = json!(520);
    std::fs::write(target.path().join("config.json"), config.to_string()).unwrap();
    for mtp in [false, true] {
        let engine: Box<dyn TextGenerator> = if mtp {
            Box::new(
                MtpChatEngine::load(target.path(), &fixtures.join("tiny-mtp"), 520, 3).unwrap(),
            )
        } else {
            Box::new(ChatEngine::load(target.path(), 520).unwrap())
        };
        let model = engine.model_id().to_owned();
        let vocab_size = engine.vocab_size();
        let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
        std::thread::spawn(move || worker(engine, receiver));
        let app = router(ServerState {
            sender,
            model: model.clone(),
            vocab_size,
        });
        // The real tiny tokenizer renders four tokens: user w6 w7 assistant.
        // A strong bias avoids EOS, so output must reach the actual context boundary.
        for (budget, expected) in [
            (None, 516),
            (Some(json!(null)), 516),
            (Some(json!(8192)), 516),
            (Some(json!(u64::MAX)), 516),
            (Some(json!(8)), 8),
        ] {
            let mut payload = json!({"model":model,"messages":[{"role":"user","content":"w6 w7"}],"temperature":0,"logit_bias":{"6":100}});
            if let Some(budget) = budget {
                payload["max_completion_tokens"] = budget;
            }
            let (status, body) = post(app.clone(), payload).await;
            assert_eq!(status, StatusCode::OK, "mtp={mtp}: {body}");
            let response: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(response["usage"]["prompt_tokens"], 4);
            assert_eq!(
                response["usage"]["completion_tokens"], expected,
                "mtp={mtp}: {body}"
            );
            assert_eq!(response["choices"][0]["finish_reason"], "length");
        }
        for stream in [false, true] {
            let payload = json!({"model":model,"messages":[{"role":"user","content":"w6 ".repeat(518)}],"stream":stream});
            let (status, body) = post(app.clone(), payload).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "mtp={mtp}: {body}");
            assert_eq!(
                serde_json::from_str::<Value>(&body).unwrap()["error"]["code"],
                "context_length_exceeded"
            );
        }
    }
}

#[tokio::test]
async fn optional_defaults_and_sampling_controls_are_forwarded() {
    let (app, requests) = script_app("okay");
    let payload = json!({"model":"test-model","messages":[{"role":"developer","content":"Instructions"},{"role":"user","content":"Hi"}],"max_tokens":null,"max_completion_tokens":24,"temperature":null,"top_p":null,"stream":null,"stream_options":null,"n":1,"store":false,"logprobs":false,"top_logprobs":0,"response_format":{"type":"text"},"modalities":["text"],"frequency_penalty":1.5,"presence_penalty":-0.5,"logit_bias":{"23":40},"seed":-1,"user":"local","metadata":{"project":"demo"},"service_tier":"auto","reasoning_effort":"low"});
    let (status, body) = post(app, payload).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let r = requests.recv().unwrap();
    assert_eq!(r.max_tokens, 24);
    assert_eq!(r.sampling.frequency_penalty, 1.5);
    assert_eq!(r.sampling.presence_penalty, -0.5);
    assert_eq!(r.sampling.logit_bias.get(&23), Some(&40.0));
    assert_eq!(r.seed, u64::MAX);
    assert!(r.enable_thinking);
    assert_eq!(r.reasoning_effort.as_deref(), Some("low"));
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["service_tier"],
        "default"
    );
}

#[tokio::test]
async fn invalid_schema_options_fail_before_generation_with_parameter_errors() {
    let baseline = json!({"model":"test-model","messages":[{"role":"user","content":"hi"}]});
    for (field, value) in [
        ("max_tokens", json!(0)),
        ("max_completion_tokens", json!(-1)),
        ("max_tokens", json!(1.5)),
        ("max_tokens", json!("8192")),
        ("tool_choice", json!("required")),
        ("stream_options", json!({"include_usage":true})),
        ("frequency_penalty", json!(2.1)),
        ("logit_bias", json!({"1000":1})),
        ("n", json!(2)),
        ("stop", json!(["a", "b", "c", "d", "e"])),
        ("logprobs", json!(true)),
        ("response_format", json!({"type":"json_schema"})),
        ("store", json!(true)),
        ("reasoning_effort", json!("turbo")),
    ] {
        let (app, requests) = script_app("never");
        let mut payload = baseline.clone();
        payload[field] = value;
        let (status, body) = post(app, payload).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{field}: {body}");
        let e: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(e["error"]["type"], "invalid_request_error");
        assert!(e["error"]["param"].is_string());
        assert!(requests.try_recv().is_err());
    }
}

#[tokio::test]
async fn json_object_is_validated_before_streaming_content() {
    for stream in [false, true] {
        let (app, requests) = script_app("{\"city\":\"Łódź\"}");
        let payload = json!({"model":"test-model","messages":[{"role":"user","content":"JSON"}],"response_format":{"type":"json_object"},"stream":stream});
        let (status, body) = post(app, payload.clone()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(
            requests.recv().unwrap().messages[0]
                .content
                .contains("JSON object")
        );
        if stream {
            let c = chunks(&body);
            let text = c
                .iter()
                .filter_map(|v| v["choices"][0]["delta"]["content"].as_str())
                .collect::<String>();
            assert!(serde_json::from_str::<Value>(&text).unwrap().is_object());
        }
        let (app, _) = script_app("invalid json");
        let (status, body) = post(app, payload).await;
        if stream {
            let c = chunks(&body);
            assert!(c.iter().any(|v| v.get("error").is_some()));
            assert!(!body.contains("invalid json"));
        } else {
            assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        }
    }
}

#[tokio::test]
async fn malformed_json_has_openai_error_envelope() {
    let response = app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from("{"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), 100_000).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["error"]["message"].is_string());
    assert!(v["error"].get("param").is_some());
    assert!(v["error"].get("code").is_some());
}

#[tokio::test]
async fn json_and_plain_content_keep_literal_think_tags() {
    for format in [json!({"type":"text"}), json!({"type":"json_object"})] {
        let text = "{\"text\":\"<think>literal</think>\"}";
        let (app, _) = script_app(text);
        let payload = json!({"model":"test-model","messages":[{"role":"user","content":"literal tags"}],"response_format":format});
        let (status, body) = post(app, payload).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let value: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["choices"][0]["message"]["content"], text);
        assert!(
            value["choices"][0]["message"]
                .get("reasoning_content")
                .is_none()
        );
    }
    let (app, _) = script_app("actual thought</think>{\"text\":\"<think>literal</think>\"}");
    let(status,body)=post(app,json!({"model":"test-model","messages":[{"role":"user","content":"test"}],"enable_thinking":true,"response_format":{"type":"json_object"}})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        value["choices"][0]["message"]["content"],
        "{\"text\":\"<think>literal</think>\"}"
    );
    assert_eq!(
        value["choices"][0]["message"]["reasoning_content"],
        "actual thought"
    );
}

#[tokio::test]
async fn context_overflow_is_400_before_streaming_or_generation() {
    struct ContextEngine;
    impl TextGenerator for ContextEngine {
        fn model_id(&self) -> &str {
            "test-model"
        }
        fn prepare_request(&self, _: &mut GenerationRequest) -> Result<()> {
            Err(crate::chat::ContextLengthExceeded {
                prompt_tokens: 8192,
                capacity: 8192,
            }
            .into())
        }
        fn generate(
            &mut self,
            _: &GenerationRequest,
            _: &mut dyn FnMut(&str) -> bool,
        ) -> Result<GenerationOutput> {
            Ok(GenerationOutput {
                text: String::new(),
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: "stop".into(),
                prefill_seconds: 0.0,
                decode_seconds: 0.0,
            })
        }
    }
    for stream in [false, true] {
        let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
        std::thread::spawn(move || worker(Box::new(ContextEngine), receiver));
        let app = router(ServerState {
            sender,
            model: "test-model".into(),
            vocab_size: None,
        });
        let(status,body)=post(app,json!({"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":stream})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        let error: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(error["error"]["code"], "context_length_exceeded");
        assert_eq!(error["error"]["type"], "invalid_request_error");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires Node.js and @ai-sdk/openai-compatible@2.0.41 in OPENCODE_SDK_DIR"]
async fn opencode_sdk_tool_round_trip() {
    struct RoundTripEngine;
    impl TextGenerator for RoundTripEngine {
        fn prepare_request(&self, _: &mut GenerationRequest) -> Result<()> {
            Ok(())
        }
        fn model_id(&self) -> &str {
            "test-model"
        }
        fn vocab_size(&self) -> Option<usize> {
            Some(1000)
        }
        fn generate(
            &mut self,
            r: &GenerationRequest,
            callback: &mut dyn FnMut(&str) -> bool,
        ) -> Result<GenerationOutput> {
            let text = if r.messages.iter().any(|m| m.role == "tool") {
                let call = &r
                    .messages
                    .iter()
                    .find(|m| !m.tool_calls.is_empty())
                    .unwrap()
                    .tool_calls[0];
                let reply = r.messages.iter().find(|m| m.role == "tool").unwrap();
                anyhow::ensure!(
                    reply.tool_call_id.as_deref() == Some(&call.id),
                    "tool ID not preserved"
                );
                "Done."
            } else {
                anyhow::ensure!(r.tools.enabled(), "tools missing from SDK request");
                anyhow::ensure!(
                    r.messages.iter().any(|m| m.role == "user"
                        && m.content
                            == "Read README.md. After receiving its contents, answer Done."),
                    "multipart request changed"
                );
                "<tool_call><function=read_file><parameter=path>README.md</parameter></function></tool_call>"
            };
            for c in text.chars() {
                anyhow::ensure!(callback(&c.to_string()), "cancelled");
            }
            Ok(GenerationOutput {
                text: text.into(),
                prompt_tokens: 12,
                completion_tokens: 7,
                finish_reason: "stop".into(),
                prefill_seconds: 0.0,
                decode_seconds: 0.01,
            })
        }
    }
    let sdk = std::env::var("OPENCODE_SDK_DIR")
        .expect("set OPENCODE_SDK_DIR to isolated npm installation");
    let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
    std::thread::spawn(move || worker(Box::new(RoundTripEngine), receiver));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            router(ServerState {
                sender,
                model: "test-model".into(),
                vocab_size: Some(1000),
            }),
        )
        .await
        .unwrap()
    });
    let run =
        std::process::Command::new(std::env::var("NODE_BINARY").unwrap_or_else(|_| "node".into()))
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/opencode_sdk_contract.cjs"
            ))
            .env("OPENCODE_SDK_DIR", sdk)
            .env("MODEL_ID", "test-model")
            .env("BASE_URL", format!("http://{address}/v1"))
            .output()
            .unwrap();
    server.abort();
    assert!(
        run.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    println!("{}", String::from_utf8_lossy(&run.stdout));
}
