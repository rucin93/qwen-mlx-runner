//! Format/execution smoke check using a real adapter and synthetic shared target weights.
//! This does not evaluate target generation quality or model throughput.
use anyhow::{Context, Result};
use qwen_metal::engine::{Engine, Mtp};
use serde_json::json;
use std::{path::PathBuf, time::Instant};

fn main() -> Result<()> {
    let path = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .context("pass the native MTP adapter directory")?,
    );
    let target = Engine::synthetic_qwen27b(8, 0)?;
    let start = Instant::now();
    let mut mtp = Mtp::load(&target, &path, 8)?;
    let load_seconds = start.elapsed().as_secs_f64();
    let mut hidden = vec![1.0; target.config().hidden_size];
    let mut rows = Vec::new();
    for token in [3, 17, 29] {
        let started = Instant::now();
        let output = mtp.forward(token, &hidden, true)?;
        rows.push(
            json!({"position":mtp.position(), "seconds":started.elapsed().as_secs_f64(),
            "hidden_elements":output.hidden.len(), "logit_elements":output.logits.len(),
            "max_abs_hidden":output.hidden.iter().map(|v|v.abs()).fold(0.0f32,f32::max),
            "max_abs_logit":output.logits.iter().map(|v|v.abs()).fold(0.0f32,f32::max)}),
        );
        hidden = output.hidden;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "kind":"native_adapter_format_execution_check", "adapter":path,
            "device":target.device_name(), "allocated_bytes":target.allocated_bytes(),
            "adapter_load_seconds":load_seconds, "steps":rows,
            "note":"Real adapter, SYNTHETIC target embedding/head and input hidden. Validates loading, dimensions and finite execution only; NOT model throughput or chat quality."
        }))?
    );
    Ok(())
}
