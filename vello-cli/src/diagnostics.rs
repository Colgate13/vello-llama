//! Runtime diagnostics: health check, GPU monitor, throughput bench, tool-call test.
//!
//! HTTP work is delegated to `curl` + `jq` (already required for raw downloads).
//! Avoids pulling in a heavy HTTP stack just for two endpoints.

use crate::paths::Paths;
use crate::system::SystemConfig;
use anyhow::{bail, Context, Result};
use std::io::IsTerminal;
use std::process::{Command, Stdio};

pub fn health(sys: &SystemConfig) -> Result<()> {
    let port = sys.ports.llama;
    let out = Command::new("curl")
        .args(["-fsS", &format!("http://localhost:{port}/health")])
        .output()
        .context("invoking curl")?;
    if out.status.success() {
        println!("{} API healthy on :{}", green("ok"), port);
        Ok(())
    } else {
        bail!("API not responding on :{port}")
    }
}

pub fn gpu() -> Result<()> {
    if !command_exists("watch") {
        bail!("'watch' not installed (apt install procps)");
    }
    let status = Command::new("watch")
        .args(["-n", "1", "nvidia-smi"])
        .status()
        .context("invoking watch")?;
    if !status.success() {
        // ctrl-c is the normal exit; that's fine
    }
    Ok(())
}

pub fn bench(sys: &SystemConfig, prompt: &str, n: u32) -> Result<()> {
    require_running()?;
    let port = sys.ports.llama;
    let req = format!(
        r#"{{"prompt":{prompt},"n_predict":{n},"cache_prompt":false,"stream":false}}"#,
        prompt = json_string(prompt),
        n = n
    );
    let out = Command::new("curl")
        .args([
            "-fsS",
            &format!("http://localhost:{port}/completion"),
            "-H",
            "Content-Type: application/json",
            "-d",
            &req,
        ])
        .output()
        .context("invoking curl")?;
    if !out.status.success() {
        bail!("benchmark request failed");
    }
    let body = String::from_utf8_lossy(&out.stdout);
    let timings = jq_query(&body, ".timings // {}")?;
    let prompt_n = jq_query(&timings, ".prompt_n // 0")?;
    let prompt_ms = jq_query(&timings, ".prompt_ms // 0")?;
    let prompt_per_s = jq_query(&timings, ".prompt_per_second // 0 | round")?;
    let pred_n = jq_query(&timings, ".predicted_n // 0")?;
    let pred_ms = jq_query(&timings, ".predicted_ms // 0")?;
    let pred_per_s = jq_query(&timings, ".predicted_per_second // 0 | round")?;
    println!(
        "  prompt eval:  {} tokens / {:.2}s = {} tok/s",
        prompt_n.trim(),
        prompt_ms.trim().parse::<f64>().unwrap_or(0.0) / 1000.0,
        prompt_per_s.trim()
    );
    println!(
        "  generation:   {} tokens / {:.2}s = {} tok/s",
        pred_n.trim(),
        pred_ms.trim().parse::<f64>().unwrap_or(0.0) / 1000.0,
        pred_per_s.trim()
    );
    Ok(())
}

pub fn tools_test(paths: &Paths, sys: &SystemConfig) -> Result<()> {
    require_running()?;
    let port = sys.ports.llama;
    let alias = active_alias(paths).unwrap_or_else(|| "local-model".into());

    let req = format!(
        r#"{{"model":{m},"messages":[{{"role":"system","content":"You use tools when external data is needed."}},{{"role":"user","content":"What is the weather in Paris right now? Use the tool."}}],"tools":[{{"type":"function","function":{{"name":"get_weather","description":"Get current weather for a city","parameters":{{"type":"object","properties":{{"city":{{"type":"string"}}}},"required":["city"]}}}}}}],"tool_choice":"auto","max_tokens":200,"temperature":0}}"#,
        m = json_string(&alias)
    );

    println!("==> tool-calling test on {alias}");
    let out = Command::new("curl")
        .args([
            "-fsS",
            &format!("http://localhost:{port}/v1/chat/completions"),
            "-H",
            "Content-Type: application/json",
            "-d",
            &req,
        ])
        .output()
        .context("invoking curl")?;
    if !out.status.success() {
        bail!("test request failed");
    }
    let body = String::from_utf8_lossy(&out.stdout);
    let name = jq_query(
        &body,
        ".choices[0].message.tool_calls[0].function.name // empty",
    )?;
    if !name.trim().is_empty() {
        println!("{} tool calling works ({})", green("ok"), name.trim());
        Ok(())
    } else {
        let text = jq_query(&body, ".choices[0].message.content // empty")?;
        eprintln!("{} model returned plain text, not a tool_call", red("fail"));
        if !text.trim().is_empty() {
            eprintln!("  raw: {}", text.trim());
        }
        eprintln!("  Tip: switch to a tools-friendly model: vello switch qwen3-8b");
        bail!("tool-calling validation failed")
    }
}

fn require_running() -> Result<()> {
    // We use docker compose status for this; but importing docker creates a
    // cycle with paths. Cheap fallback: try connecting. The /health check in
    // health() already covers this; for bench/test we can let curl fail.
    Ok(())
}

fn active_alias(paths: &Paths) -> Option<String> {
    let env = std::fs::read_to_string(paths.project_root.join(".env")).ok()?;
    for line in env.lines() {
        if let Some(rest) = line.strip_prefix("LLAMA_MODEL_ALIAS=") {
            return Some(rest.trim().trim_matches('"').to_string());
        }
    }
    None
}

fn jq_query(input: &str, expr: &str) -> Result<String> {
    let mut child = Command::new("jq")
        .args(["-r", expr])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("invoking jq — install with: apt install jq")?;
    {
        use std::io::Write;
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("jq stdin unavailable"))?;
        stdin.write_all(input.as_bytes())?;
    }
    let out = child.wait_with_output()?;
    if !out.status.success() {
        bail!("jq failed");
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn json_string(s: &str) -> String {
    let escaped = s
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!("\"{escaped}\"")
}

fn command_exists(cmd: &str) -> bool {
    Command::new("which")
        .arg(cmd)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn use_color() -> bool {
    std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none()
}

fn green(s: &str) -> String {
    if use_color() {
        format!("\x1b[32m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

fn red(s: &str) -> String {
    if use_color() {
        format!("\x1b[31m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}
