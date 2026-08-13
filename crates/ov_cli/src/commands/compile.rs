use std::io::Write;
use std::time::Duration;

use tokio::time::Instant;

use crate::client::{CompileAccepted, CompileTaskStatus, CompileTerminalStatus, HttpClient};
use crate::error::{Error, Result};
use crate::output::{output_success, OutputFormat};

pub async fn run(
    client: &HttpClient,
    from_uris: Vec<String>,
    to: String,
    skill: String,
    reason: Option<String>,
    wait: bool,
    timeout: Option<f64>,
    runtime_timeout: Option<f64>,
    output_format: OutputFormat,
    compact: bool,
) -> Result<()> {
    let sources = normalize_sources(from_uris)?;
    if timeout.is_some_and(|seconds| !crate::config::timeout_is_valid(seconds)) {
        return Err(Error::Client(
            "--timeout must be a positive finite number of seconds".into(),
        ));
    }
    let reason = reason
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let accepted = client
        .create_compile(&sources, to.trim(), skill.trim(), reason, runtime_timeout)
        .await?;
    if !wait {
        render_accepted(&accepted, output_format, compact);
        return Ok(());
    }

    let deadline = timeout.map(|seconds| Instant::now() + Duration::from_secs_f64(seconds));
    let mut polling = Duration::from_millis(500);
    loop {
        if deadline.is_some_and(|value| Instant::now() >= value) {
            return Err(Error::Client(format!(
                "Timed out waiting for compile task {}; the task is still running",
                accepted.task_id
            )));
        }
        let status = client.get_compile(&accepted.task_id).await?;
        match status.terminal_status() {
            Some(terminal_status) => {
                render_terminal(&status, output_format, compact)?;
                return if terminal_status.is_success() {
                    Ok(())
                } else {
                    Err(Error::AlreadyReported)
                };
            }
            None => {}
        }

        let sleep_for = deadline
            .map(|value| polling.min(value.saturating_duration_since(Instant::now())))
            .unwrap_or(polling);
        tokio::time::sleep(sleep_for).await;
        polling = (polling * 2).min(Duration::from_secs(2));
    }
}

fn normalize_sources(values: Vec<String>) -> Result<Vec<String>> {
    let mut result = Vec::new();
    for value in values {
        for item in value.split(',') {
            let item = item.trim();
            if item.is_empty() {
                return Err(Error::Client("--from contains an empty directory".into()));
            }
            if !result.iter().any(|existing| existing == item) {
                result.push(item.to_string());
            }
        }
    }
    if result.is_empty() {
        return Err(Error::Client(
            "at least one --from directory is required".into(),
        ));
    }
    Ok(result)
}

fn render_accepted(value: &CompileAccepted, format: OutputFormat, compact: bool) {
    if matches!(format, OutputFormat::Json) {
        output_success(value, format, compact);
    } else {
        println!("task_id: {}", value.task_id);
        println!("status: {}", value.status);
        println!("to: {}", value.to);
    }
}

fn render_terminal(value: &CompileTaskStatus, format: OutputFormat, compact: bool) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        output_success(value, format, compact);
    } else {
        println!("{}", format_terminal_human(value));
    }

    // main exits immediately for failed terminal states, so flush the status (and
    // any saved-output URIs) before returning the sentinel error that sets exit 1.
    std::io::stdout().flush()?;
    Ok(())
}

fn format_terminal_human(value: &CompileTaskStatus) -> String {
    let mut lines = vec![format!("status: {}", value.status)];
    if value.status == "partial" {
        lines.push("usable output saved".to_string());
    }

    if let Some(result) = value.result.as_ref() {
        lines.push(format!("to: {}", result.to));
        push_uri_effects(&mut lines, "created", &result.created);
        push_uri_effects(&mut lines, "updated", &result.updated);
        push_uri_effects(&mut lines, "unchanged", &result.unchanged);
        lines.push(format!("page_count: {}", result.page_count));
        lines.push(format!("link_count: {}", result.link_count));
        lines.extend(
            result
                .warnings
                .iter()
                .map(|warning| format!("warning: {warning}")),
        );
    }

    if let Some(error) = value.error.as_ref() {
        lines.push(format!("error: {}: {}", error.code, error.message));
    } else if matches!(
        value.terminal_status(),
        Some(CompileTerminalStatus::Incomplete | CompileTerminalStatus::Failed)
    ) {
        lines.push(format!("error: Compile task {}", value.status));
    }

    lines.join("\n")
}

fn push_uri_effects(lines: &mut Vec<String>, label: &str, uris: &[String]) {
    lines.push(format!("{label}: {}", uris.len()));
    lines.extend(uris.iter().map(|uri| format!("  {uri}")));
}

#[cfg(test)]
mod tests {
    use super::{format_terminal_human, normalize_sources};
    use crate::client::{
        CompileErrorInfo, CompileResult, CompileTaskStatus, CompileTerminalStatus,
    };

    fn result() -> CompileResult {
        CompileResult {
            from_uris: vec!["viking://resources/source".into()],
            to: "viking://resources/wiki".into(),
            skill: "viking://agent/skills/wiki".into(),
            okf_version: "0.1".into(),
            created: vec!["viking://resources/wiki/new.md".into()],
            updated: vec!["viking://resources/wiki/existing.md".into()],
            unchanged: vec!["viking://resources/wiki/stable.md".into()],
            page_count: 3,
            link_count: 1,
            warnings: vec!["one optional link was skipped".into()],
        }
    }

    fn status(name: &str) -> CompileTaskStatus {
        CompileTaskStatus {
            task_id: "cmp_1".into(),
            status: name.into(),
            stage: name.into(),
            created_at: "2026-08-13T00:00:00Z".into(),
            updated_at: "2026-08-13T00:01:00Z".into(),
            result: Some(result()),
            error: None,
        }
    }

    #[test]
    fn expands_comma_separated_and_repeated_sources_stably() {
        let result = normalize_sources(vec![
            "viking://resources/a,viking://resources/b".into(),
            "viking://resources/a".into(),
        ])
        .expect("sources should be valid");
        assert_eq!(result, vec!["viking://resources/a", "viking://resources/b"]);
    }

    #[test]
    fn rejects_empty_source_items() {
        assert!(normalize_sources(vec!["viking://resources/a,".into()]).is_err());
    }

    #[test]
    fn recognizes_all_compile_terminal_states() {
        assert_eq!(
            status("completed").terminal_status(),
            Some(CompileTerminalStatus::Completed)
        );
        assert_eq!(
            status("partial").terminal_status(),
            Some(CompileTerminalStatus::Partial)
        );
        assert_eq!(
            status("incomplete").terminal_status(),
            Some(CompileTerminalStatus::Incomplete)
        );
        assert_eq!(
            status("failed").terminal_status(),
            Some(CompileTerminalStatus::Failed)
        );
        assert_eq!(status("running").terminal_status(), None);
        assert!(CompileTerminalStatus::Completed.is_success());
        assert!(CompileTerminalStatus::Partial.is_success());
        assert!(!CompileTerminalStatus::Incomplete.is_success());
        assert!(!CompileTerminalStatus::Failed.is_success());
    }

    #[test]
    fn partial_human_output_reports_saved_output_and_actual_uris() {
        let rendered = format_terminal_human(&status("partial"));

        assert!(rendered.contains("status: partial\nusable output saved"));
        assert!(rendered.contains("created: 1\n  viking://resources/wiki/new.md"));
        assert!(rendered.contains("updated: 1\n  viking://resources/wiki/existing.md"));
        assert!(rendered.contains("unchanged: 1\n  viking://resources/wiki/stable.md"));
    }

    #[test]
    fn incomplete_human_output_lists_side_effects_before_error() {
        let mut value = status("incomplete");
        value.error = Some(CompileErrorInfo {
            code: "COVERAGE_INCOMPLETE".into(),
            message: "two review units remain pending".into(),
        });

        let rendered = format_terminal_human(&value);
        let updated = rendered
            .find("viking://resources/wiki/existing.md")
            .expect("updated URI should be rendered");
        let error = rendered
            .find("error: COVERAGE_INCOMPLETE")
            .expect("terminal error should be rendered");

        assert!(updated < error);
    }
}
