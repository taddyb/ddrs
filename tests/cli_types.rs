use ddrs::cli::types::{ExitCode, RunStatus, Workflow};

#[test]
fn workflow_serializes_kebab_case() {
    assert_eq!(serde_json::to_string(&Workflow::Train).unwrap(), "\"train\"");
    assert_eq!(serde_json::to_string(&Workflow::Eval).unwrap(), "\"eval\"");
    assert_eq!(
        serde_json::to_string(&Workflow::TrainAndTest).unwrap(),
        "\"train-and-test\""
    );
}

#[test]
fn run_status_round_trips() {
    for s in [RunStatus::Ok, RunStatus::Failed, RunStatus::Interrupted] {
        let s2: RunStatus = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(s, s2);
    }
}

#[test]
fn exit_code_values_match_spec() {
    assert_eq!(ExitCode::Success as i32, 0);
    assert_eq!(ExitCode::Generic as i32, 1);
    assert_eq!(ExitCode::ConfigInvalid as i32, 2);
    assert_eq!(ExitCode::DataSourceMissing as i32, 3);
    assert_eq!(ExitCode::LockDrift as i32, 4);
    assert_eq!(ExitCode::RuntimeFailure as i32, 5);
    assert_eq!(ExitCode::WorkspaceNotInitialized as i32, 6);
}
