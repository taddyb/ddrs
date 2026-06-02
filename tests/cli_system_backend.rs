use ddrs::cli::manifest::{SmokeTestRecord, SystemProbe};

#[test]
fn smoke_record_with_backend_roundtrips() {
    let r = SmokeTestRecord {
        key: "x".into(),
        passed_at: "2026-06-01T00:00:00Z".into(),
        backend: Some("cuda".into()),
    };
    let json = serde_json::to_string(&r).unwrap();
    assert!(json.contains("\"backend\":\"cuda\""));
    let r2: SmokeTestRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(r2, r);
}

#[test]
fn smoke_record_old_format_deserializes_with_none_backend() {
    let json = r#"{"key":"x","passed_at":"2026-06-01T00:00:00Z"}"#;
    let r: SmokeTestRecord = serde_json::from_str(json).unwrap();
    assert_eq!(r.backend, None);
}

// Suppress unused import warning if needed.
#[allow(dead_code)]
fn _use_probe(_: SystemProbe) {}
