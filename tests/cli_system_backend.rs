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

#[test]
fn smoke_key_includes_backend() {
    let probe = SystemProbe {
        ddrs_version: "1".into(),
        probed_at: "t".into(),
        gpu: "g".into(),
        cuda_runtime: "12.4".into(),
        driver: "530".into(),
        sm: "8.0".into(),
        free_gpu_gb_at_probe: 1.0,
        smoke_test: None,
    };
    let k_cuda = ddrs::cli::system::smoke_key(&probe, "cuda");
    let k_cpu  = ddrs::cli::system::smoke_key(&probe, "cpu");
    assert_ne!(k_cuda, k_cpu);
    assert!(k_cuda.contains("backend=cuda"));
    assert!(k_cpu.contains("backend=cpu"));
}
