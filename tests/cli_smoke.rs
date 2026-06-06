use ddrs::cli::manifest::SystemProbe;

#[test]
fn run_smoke_returns_cpu_when_no_cuda() {
    let mut probe = SystemProbe::default();
    probe.gpu = String::new();
    let (passed, backend) = ddrs::cli::system::run_smoke_for_test(&probe).unwrap();
    assert!(passed, "CPU smoke must pass on the bundled sandbox fixture");
    assert_eq!(backend, "cpu");
}
