use ddrs::cli::fingerprint::{Fingerprint, fingerprint_path, reuse_if_unchanged};
use std::fs;

#[test]
fn fingerprint_blake3_csv_matches_known_content() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("a.csv");
    fs::write(&p, b"hello").unwrap();
    let fp = fingerprint_path(&p).unwrap();
    assert_eq!(fp.size, 5);
    assert!(fp.fp.starts_with("blake3:"));
    let again = fingerprint_path(&p).unwrap();
    assert_eq!(fp.fp, again.fp);
}

#[test]
fn fingerprint_changes_when_content_changes() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("a.csv");
    fs::write(&p, b"hello").unwrap();
    let fp1 = fingerprint_path(&p).unwrap();
    // Sleep so mtime resolution definitely changes
    std::thread::sleep(std::time::Duration::from_millis(50));
    fs::write(&p, b"world!").unwrap();
    let fp2 = fingerprint_path(&p).unwrap();
    assert_ne!(fp1.fp, fp2.fp);
    assert_ne!(fp1.size, fp2.size);
}

#[test]
fn reuse_returns_locked_fp_when_stat_matches() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("a.csv");
    fs::write(&p, b"hello").unwrap();
    let fp = fingerprint_path(&p).unwrap();
    let reused = reuse_if_unchanged(&p, &fp).unwrap();
    assert_eq!(reused.fp, fp.fp);
    assert!(reused.reused);
}

#[test]
fn reuse_recomputes_when_size_changes() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("a.csv");
    fs::write(&p, b"hello").unwrap();
    let fp = fingerprint_path(&p).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(50));
    fs::write(&p, b"goodbye").unwrap();
    let reused = reuse_if_unchanged(&p, &fp).unwrap();
    assert_ne!(reused.fp, fp.fp);
    assert!(!reused.reused);
}
