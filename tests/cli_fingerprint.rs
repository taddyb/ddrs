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

// ---------------------------------------------------------------------------
// Directory stores: content fingerprints (Task 1 of the pins plan).
// ---------------------------------------------------------------------------

/// Build a zarr-v3-shaped store: root `zarr.json`, a nested array with two
/// chunk files.
fn zarr_like(root: &std::path::Path) {
    fs::create_dir_all(root.join("Qr/c/0")).unwrap();
    fs::write(root.join("zarr.json"), br#"{"zarr_format":3,"node_type":"group"}"#).unwrap();
    fs::write(root.join("Qr/zarr.json"), br#"{"zarr_format":3,"node_type":"array"}"#).unwrap();
    fs::write(root.join("Qr/c/0/0"), b"chunk-zero").unwrap();
    fs::write(root.join("Qr/c/0/1"), b"chunk-one!").unwrap();
}

#[test]
fn directory_fp_changes_when_a_nested_chunk_changes() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("store.zarr");
    zarr_like(&root);
    let before = fingerprint_path(&root).unwrap();

    // Edit a chunk two levels down. The fingerprint hashes the recursive
    // (relative path, size) listing, so the edit is seen through the size.
    // KNOWN LIMIT: an in-place rewrite that preserves length is NOT detected
    // — hashing every byte of the CONUS/global stores on each `ddrs plan` was
    // judged too expensive. See `fingerprint.rs::hash_dir`.
    fs::write(root.join("Qr/c/0/1"), b"chunk-one-rewritten").unwrap();
    let after = fingerprint_path(&root).unwrap();
    assert_ne!(
        before.fp, after.fp,
        "nested chunk edit must change the directory fingerprint"
    );
}

#[test]
fn directory_fp_stable_when_only_mtimes_change() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("store.zarr");
    zarr_like(&root);
    let before = fingerprint_path(&root).unwrap();

    std::thread::sleep(std::time::Duration::from_millis(50));
    for rel in ["zarr.json", "Qr/zarr.json", "Qr/c/0/0", "Qr/c/0/1"] {
        let p = root.join(rel);
        let bytes = fs::read(&p).unwrap();
        fs::write(&p, bytes).unwrap(); // rewrite identical content, new mtime
    }
    let after = fingerprint_path(&root).unwrap();
    assert_eq!(
        before.fp, after.fp,
        "rewriting identical bytes must not change the fingerprint"
    );
}

#[test]
fn the_two_juniata_icechunk_stores_have_distinct_fingerprints() {
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/juniata/data");
    let qprime = fingerprint_path(&data.join("juniata_qprime.ic")).unwrap();
    let obs = fingerprint_path(&data.join("juniata_obs.ic")).unwrap();
    assert_ne!(
        qprime.fp, obs.fp,
        "two different icechunk stores must not share a fingerprint"
    );
}

/// Open the repo the way `src/data/store/icechunk.rs::open_session` does and
/// read the `main` branch tip, so the assertion below is independent of the
/// production helper.
fn main_branch_tip(path: &std::path::Path) -> String {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let storage = icechunk::new_local_filesystem_storage(path).await.unwrap();
        let repo = icechunk::Repository::open(None, storage, std::collections::HashMap::new())
            .await
            .unwrap();
        repo.lookup_branch("main").await.unwrap().to_string()
    })
}

#[test]
fn icechunk_fp_is_the_main_branch_snapshot_id() {
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/juniata/data");
    for name in ["juniata_qprime.ic", "juniata_obs.ic"] {
        let p = data.join(name);
        let fp = fingerprint_path(&p).unwrap();
        let tip = main_branch_tip(&p);
        assert_eq!(fp.fp, format!("icechunk:{tip}"), "{name}");
        assert_eq!(fp.snapshot.as_deref(), Some(tip.as_str()), "{name}");
    }
}

#[test]
fn directory_reuse_always_recomputes() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("store.zarr");
    zarr_like(&root);
    let locked = fingerprint_path(&root).unwrap();

    // Edit a nested chunk in place: the root directory's own mtime and size
    // are untouched, so a stat short-circuit would wrongly report "reused".
    fs::write(root.join("Qr/c/0/1"), b"chunk-one-rewritten").unwrap();
    let r = reuse_if_unchanged(&root, &locked).unwrap();
    assert!(!r.reused, "directories must never short-circuit on stat");
    assert_ne!(r.fp, locked.fp);
}

#[test]
fn fingerprint_json_without_snapshot_still_parses() {
    let json = r#"{"path":"/a/b.csv","mtime":"2026-01-01T00:00:00Z","size":5,"fp":"blake3:dead"}"#;
    let fp: Fingerprint = serde_json::from_str(json).unwrap();
    assert_eq!(fp.snapshot, None);
    assert_eq!(fp.fp, "blake3:dead");
    // And a snapshot-less fingerprint round-trips without emitting the key.
    let back = serde_json::to_string(&fp).unwrap();
    assert!(!back.contains("snapshot"), "{back}");
}

