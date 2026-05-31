use ddrs::cli::lockfile::{Lockfile, diff_against_live};
use ddrs::cli::fingerprint::Fingerprint;
use std::collections::BTreeMap;
use std::fs;

fn fp(path: &str, fp: &str) -> Fingerprint {
    Fingerprint {
        path: path.into(), mtime: "2026-05-30T00:00:00Z".into(),
        size: 1, fp: fp.into(),
    }
}

#[test]
fn lockfile_round_trips() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("sources.lock");
    let mut sources = BTreeMap::new();
    sources.insert("attributes".into(), fp("/x", "blake3:aaa"));
    let lock = Lockfile { ddrs_version: "0.1.0".into(),
        created_at: "2026-05-30T00:00:00Z".into(), sources };
    lock.write_atomic(&p).unwrap();
    let loaded = Lockfile::read(&p).unwrap();
    assert_eq!(loaded, lock);
}

#[test]
fn diff_lists_drifted_keys() {
    let mut sources = BTreeMap::new();
    sources.insert("attributes".into(), fp("/x", "blake3:aaa"));
    sources.insert("conus_adjacency".into(), fp("/y", "blake3:bbb"));
    let lock = Lockfile { ddrs_version: "x".into(),
        created_at: "x".into(), sources };
    let mut live = BTreeMap::new();
    live.insert("attributes".into(), fp("/x", "blake3:aaa"));         // unchanged
    live.insert("conus_adjacency".into(), fp("/y", "blake3:CHANGED"));// drifted
    let drift = diff_against_live(&lock, &live);
    assert_eq!(drift, vec!["conus_adjacency".to_string()]);
}
