use ddrs::cli::workspace::{discover_config, Workspace};
use std::fs;

fn tmp() -> tempfile::TempDir { tempfile::tempdir().unwrap() }

#[test]
fn discover_finds_ddrs_yaml_in_cwd() {
    let d = tmp();
    fs::write(d.path().join("ddrs.yaml"), "workflow: train\n").unwrap();
    let found = discover_config(d.path()).unwrap();
    assert_eq!(found, d.path().join("ddrs.yaml"));
}

#[test]
fn discover_walks_up_until_git_root() {
    let root = tmp();
    fs::create_dir_all(root.path().join(".git")).unwrap();
    fs::write(root.path().join("ddrs.yaml"), "workflow: train\n").unwrap();
    let sub = root.path().join("a").join("b");
    fs::create_dir_all(&sub).unwrap();
    let found = discover_config(&sub).unwrap();
    assert_eq!(found, root.path().join("ddrs.yaml"));
}

#[test]
fn discover_stops_at_git_root_without_config() {
    let root = tmp();
    fs::create_dir_all(root.path().join(".git")).unwrap();
    let sub = root.path().join("a");
    fs::create_dir_all(&sub).unwrap();
    assert!(discover_config(&sub).is_none());
}

#[test]
fn workspace_paths_resolve_relative_to_config() {
    let d = tmp();
    let cfg = d.path().join("ddrs.yaml");
    fs::write(&cfg, "workflow: train\n").unwrap();
    let ws = Workspace::beside(&cfg);
    assert_eq!(ws.root(), d.path().join(".ddrs"));
    assert_eq!(ws.runs_dir(), d.path().join(".ddrs").join("runs"));
    assert_eq!(ws.lockfile(), d.path().join(".ddrs").join("sources.lock"));
    assert_eq!(ws.system_json(), d.path().join(".ddrs").join("system.json"));
}
