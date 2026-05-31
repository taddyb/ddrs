use std::path::{Path, PathBuf};

/// Walk up from `start` to find a `ddrs.yaml`. Stops at the first `.git`
/// ancestor (inclusive — the dir containing `.git` is searched, but no
/// further). Returns `None` if not found.
pub fn discover_config(start: &Path) -> Option<PathBuf> {
    let mut cur = start;
    loop {
        let cand = cur.join("ddrs.yaml");
        if cand.is_file() {
            return Some(cand);
        }
        if cur.join(".git").exists() {
            return None;
        }
        match cur.parent() {
            Some(p) => cur = p,
            None => return None,
        }
    }
}

/// All `.ddrs/` paths derived from a config location.
pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    pub fn beside(config: &Path) -> Self {
        let parent = config.parent().unwrap_or_else(|| Path::new("."));
        Self { root: parent.join(".ddrs") }
    }
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
    pub fn root(&self) -> &Path { &self.root }
    pub fn runs_dir(&self) -> PathBuf { self.root.join("runs") }
    pub fn lockfile(&self) -> PathBuf { self.root.join("sources.lock") }
    pub fn system_json(&self) -> PathBuf { self.root.join("system.json") }
    pub fn version_file(&self) -> PathBuf { self.root.join("version") }
}
