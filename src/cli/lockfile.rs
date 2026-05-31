use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::cli::CliError;
use crate::cli::fingerprint::Fingerprint;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lockfile {
    pub ddrs_version: String,
    pub created_at: String,
    pub sources: BTreeMap<String, Fingerprint>,
}

impl Lockfile {
    pub fn read(path: &Path) -> Result<Self, CliError> {
        let s = fs::read_to_string(path)?;
        serde_json::from_str(&s).map_err(|e| CliError::Other(Box::new(e)))
    }

    pub fn write_atomic(&self, path: &Path) -> Result<(), CliError> {
        let tmp = path.with_extension("lock.tmp");
        let s = serde_json::to_string_pretty(self)
            .map_err(|e| CliError::Other(Box::new(e)))?;
        let mut f = fs::File::create(&tmp)?;
        f.write_all(s.as_bytes())?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, path)?;
        Ok(())
    }
}

/// Return field names whose `fp` differs between the locked snapshot and
/// the live fingerprints. Keys only present on one side are also reported.
pub fn diff_against_live(
    lock: &Lockfile,
    live: &BTreeMap<String, Fingerprint>,
) -> Vec<String> {
    let mut drift = Vec::new();
    for (k, v) in &lock.sources {
        match live.get(k) {
            Some(l) if l.fp == v.fp => {}
            _ => drift.push(k.clone()),
        }
    }
    for k in live.keys() {
        if !lock.sources.contains_key(k) {
            drift.push(k.clone());
        }
    }
    drift.sort();
    drift
}
