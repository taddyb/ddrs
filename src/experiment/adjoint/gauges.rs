//! Gauge population for the adjoint study (spec §2.1).
//!
//! `nested-reference`: a downstream gauge is selected when its GAGES-II CLASS
//! is `Ref` and at least one other training gauge's outlet COMID lies in its
//! subgraph. Its upstream partners are the *maximal* nested gauges — those not
//! themselves contained in another nested gauge's subgraph — so their
//! subgraphs are disjoint and the inherited-bias decomposition sums cleanly.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::data::dataset::MeritGagesDataset;

use super::super::BoxError;

#[derive(Debug, Clone)]
pub struct GaugeEntry {
    pub staid: String,
    pub role: &'static str,
    pub pair: usize,
    pub upstream: Vec<String>,
    pub comid: Option<i64>,
    pub class: Option<String>,
}

/// Unique gauges from an explicit `[upstream, downstream]` pair list,
/// upstream before downstream within a pair.
pub fn gauge_list_from_pairs(pairs: &[[String; 2]]) -> Vec<GaugeEntry> {
    let mut out: Vec<GaugeEntry> = Vec::new();
    for (pi, [up, down]) in pairs.iter().enumerate() {
        if !out.iter().any(|g| &g.staid == up) {
            out.push(GaugeEntry { staid: up.clone(), role: "upstream", pair: pi, upstream: vec![], comid: None, class: None });
        }
        match out.iter_mut().find(|g| &g.staid == down) {
            Some(g) => {
                g.role = "downstream";
                g.upstream.push(up.clone());
            }
            None => out.push(GaugeEntry {
                staid: down.clone(),
                role: "downstream",
                pair: pi,
                upstream: vec![up.clone()],
                comid: None,
                class: None,
            }),
        }
    }
    out
}

/// `STAID → CLASS` from the GAGES-II point shapefile's dbf.
pub fn read_gages_ii_class(dbf: &Path) -> Result<HashMap<String, String>, BoxError> {
    let mut reader = dbase::Reader::from_path(dbf).map_err(|e| format!("{}: {e}", dbf.display()))?;
    let mut out = HashMap::new();
    for rec in reader.iter_records() {
        let rec = rec.map_err(|e| format!("{}: {e}", dbf.display()))?;
        let staid = match rec.get("STAID") {
            Some(dbase::FieldValue::Character(Some(s))) => s.trim().to_string(),
            _ => continue,
        };
        let class = match rec.get("CLASS") {
            Some(dbase::FieldValue::Character(Some(s))) => s.trim().to_string(),
            _ => continue,
        };
        out.insert(staid, class);
    }
    if out.is_empty() {
        return Err(format!("{}: no STAID/CLASS records", dbf.display()).into());
    }
    Ok(out)
}

/// Nested-reference selection over the dataset's (already filtered) gauges.
pub fn nested_reference_selection(
    dataset: &MeritGagesDataset,
    class: &HashMap<String, String>,
    max_downstream: Option<usize>,
) -> Result<Vec<GaugeEntry>, BoxError> {
    // Outlet COMID per gauge, and upstream COMID sets (lazily, all gauges).
    let mut outlet: HashMap<String, i64> = HashMap::new();
    let mut upstream: HashMap<String, HashSet<i64>> = HashMap::new();
    for staid in &dataset.gauges {
        let Some(sg) = dataset.gages_adj.get(staid) else { continue };
        let Ok(comid) = sg.gage_catchment.trim().parse::<i64>() else { continue };
        outlet.insert(staid.as_str().to_string(), comid);
        let set: HashSet<i64> = sg.upstream_comids(&dataset.conus).into_iter().map(|c| c.0).collect();
        upstream.insert(staid.as_str().to_string(), set);
    }
    let mut downs: Vec<&String> = outlet
        .keys()
        .filter(|s| class.get(*s).map(|c| c == "Ref").unwrap_or(false))
        .collect();
    downs.sort();

    let mut pairs: Vec<[String; 2]> = Vec::new();
    let mut classes: HashMap<String, String> = HashMap::new();
    let mut n_down = 0usize;
    for down in downs {
        let set = &upstream[down];
        let mut nested: Vec<&String> = outlet
            .iter()
            .filter(|(s, c)| *s != down && set.contains(c))
            .map(|(s, _)| s)
            .collect();
        if nested.is_empty() {
            continue;
        }
        // Maximal nested gauges: not inside another nested gauge's subgraph.
        nested.sort();
        let maximal: Vec<&String> = nested
            .iter()
            .filter(|b| !nested.iter().any(|c| c != *b && upstream[*c].contains(&outlet[**b])))
            .copied()
            .collect();
        for up in maximal {
            pairs.push([up.clone(), down.clone()]);
            classes.insert(up.clone(), class.get(up).cloned().unwrap_or_else(|| "n/a".into()));
        }
        classes.insert(down.clone(), "Ref".into());
        n_down += 1;
        if max_downstream.map_or(false, |m| n_down >= m) {
            break;
        }
    }
    let mut list = gauge_list_from_pairs(&pairs);
    for g in &mut list {
        g.comid = outlet.get(&g.staid).copied();
        g.class = classes.get(&g.staid).cloned();
    }
    Ok(list)
}

/// Every gauge the dataset can evaluate: `MeritGagesDataset::open`'s filter
/// pipeline (DA_VALID + adjacency present, non-headwater + observations
/// present — the `gages_adjacency filter: kept X gauges` / `observations
/// filter: kept X/Y gauges` log lines) already leaves exactly this
/// population in `dataset.gauges`, the same one `evaluate` scores. No
/// pairing: every entry stands alone (`upstream` empty, `role: "gauge"`).
pub fn all_gauges_selection(dataset: &MeritGagesDataset) -> Vec<GaugeEntry> {
    let mut list: Vec<GaugeEntry> = dataset
        .gauges
        .iter()
        .map(|s| {
            let comid = dataset.gages_adj.get(s).and_then(|sg| sg.gage_catchment.trim().parse::<i64>().ok());
            GaugeEntry { staid: s.as_str().to_string(), role: "gauge", pair: 0, upstream: vec![], comid, class: None }
        })
        .collect();
    list.sort_by(|a, b| a.staid.cmp(&b.staid));
    list
}

pub fn write_gauges_csv(path: &Path, gauges: &[GaugeEntry]) -> Result<(), BoxError> {
    use std::io::Write;
    let mut w = std::fs::File::create(path)?;
    writeln!(w, "staid,role,pair,comid,class,upstream_staids")?;
    for g in gauges {
        writeln!(
            w,
            "{},{},{},{},{},{}",
            g.staid,
            g.role,
            g.pair,
            g.comid.map(|c| c.to_string()).unwrap_or_default(),
            g.class.clone().unwrap_or_default(),
            g.upstream.join(";")
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauge_list_marks_roles() {
        let g = gauge_list_from_pairs(&[["A".into(), "B".into()], ["B".into(), "C".into()]]);
        assert_eq!(g.iter().map(|x| x.staid.as_str()).collect::<Vec<_>>(), vec!["A", "B", "C"]);
        assert_eq!(g[1].role, "downstream");
        assert_eq!(g[1].upstream, vec!["A".to_string()]);
        assert_eq!(g[2].upstream, vec!["B".to_string()]);
    }
}
