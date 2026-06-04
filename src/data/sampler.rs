//! Batch index samplers.
//!
//! Mirrors `torch.utils.data.{RandomSampler, SequentialSampler}` for our
//! batching needs. Not PyTorch-bit-identical — the project's verification
//! bar doesn't require it. Determinism via `rand::SeedableRng`.

use rand::seq::SliceRandom;
use rand::Rng;

use crate::data::ids::Staid;

pub struct RandomSampler {
    indices: Vec<usize>,
    batch_size: usize,
    cursor: usize,
    drop_last: bool,
}

impl RandomSampler {
    /// Build a sampler over `n` items. Permute the index list once per
    /// epoch via `reshuffle`, then call `next_batch` to drive iteration.
    pub fn new(n: usize, batch_size: usize, drop_last: bool) -> Self {
        Self {
            indices: (0..n).collect(),
            batch_size,
            cursor: 0,
            drop_last,
        }
    }

    /// Permute the index list for a fresh epoch.
    pub fn reshuffle<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        self.indices.shuffle(rng);
        self.cursor = 0;
    }

    /// Return the next batch's indices, or `None` if the epoch is done.
    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        let remaining = self.indices.len().saturating_sub(self.cursor);
        if remaining == 0 {
            return None;
        }
        let take = if remaining >= self.batch_size {
            self.batch_size
        } else if self.drop_last {
            return None;
        } else {
            remaining
        };
        let out = self.indices[self.cursor..self.cursor + take].to_vec();
        self.cursor += take;
        Some(out)
    }
}

/// Source of mini-batches for the training loop. Either generates batches
/// on the fly via `RandomSampler` (default) or replays a captured order
/// via `ReplaySampler` (matched-batch parity experiment — see spec
/// `docs/superpowers/specs/2026-06-04-matched-batch-replay-design.md`).
pub enum BatchSource {
    Shuffle(RandomSampler),
    Replay(ReplaySampler),
}

impl BatchSource {
    /// Re-shuffle the underlying sampler for a new epoch. No-op for replay
    /// (the recorded trace IS the full multi-epoch sequence; the replay
    /// sampler exhausts after the trace ends regardless of the driver's
    /// outer epoch loop).
    pub fn reshuffle<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        if let BatchSource::Shuffle(s) = self {
            s.reshuffle(rng);
        }
    }

    /// Get the next batch's dataset-row indices, or None if exhausted.
    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        match self {
            BatchSource::Shuffle(s) => s.next_batch(),
            BatchSource::Replay(s) => s.next_batch(),
        }
    }
}

pub struct SequentialSampler {
    n: usize,
    batch_size: usize,
    cursor: usize,
}

impl SequentialSampler {
    pub fn new(n: usize, batch_size: usize) -> Self {
        Self {
            n,
            batch_size,
            cursor: 0,
        }
    }

    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        if self.cursor >= self.n {
            return None;
        }
        let end = (self.cursor + self.batch_size).min(self.n);
        let out: Vec<usize> = (self.cursor..end).collect();
        self.cursor = end;
        Some(out)
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }
}

/// Yields pre-recorded mini-batches in order. Companion to `RandomSampler`
/// for the matched-batch replay experiment.
pub struct ReplaySampler {
    /// Each inner Vec is one mini-batch of dataset-row indices.
    batches: Vec<Vec<usize>>,
    cursor: usize,
}

impl ReplaySampler {
    /// Build from a list of mini-batches of STAIDs. Resolves each STAID
    /// against the dataset's `all_staids` array to recover dataset-row
    /// indices. Panics if any recorded STAID is not present in `all_staids`.
    pub fn new(recorded: Vec<Vec<Staid>>, all_staids: &[Staid]) -> Self {
        let lookup: std::collections::HashMap<&Staid, usize> = all_staids
            .iter()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();
        let batches: Vec<Vec<usize>> = recorded
            .into_iter()
            .map(|batch| {
                batch
                    .into_iter()
                    .map(|s| {
                        *lookup.get(&s).unwrap_or_else(|| {
                            panic!(
                                "ReplaySampler: STAID {s:?} from recorded order not \
                                 present in dataset's all_staids (len={})",
                                all_staids.len()
                            )
                        })
                    })
                    .collect()
            })
            .collect();
        Self { batches, cursor: 0 }
    }

    /// Yield the next pre-recorded batch's row indices, or `None` if exhausted.
    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        if self.cursor >= self.batches.len() {
            return None;
        }
        let out = self.batches[self.cursor].clone();
        self.cursor += 1;
        Some(out)
    }

    /// Number of mini-batches remaining in the replay queue.
    pub fn remaining(&self) -> usize {
        self.batches.len().saturating_sub(self.cursor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn random_sampler_covers_all_indices_with_drop_last_false() {
        let mut s = RandomSampler::new(10, 3, false);
        let mut rng = StdRng::seed_from_u64(42);
        s.reshuffle(&mut rng);
        let mut seen: Vec<usize> = Vec::new();
        while let Some(b) = s.next_batch() {
            seen.extend(b);
        }
        let mut seen_sorted = seen.clone();
        seen_sorted.sort();
        assert_eq!(seen_sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn random_sampler_drop_last_skips_partial_batch() {
        let mut s = RandomSampler::new(10, 3, true);
        let mut rng = StdRng::seed_from_u64(42);
        s.reshuffle(&mut rng);
        let mut total = 0;
        while let Some(b) = s.next_batch() {
            assert_eq!(b.len(), 3);
            total += b.len();
        }
        assert_eq!(total, 9);
    }

    #[test]
    fn random_sampler_seeded_reproducible() {
        let mut s1 = RandomSampler::new(8, 2, false);
        let mut r1 = StdRng::seed_from_u64(7);
        s1.reshuffle(&mut r1);
        let b1: Vec<Vec<usize>> = std::iter::from_fn(|| s1.next_batch()).collect();

        let mut s2 = RandomSampler::new(8, 2, false);
        let mut r2 = StdRng::seed_from_u64(7);
        s2.reshuffle(&mut r2);
        let b2: Vec<Vec<usize>> = std::iter::from_fn(|| s2.next_batch()).collect();

        assert_eq!(b1, b2);
    }

    #[test]
    fn sequential_sampler_yields_in_order_with_partial_tail() {
        let mut s = SequentialSampler::new(7, 3);
        assert_eq!(s.next_batch(), Some(vec![0, 1, 2]));
        assert_eq!(s.next_batch(), Some(vec![3, 4, 5]));
        assert_eq!(s.next_batch(), Some(vec![6]));
        assert_eq!(s.next_batch(), None);
    }

    #[test]
    fn sequential_sampler_reset_restarts() {
        let mut s = SequentialSampler::new(4, 2);
        let _ = s.next_batch();
        let _ = s.next_batch();
        assert_eq!(s.next_batch(), None);
        s.reset();
        assert_eq!(s.next_batch(), Some(vec![0, 1]));
    }

    #[test]
    fn replay_sampler_yields_recorded_batches_in_order() {
        use crate::data::ids::Staid;
        // STAIDs as registered with the dataset.
        let all_staids: Vec<Staid> = (0..10)
            .map(|i| Staid::new(&format!("STAID_{i:02}")))
            .collect();
        // Record: 3 mini-batches of 2 STAIDs each.
        let recorded: Vec<Vec<Staid>> = vec![
            vec![all_staids[3].clone(), all_staids[1].clone()],
            vec![all_staids[7].clone(), all_staids[2].clone()],
            vec![all_staids[5].clone(), all_staids[8].clone()],
        ];

        let mut replay = ReplaySampler::new(recorded, &all_staids);

        let b1 = replay.next_batch().expect("batch 1");
        assert_eq!(b1, vec![3, 1]);
        let b2 = replay.next_batch().expect("batch 2");
        assert_eq!(b2, vec![7, 2]);
        let b3 = replay.next_batch().expect("batch 3");
        assert_eq!(b3, vec![5, 8]);
        assert!(replay.next_batch().is_none(),
            "ReplaySampler should exhaust after the recorded batches");
    }

    #[test]
    fn replay_sampler_rejects_unknown_staid() {
        use crate::data::ids::Staid;
        let all_staids = vec![Staid::new("KNOWN")];
        let recorded = vec![vec![Staid::new("UNKNOWN")]];
        let result = std::panic::catch_unwind(|| {
            let _ = ReplaySampler::new(recorded, &all_staids);
        });
        assert!(result.is_err(),
            "ReplaySampler::new should panic when a recorded STAID is not in the dataset");
    }
}
