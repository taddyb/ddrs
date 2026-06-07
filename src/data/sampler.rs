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

    /// Capture the current permutation + cursor for a checkpoint sidecar.
    pub fn snapshot(&self) -> (Vec<usize>, usize) {
        (self.indices.clone(), self.cursor)
    }

    /// Restore a permutation + cursor captured by [`snapshot`](Self::snapshot)
    /// so a resumed run continues mid-epoch with the same gauge batches.
    pub fn restore(&mut self, indices: Vec<usize>, cursor: usize) {
        self.indices = indices;
        self.cursor = cursor;
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
    /// Re-shuffle the underlying sampler for a new epoch. For `Shuffle`,
    /// permutes the index list. For `Replay`, advances the internal epoch
    /// counter so `next_batch` yields batches for the new epoch.
    pub fn reshuffle<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        match self {
            BatchSource::Shuffle(s) => s.reshuffle(rng),
            BatchSource::Replay(s) => s.advance_epoch(),
        }
    }

    /// Get the next batch's dataset-row indices, or None if exhausted.
    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        match self {
            BatchSource::Shuffle(s) => s.next_batch(),
            BatchSource::Replay(s) => s.next_batch(),
        }
    }

    /// Snapshot the underlying `RandomSampler` for a checkpoint sidecar.
    /// `None` for `Replay` — the matched-batch experiment carries its own
    /// batch record and doesn't participate in checkpoint resume.
    pub fn snapshot(&self) -> Option<(Vec<usize>, usize)> {
        match self {
            BatchSource::Shuffle(s) => Some(s.snapshot()),
            BatchSource::Replay(_) => None,
        }
    }

    /// Restore a `Shuffle` sampler from a checkpoint sidecar (mid-epoch
    /// resume). No-op for `Replay`.
    pub fn restore(&mut self, indices: Vec<usize>, cursor: usize) {
        if let BatchSource::Shuffle(s) = self {
            s.restore(indices, cursor);
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

/// Yields pre-recorded mini-batches in order, respecting per-batch epoch
/// labels. Companion to `RandomSampler` for the matched-batch replay
/// experiment.
pub struct ReplaySampler {
    /// Each entry: (original_epoch, dataset-row indices for the batch).
    /// Sorted by (epoch, mb_within_epoch) at construction.
    batches: Vec<(u32, Vec<usize>)>,
    /// Cursor into `batches`. Monotonically increasing across the whole
    /// run; never reset.
    cursor: usize,
    /// Current outer epoch the driver is requesting. 0 = "before any
    /// outer epoch has started"; `advance_epoch` bumps this on each call.
    /// `next_batch` returns `Some(...)` iff `batches[cursor].0 == current_epoch`.
    current_epoch: u32,
}

impl ReplaySampler {
    /// Build from per-batch records: `(epoch, ordered_staids)`.
    /// Resolves each STAID against `all_staids` to recover row indices.
    /// Panics if any recorded STAID is missing from the dataset.
    pub fn new(recorded: Vec<(u32, Vec<Staid>)>, all_staids: &[Staid]) -> Self {
        let lookup: std::collections::HashMap<&Staid, usize> = all_staids
            .iter()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();
        let batches: Vec<(u32, Vec<usize>)> = recorded
            .into_iter()
            .map(|(ep, batch_staids)| {
                let indices: Vec<usize> = batch_staids
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
                    .collect();
                (ep, indices)
            })
            .collect();
        Self { batches, cursor: 0, current_epoch: 0 }
    }

    /// Yield the next batch if it belongs to the current epoch; else
    /// return None so the driver moves on to the next epoch.
    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        if self.cursor >= self.batches.len() {
            return None;
        }
        let (batch_epoch, _) = &self.batches[self.cursor];
        if *batch_epoch != self.current_epoch {
            return None;
        }
        let indices = self.batches[self.cursor].1.clone();
        self.cursor += 1;
        Some(indices)
    }

    /// Advance to the next epoch. Called by `BatchSource::reshuffle`.
    pub(crate) fn advance_epoch(&mut self) {
        self.current_epoch += 1;
    }

    /// Number of mini-batches remaining across all unprocessed epochs.
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
    fn replay_sampler_yields_recorded_batches_in_epoch_order() {
        let all_staids: Vec<crate::data::ids::Staid> = (0..10)
            .map(|i| crate::data::ids::Staid::new(&format!("STAID_{i:02}")))
            .collect();
        // Trace: epoch 1 has 2 batches, epoch 2 has 1 batch.
        let recorded: Vec<(u32, Vec<crate::data::ids::Staid>)> = vec![
            (1, vec![all_staids[3].clone(), all_staids[1].clone()]),
            (1, vec![all_staids[7].clone(), all_staids[2].clone()]),
            (2, vec![all_staids[5].clone(), all_staids[8].clone()]),
        ];

        let mut replay = ReplaySampler::new(recorded, &all_staids);

        // Driver: epoch 1 starts.
        replay.advance_epoch(); // current_epoch = 1
        assert_eq!(replay.next_batch(), Some(vec![3, 1]));
        assert_eq!(replay.next_batch(), Some(vec![7, 2]));
        // Next batch is epoch 2; should stop yielding for epoch 1.
        assert_eq!(replay.next_batch(), None);

        // Driver: epoch 2 starts.
        replay.advance_epoch(); // current_epoch = 2
        assert_eq!(replay.next_batch(), Some(vec![5, 8]));
        // Trace exhausted.
        assert_eq!(replay.next_batch(), None);

        // Driver: epoch 3 starts (DDRS may iterate more outer epochs than
        // the trace covers — should just yield None forever).
        replay.advance_epoch();
        assert_eq!(replay.next_batch(), None);
    }

    #[test]
    fn replay_sampler_via_batch_source_reshuffle_advances_epoch() {
        let all_staids = vec![
            crate::data::ids::Staid::new("A"),
            crate::data::ids::Staid::new("B"),
        ];
        let recorded = vec![
            (1, vec![all_staids[0].clone()]),
            (2, vec![all_staids[1].clone()]),
        ];
        let mut source = BatchSource::Replay(ReplaySampler::new(recorded, &all_staids));

        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(0);

        // Epoch 1.
        source.reshuffle(&mut rng);
        assert_eq!(source.next_batch(), Some(vec![0]));
        assert_eq!(source.next_batch(), None);
        // Epoch 2.
        source.reshuffle(&mut rng);
        assert_eq!(source.next_batch(), Some(vec![1]));
        assert_eq!(source.next_batch(), None);
    }

    #[test]
    fn replay_sampler_rejects_unknown_staid() {
        let all_staids = vec![crate::data::ids::Staid::new("KNOWN")];
        let recorded = vec![(1, vec![crate::data::ids::Staid::new("UNKNOWN")])];
        let result = std::panic::catch_unwind(|| {
            let _ = ReplaySampler::new(recorded, &all_staids);
        });
        assert!(result.is_err(),
            "ReplaySampler::new should panic when a recorded STAID is not in the dataset");
    }
}
