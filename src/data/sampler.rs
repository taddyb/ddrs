//! Batch index samplers.
//!
//! Mirrors `torch.utils.data.{RandomSampler, SequentialSampler}` for our
//! batching needs. Not PyTorch-bit-identical — the project's verification
//! bar doesn't require it. Determinism via `rand::SeedableRng`.

use rand::seq::SliceRandom;
use rand::Rng;

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
}
