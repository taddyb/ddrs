# Vendored cubecl + burn patches for SP-7

SP-7 (cuSPARSE stream-share + zero-copy x wrap) needs two tiny `pub`
accessors that are `pub(crate)` on crates.io today:

| Crate          | Version | Added accessor                                  |
|----------------|---------|-------------------------------------------------|
| `cubecl-cuda`  | 0.10.0  | `pub fn CudaServer::stream() -> CUstream`       |
| `burn-cubecl`  | 0.21.0  | `pub fn CubeTensor::from_handle(...) -> Self`   |

## Forks

- cubecl: `taddyb/cubecl` branch `ddrs-sp7-stream-accessor`
  - Forked from upstream commit `7cf203735e095e640a2c03b2400d0faa03196bb4` (= tag `v0.10.0`).
- burn: `taddyb/burn` branch `ddrs-sp7-primitive-ctor`
  - Forked from upstream commit `546cacb55fe00168854d19bdf0a5d79bd8060e03` (= tag `v0.21.0`).

## Diffs

Both patches are ≤ 50 lines. The single commit on each branch shows the
exact diff.

## Upgrading

When upgrading `burn` / `cubecl` major versions in ddrs:
1. On the fork: `git fetch upstream && git rebase upstream/<new-tag>`.
2. Re-apply the patch (small risk of merge conflict if touched files moved).
3. `git push --force-with-lease origin <branch>`.
4. In ddrs `Cargo.toml`, bump the upstream version constraint if needed.

## Upstream PR (SP-8 cleanup)

The accessors are vanilla "expose what already exists as `pub`" PRs.
Plan to upstream both as SP-8 once SP-7 stabilizes. After merge + release,
delete `[patch.crates-io]` entries and this file.
