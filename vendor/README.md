# Vendored cubecl + burn patches for SP-7

SP-7 (cuSPARSE stream-share + zero-copy x wrap) needs two tiny `pub`
accessors that are `pub(crate)` on crates.io today:

| Crate          | Version | Added accessor                                  |
|----------------|---------|-------------------------------------------------|
| `cubecl-cuda`  | 0.10.0  | `pub fn CudaServer::stream() -> CUstream`       |
| `burn-cubecl`  | 0.21.0  | `pub fn CubeTensor::from_handle(...) -> Self`   |

## Forks

ddrs's `Cargo.toml` `[patch.crates-io]` points at fork branches on
github.com/taddyb so anyone can `git clone ddrs && cargo build` without
needing local checkouts of cubecl or burn:

- `cubecl`: <https://github.com/taddyb/cubecl> branch `ddrs-release`
  (based on upstream `v0.10.0` tag).
- `burn`: <https://github.com/taddyb/burn> branch `ddrs-sp7-primitive-ctor`
  (based on upstream `v0.21.0` tag).

## Local fork development

If you're iterating on the fork branches:

1. Make your changes on the appropriate branch in `~/projects/cubecl` or
   `~/projects/burn`.
2. `git push --force-with-lease origin <branch>` to publish the new commit.
3. In ddrs: `cargo update -p cubecl` (or whichever crate you changed) to
   pull the new commit into ddrs's `Cargo.lock`.

This is slightly more friction than the old "patch points at local path,
edits take effect on next `cargo build`" setup, but it's the cost of
having ddrs be buildable for anyone else who clones the repo.

If you want the local-path workflow back temporarily, edit
`Cargo.toml`'s `[patch.crates-io]` block to use `path = "..."` instead
of `git = ...`. Just don't commit that change — it breaks the public
build.

## Diffs

Both patches are ≤ 50 lines. The single commit on each fork branch shows
the exact diff.

## Upgrading

When upgrading `burn` / `cubecl` major versions in ddrs:

1. On the fork: `git fetch upstream && git rebase upstream/<new-tag>`.
2. Re-apply the patch (small risk of merge conflict if touched files moved).
3. `git push --force-with-lease origin <branch>`.
4. In ddrs `Cargo.toml`, bump the upstream version constraint if needed.
5. `cargo update -p cubecl -p burn-cubecl` (etc.) to pin the new commits.

## Upstream PR (SP-8 cleanup)

The accessors are vanilla "expose what already exists as `pub`" PRs.
Plan to upstream both as SP-8 once SP-7 stabilizes. After merge + release,
delete `[patch.crates-io]` entries and this file.
