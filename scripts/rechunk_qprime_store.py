#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "zarr>=3", "icechunk"]
# ///
"""Rechunk a DDR Q' icechunk store to the training-friendly layout.

The routing trainer reads a 90-day window for a few hundred divides per batch.
A store chunked `(divides, full time)` (e.g. (200, 14976)) forces every batch to
decompress whole 41-year rows and made training 6x slower (2026-09-13). The
retrospective store's `(3080, 468)` layout is the one the reader was measured
on. This copies Qr, divide_id and time (with attrs) into a new repo with that
layout, streaming one row block at a time.

    scripts/rechunk_qprime_store.py <src.ic> <dst.ic> [--chunks 3080 468]
"""
import argparse, sys, time
import numpy as np, zarr, icechunk

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--chunks", type=int, nargs=2, default=[3080, 468])
a = ap.parse_args()
src = zarr.open_group(icechunk.Repository.open(icechunk.local_filesystem_storage(a.src)).readonly_session("main").store, mode="r")
qr = src["Qr"]; n, t = qr.shape
print(f"src Qr {qr.shape} chunks {qr.chunks}; time-major? {'no' if src['divide_id'].shape[0] == n else 'YES'}", flush=True)
repo = icechunk.Repository.create(icechunk.local_filesystem_storage(a.dst))
sess = repo.writable_session("main"); g = zarr.create_group(sess.store, overwrite=True)
g.attrs.update(dict(src.attrs))
for name in ("divide_id", "time"):
    arr = g.create_array(name, shape=src[name].shape, dtype=src[name].dtype, chunks=src[name].shape)
    arr[:] = src[name][:]; arr.attrs.update(dict(src[name].attrs))
out = g.create_array("Qr", shape=(n, t), dtype=qr.dtype, chunks=tuple(a.chunks), compressors=zarr.codecs.ZstdCodec(level=3))
out.attrs.update(dict(qr.attrs))
step = a.chunks[0]; t0 = time.time()
for i in range(0, n, step):
    out[i:i + step, :] = qr[i:i + step, :]
    if (i // step) % 8 == 0:
        print(f"  rows {i + step:>7}/{n}  {time.time() - t0:6.0f} s", flush=True)
sess.commit(f"rechunked from {a.src} to {a.chunks}")
print(f"done in {time.time() - t0:.0f} s -> {a.dst}")
