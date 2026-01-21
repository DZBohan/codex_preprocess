#!/usr/bin/env python3
"""
codex_step3_norm_zarr.py

Step 3: Per-channel robust normalization for CODEX stored in Zarr.

Pipeline (per channel):
  1) clip to [p_low, p_high] percentiles
  2) apply asinh(x / c)  (or log1p)
  3) optional min-max to [0, 1]

Design goals:
- Channel-independent (avoid learning exposure/background artifacts).
- No segmentation assumptions.
- Chunk-by-chunk processing (HPC-friendly, bounded memory).
- Preserve provenance by copying attrs from raw to norm.

Input Zarr structure (from your Step 2):
  <sample>.codex_raw.zarr/
    └── 0   (array with axes like CYX)

Output:
  <sample>.codex_norm.zarr/
    └── 0   (float32 by default)

Example:
  python codex_step3_norm_zarr.py \
    --in_zarr /path/sample.codex_raw.zarr \
    --out_zarr /path/sample.codex_norm.zarr \
    --p_low 1 --p_high 99 \
    --transform asinh --c 5 \
    --minmax \
    --overwrite
"""

import argparse
import math
import os
import shutil
from typing import Iterator, Tuple, Optional, List, Dict

import numpy as np
import zarr
from numcodecs import Blosc


def iter_chunk_slices(shape: Tuple[int, ...], chunks: Tuple[int, ...]) -> Iterator[Tuple[slice, ...]]:
    """Yield N-D slices covering the array by chunks."""
    if len(shape) != len(chunks):
        raise ValueError(f"shape ndim {len(shape)} != chunks ndim {len(chunks)}")
    for c in chunks:
        if c <= 0:
            raise ValueError(f"Invalid chunk size: {chunks}")

    starts_per_dim = [list(range(0, dim, csz)) for dim, csz in zip(shape, chunks)]

    def rec(i: int, prefix: list):
        if i == len(shape):
            yield tuple(prefix)
            return
        dim = shape[i]
        csz = chunks[i]
        for st in starts_per_dim[i]:
            en = min(st + csz, dim)
            prefix.append(slice(st, en))
            yield from rec(i + 1, prefix)
            prefix.pop()

    yield from rec(0, [])


def parse_dataset_path(zarr_path: str, dataset: str) -> Tuple[zarr.Group, zarr.Array]:
    """Open a Zarr group and return (group, dataset array)."""
    root = zarr.open(zarr_path, mode="r")
    if not isinstance(root, zarr.hierarchy.Group):
        raise ValueError(f"Expected a Zarr group at {zarr_path}, got {type(root)}")
    if dataset not in root:
        raise KeyError(f"Dataset '{dataset}' not found in {zarr_path}. Keys: {list(root.keys())}")
    arr = root[dataset]
    return root, arr


def axes_index(axes: str, char: str) -> int:
    """Find axis index for a given axis letter, e.g., 'C' in 'CYX'."""
    if axes is None:
        raise ValueError("Missing axes_order in attrs. Provide --axes_order explicitly.")
    if char not in axes:
        raise ValueError(f"Axis '{char}' not found in axes_order='{axes}'")
    return axes.index(char)


def compute_channel_percentiles(
    arr: zarr.Array,
    axes: str,
    p_low: float,
    p_high: float,
    max_samples_per_channel: int = 2_000_000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-channel percentiles by random sampling across pixels.

    Why sampling?
    - Full percentile over tens of millions of pixels per channel is expensive.
    - Sampling is typically sufficient for robust clipping.

    Returns:
      lo[C], hi[C] as float64 arrays.
    """
    rng = np.random.default_rng(seed)
    c_axis = axes_index(axes, "C")
    shape = arr.shape
    n_channels = shape[c_axis]

    # Determine Y and X axes (assume typical CYX, but we support any ordering containing C,Y,X)
    y_axis = axes_index(axes, "Y")
    x_axis = axes_index(axes, "X")

    Y = shape[y_axis]
    X = shape[x_axis]
    total_pixels = Y * X

    # Choose sample size
    n = min(max_samples_per_channel, total_pixels)
    # Random (y, x) coordinates
    ys = rng.integers(0, Y, size=n, dtype=np.int64)
    xs = rng.integers(0, X, size=n, dtype=np.int64)

    lo = np.zeros((n_channels,), dtype=np.float64)
    hi = np.zeros((n_channels,), dtype=np.float64)

    # Build an index template for fancy indexing: [C, Y, X] but axis order may differ.
    # We'll read samples channel-by-channel to keep memory bounded.
    for c in range(n_channels):
        # Construct slicers for arr[...] to fetch sampled pixels for this channel.
        # We cannot directly do arr[c, ys, xs] unless axes is exactly CYX.
        # Instead, we build a tuple of indices in the correct axis order.
        idx = [slice(None)] * arr.ndim
        idx[c_axis] = c
        idx[y_axis] = ys
        idx[x_axis] = xs

        # This returns a 1D array of length n
        samples = np.asarray(arr[tuple(idx)], dtype=np.float64).ravel()

        lo[c] = np.percentile(samples, p_low)
        hi[c] = np.percentile(samples, p_high)

        # Guard: if channel is almost constant, hi can equal lo.
        if not np.isfinite(lo[c]) or not np.isfinite(hi[c]):
            raise ValueError(f"Non-finite percentile for channel {c}: lo={lo[c]}, hi={hi[c]}")
        if hi[c] < lo[c]:
            lo[c], hi[c] = hi[c], lo[c]
        if hi[c] == lo[c]:
            # Expand slightly to avoid divide-by-zero later
            hi[c] = lo[c] + 1.0

    return lo, hi


def apply_transform(x: np.ndarray, transform: str, c: float) -> np.ndarray:
    """Apply intensity transform to a numpy array (float32/float64)."""
    if transform == "asinh":
        return np.arcsinh(x / c)
    if transform == "log1p":
        return np.log1p(x)
    if transform == "none":
        return x
    raise ValueError(f"Unknown transform: {transform}")


def main():
    ap = argparse.ArgumentParser(description="Step 3: per-channel normalization for CODEX Zarr.")
    ap.add_argument("--in_zarr", required=True, help="Input raw zarr path (e.g., sample.codex_raw.zarr)")
    ap.add_argument("--out_zarr", required=True, help="Output normalized zarr path (e.g., sample.codex_norm.zarr)")
    ap.add_argument("--dataset", default="0", help='Dataset name inside zarr (default: "0")')
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output zarr if exists")

    # Axes
    ap.add_argument("--axes_order", default=None, help='Axes string like "CYX". If omitted, read from attrs.')

    # Clipping + sampling
    ap.add_argument("--p_low", type=float, default=1.0, help="Lower percentile for clipping (default: 1)")
    ap.add_argument("--p_high", type=float, default=99.0, help="Upper percentile for clipping (default: 99)")
    ap.add_argument("--max_samples_per_channel", type=int, default=2_000_000,
                    help="Max random samples per channel to estimate percentiles (default: 2,000,000)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for sampling (default: 0)")

    # Transform
    ap.add_argument("--transform", choices=["asinh", "log1p", "none"], default="asinh",
                    help="Intensity transform after clipping (default: asinh)")
    ap.add_argument("--c", type=float, default=5.0, help="Scale factor for asinh(x/c) (default: 5)")

    # Scaling
    ap.add_argument("--minmax", action="store_true", help="Apply per-channel min-max scaling to [0,1] after transform")
    ap.add_argument("--out_dtype", default="float32", choices=["float32", "float16"],
                    help="Output dtype (default: float32)")

    # Compression
    ap.add_argument("--clevel", type=int, default=3, help="zstd compression level for Blosc (default: 3)")
    ap.add_argument("--shuffle", choices=["bitshuffle", "shuffle", "noshuffle"], default="bitshuffle",
                    help="Blosc shuffle mode (default: bitshuffle)")

    args = ap.parse_args()

    in_zarr = os.path.abspath(args.in_zarr)
    out_zarr = os.path.abspath(args.out_zarr)

    if not os.path.exists(in_zarr):
        raise FileNotFoundError(f"Input zarr not found: {in_zarr}")

    if os.path.exists(out_zarr):
        if args.overwrite:
            shutil.rmtree(out_zarr)
        else:
            raise FileExistsError(f"Output exists: {out_zarr} (use --overwrite)")

    # Open input
    in_root, in_arr = parse_dataset_path(in_zarr, args.dataset)

    # Determine axes order
    axes = args.axes_order
    if axes is None:
        axes = in_root.attrs.get("axes_order", None)
    if axes is None:
        raise ValueError("Could not determine axes_order from attrs. Provide --axes_order CYX explicitly.")

    # Validate axes contain C,Y,X
    _ = axes_index(axes, "C")
    _ = axes_index(axes, "Y")
    _ = axes_index(axes, "X")

    shape = tuple(in_arr.shape)
    chunks = tuple(in_arr.chunks)
    in_dtype = in_arr.dtype

    print("[INFO] Input:", in_zarr)
    print("[INFO] Dataset:", args.dataset)
    print("[INFO] Axes:", axes)
    print("[INFO] Shape:", shape, "Chunks:", chunks, "Dtype:", in_dtype)

    # 1) Estimate per-channel percentiles by sampling
    print("[INFO] Estimating per-channel percentiles by random sampling...")
    lo, hi = compute_channel_percentiles(
        in_arr, axes, args.p_low, args.p_high,
        max_samples_per_channel=args.max_samples_per_channel,
        seed=args.seed,
    )
    print("[INFO] Percentiles estimated.")
    print("[INFO] Example channel 0: lo=", float(lo[0]), "hi=", float(hi[0]))

    # 2) Create output Zarr
    store = zarr.DirectoryStore(out_zarr)
    out_root = zarr.group(store=store)

    shuffle_map = {
        "bitshuffle": Blosc.BITSHUFFLE,
        "shuffle": Blosc.SHUFFLE,
        "noshuffle": Blosc.NOSHUFFLE,
    }
    compressor = Blosc(cname="zstd", clevel=int(args.clevel), shuffle=shuffle_map[args.shuffle])

    out_dtype = np.float32 if args.out_dtype == "float32" else np.float16

    out_arr = out_root.create_dataset(
        args.dataset,
        shape=shape,
        chunks=chunks,           # keep the same chunking as Step 2 for patch access
        dtype=out_dtype,
        compressor=compressor,
        overwrite=True,
    )

    # Copy attrs + add normalization recipe
    out_root.attrs.update(dict(in_root.attrs))
    out_root.attrs["norm_recipe"] = {
        "p_low": float(args.p_low),
        "p_high": float(args.p_high),
        "transform": args.transform,
        "c": float(args.c),
        "minmax": bool(args.minmax),
        "sampling": {
            "max_samples_per_channel": int(args.max_samples_per_channel),
            "seed": int(args.seed),
        },
    }
    # Save per-channel percentile values (useful for reproducibility)
    out_root.attrs["norm_percentiles"] = {
        "lo": lo.tolist(),
        "hi": hi.tolist(),
    }

    # 3) Chunk-by-chunk normalize and write
    c_axis = axes_index(axes, "C")

    print("[INFO] Normalizing chunk-by-chunk...")
    chunk_count = 0
    for slc in iter_chunk_slices(shape, chunks):
        block = np.asarray(in_arr[slc], dtype=np.float32)

        # Determine which channels are included in this block slice
        c_sl = slc[c_axis]
        c_start = 0 if c_sl.start is None else c_sl.start
        c_stop = shape[c_axis] if c_sl.stop is None else c_sl.stop

        # Apply per-channel operations within this block
        # block has the same axis order as input; we need to broadcast lo/hi properly.
        # We'll reshape per-channel vectors into broadcastable shape.
        nch = c_stop - c_start
        lo_v = lo[c_start:c_stop].astype(np.float32)
        hi_v = hi[c_start:c_stop].astype(np.float32)

        # Build broadcast shape: [1,1,1...] with channel dimension = nch
        bshape = [1] * block.ndim
        bshape[c_axis] = nch
        lo_b = lo_v.reshape(bshape)
        hi_b = hi_v.reshape(bshape)

        # Clip
        block = np.clip(block, lo_b, hi_b)

        # Shift to non-negative (optional but nice before log/asinh)
        # After clipping, values are in [lo, hi]. Subtract lo to set background near 0.
        block = block - lo_b

        # Transform
        block = apply_transform(block, args.transform, args.c)

        # Optional min-max to [0,1] per channel (within global channel range after transform)
        if args.minmax:
            # After shift+transform, per-channel max corresponds to transform(hi-lo)
            max_v = apply_transform((hi_v - lo_v).astype(np.float32), args.transform, args.c)
            max_v = np.maximum(max_v, 1e-6).reshape(bshape)
            block = block / max_v

        out_arr[slc] = block.astype(out_dtype, copy=False)

        chunk_count += 1
        if chunk_count % 200 == 0:
            print(f"[INFO] wrote {chunk_count} chunks...")

    print("Wrote:")
    print("  ", out_zarr)
    print("[INFO] Done. Output dtype:", out_dtype)


if __name__ == "__main__":
    main()
