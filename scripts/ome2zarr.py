#!/usr/bin/env python3
"""
ome_to_zarr.py

Convert a CODEX OME-TIFF (typically axes like CYX) into a chunked Zarr directory
for fast random patch reads.

Key goals:
- Avoid loading the whole image into RAM.
- Write chunk-by-chunk.
- Be robust to tifffile.aszarr() returning either a zarr.Array-like object or a zarr.Group.

Output:
  <outdir>/<sample_id>.codex_raw.zarr/
    └── 0   (zarr array dataset)

Example:
  python ome_to_zarr.py /path/to/sample.ome.tiff --outdir /path/to/zarr --chunks 1,512,512 --overwrite
"""

import argparse
import json
import os
import shutil
from typing import Iterator, Tuple, Union, Optional

import numpy as np
import tifffile as tiff
import zarr
from numcodecs import Blosc


def parse_chunks(s: str) -> Tuple[int, ...]:
    """Parse a chunk string like '1,512,512' into a tuple of ints."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty --chunks. Example: --chunks 1,512,512")
    try:
        return tuple(int(p) for p in parts)
    except Exception as e:
        raise ValueError(f"Invalid --chunks '{s}'. Expected comma-separated ints.") from e


def iter_chunk_slices(shape: Tuple[int, ...], chunks: Tuple[int, ...]) -> Iterator[Tuple[slice, ...]]:
    """
    Yield N-dimensional slice tuples that cover the entire array by chunks.

    For shape=(C,Y,X) and chunks=(1,512,512), yields slices like:
      (slice(c0,c1), slice(y0,y1), slice(x0,x1))
    """
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


def sample_id_from_path(path: str) -> str:
    """
    Convert '/path/foo.ome.tiff' -> 'foo'
            '/path/foo.tiff'     -> 'foo'
            '/path/foo.ome.tif'  -> 'foo'
    """
    base = os.path.splitext(os.path.basename(path))[0]  # foo.ome or foo
    if base.endswith(".ome"):
        base = base[:-4]
    return base


def open_tiff_as_zarr_array(in_path: str, series_index: int = 0) -> Tuple[zarr.Array, Optional[str], Tuple[int, ...], np.dtype]:
    """
    Open an OME-TIFF using tifffile and expose it as a Zarr Array for reading,
    without loading the full pixel data into RAM.

    Important: tifffile.series[i].aszarr() may produce a store that, when opened with zarr,
    yields either:
      - a zarr.Array directly, OR
      - a zarr.Group containing one or more arrays (often key '0').

    This function normalizes the result to a zarr.Array.
    """
    with tiff.TiffFile(in_path) as tf:
        if series_index < 0 or series_index >= len(tf.series):
            raise IndexError(f"series_index {series_index} out of range (num_series={len(tf.series)})")

        series = tf.series[series_index]
        axes = getattr(series, "axes", None)
        shape = tuple(series.shape)
        dtype = series.dtype

        # Create a Zarr-store view of the TIFF data (read-only).
        src_store = series.aszarr()

    # Open with zarr (outside TiffFile context is okay; the store is independent).
    src_obj = zarr.open(src_store, mode="r")

    # Case 1: zarr.Array-like (has 'shape' attribute)
    if hasattr(src_obj, "shape") and hasattr(src_obj, "__getitem__"):
        src_arr = src_obj  # type: ignore
        return src_arr, axes, shape, dtype

    # Case 2: zarr.Group-like (needs a key to access an array)
    if isinstance(src_obj, zarr.hierarchy.Group):
        # Prefer conventional dataset name "0" if present.
        if "0" in src_obj:
            src_arr = src_obj["0"]
            return src_arr, axes, shape, dtype

        # Otherwise pick the first array key in the group.
        array_keys = list(src_obj.array_keys())
        if not array_keys:
            raise RuntimeError("aszarr() returned a Zarr Group but it contains no arrays.")
        src_arr = src_obj[array_keys[0]]
        return src_arr, axes, shape, dtype

    raise RuntimeError(f"Unexpected Zarr object type from aszarr(): {type(src_obj)}")


def main():
    ap = argparse.ArgumentParser(
        description="Convert CODEX OME-TIFF to chunked Zarr for fast random patch access."
    )
    ap.add_argument("ome_tiff", help="Path to input .ome.tiff/.ome.tif/.tiff")
    ap.add_argument("--outdir", default=None, help="Output directory (default: alongside input file)")
    ap.add_argument("--name", default=None, help="Output name prefix (default: derived from input filename)")
    ap.add_argument("--series", type=int, default=0, help="TIFF series index (default: 0)")
    ap.add_argument("--chunks", default="1,512,512", help='Chunk shape, e.g. "1,512,512" (default: 1,512,512)')
    ap.add_argument("--clevel", type=int, default=3, help="zstd compression level for Blosc (default: 3)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output zarr directory")
    ap.add_argument("--meta_json", default=None, help="Optional path to codex_meta.json to store as Zarr attrs")
    ap.add_argument("--dataset_name", default="0", help='Dataset name inside Zarr group (default: "0")')
    args = ap.parse_args()

    in_path = os.path.abspath(args.ome_tiff)
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input not found: {in_path}")

    outdir = os.path.abspath(args.outdir) if args.outdir else os.path.dirname(in_path)
    os.makedirs(outdir, exist_ok=True)

    prefix = args.name if args.name else sample_id_from_path(in_path)
    out_zarr = os.path.join(outdir, f"{prefix}.codex_raw.zarr")

    if os.path.exists(out_zarr):
        if args.overwrite:
            shutil.rmtree(out_zarr)
        else:
            raise FileExistsError(
                f"Output exists: {out_zarr}\n"
                f"Use --overwrite to replace it."
            )

    chunks = parse_chunks(args.chunks)

    # Open TIFF-backed data as a Zarr Array (robust to Group/Array differences).
    src, axes, shape, dtype = open_tiff_as_zarr_array(in_path, series_index=args.series)

    if len(chunks) != len(shape):
        raise ValueError(
            f"--chunks {chunks} ndim={len(chunks)} does not match data shape {shape} ndim={len(shape)}.\n"
            f"Detected axes={axes}. Example for CYX: --chunks 1,512,512"
        )

    # Create destination Zarr array.
    # Using Blosc(zstd) is a good balance for speed + size.
    compressor = Blosc(cname="zstd", clevel=int(args.clevel), shuffle=Blosc.BITSHUFFLE)
    store = zarr.DirectoryStore(out_zarr)
    root = zarr.group(store=store)

    dst = root.create_dataset(
        args.dataset_name,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        overwrite=True,
    )

    # Store useful provenance and structural metadata as Zarr attributes.
    root.attrs["input_file"] = in_path
    root.attrs["series_index"] = int(args.series)
    root.attrs["axes_order"] = axes
    root.attrs["shape"] = shape
    root.attrs["dtype"] = str(dtype)
    root.attrs["chunks"] = chunks
    root.attrs["compressor"] = {
        "codec": "Blosc",
        "cname": "zstd",
        "clevel": int(args.clevel),
        "shuffle": "BITSHUFFLE",
    }

    # Optionally embed your Step 1 meta JSON into the Zarr attrs (nice for provenance).
    if args.meta_json:
        meta_path = os.path.abspath(args.meta_json)
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"--meta_json not found: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            root.attrs["codex_meta_json"] = json.load(f)

    # Copy data chunk-by-chunk to keep memory usage bounded.
    # This is I/O bound on HPC filesystems.
    chunk_count = 0
    for slc in iter_chunk_slices(shape, chunks):
        dst[slc] = src[slc]
        chunk_count += 1
        if chunk_count % 200 == 0:
            print(f"[INFO] wrote {chunk_count} chunks...")

    print("Wrote:")
    print("  ", out_zarr)
    print("[INFO] dataset path inside zarr:", os.path.join(out_zarr, args.dataset_name))
    print("[INFO] axes:", axes, "shape:", shape, "dtype:", dtype, "chunks:", chunks)


if __name__ == "__main__":
    main()
