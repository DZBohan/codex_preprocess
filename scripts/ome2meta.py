#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tifffile as tiff
from lxml import etree


def _safe_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def parse_ome_xml(ome_xml: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    Parse OME-XML to extract:
      - channel names (marker names)
      - pixel sizes (PhysicalSizeX/Y) and their units if present
      - image dimensions if present (SizeX/SizeY/SizeC/SizeZ/SizeT)
    """
    root = etree.fromstring(ome_xml.encode("utf-8"))

    # OME namespace handling
    # OME-XML typically has default namespace like: {http://www.openmicroscopy.org/Schemas/OME/2016-06}
    nsmap = root.nsmap.copy()
    if None in nsmap:
        nsmap["ome"] = nsmap.pop(None)
    elif "ome" not in nsmap:
        nsmap["ome"] = "http://www.openmicroscopy.org/Schemas/OME/2016-06"

    # Use first Image as default
    image = root.find(".//ome:Image", namespaces=nsmap)
    meta: Dict[str, Any] = {}

    channel_names: List[str] = []

    if image is not None:
        pixels = image.find(".//ome:Pixels", namespaces=nsmap)
        if pixels is not None:
            # Physical pixel sizes
            psx = pixels.get("PhysicalSizeX")
            psy = pixels.get("PhysicalSizeY")
            psx_unit = pixels.get("PhysicalSizeXUnit")
            psy_unit = pixels.get("PhysicalSizeYUnit")

            meta["physical_size_x"] = _safe_float(psx)
            meta["physical_size_y"] = _safe_float(psy)
            meta["physical_size_x_unit"] = psx_unit
            meta["physical_size_y_unit"] = psy_unit

            # Image sizes in OME metadata (may exist)
            for k in ["SizeX", "SizeY", "SizeC", "SizeZ", "SizeT"]:
                v = pixels.get(k)
                meta[k.lower()] = int(v) if v is not None and v.isdigit() else v

            # Channel list
            channels = pixels.findall(".//ome:Channel", namespaces=nsmap)
            for ch in channels:
                # Most common place: Channel/@Name
                name = ch.get("Name")

                # Some datasets put marker name into other fields; try fallbacks
                if not name:
                    name = ch.get("ID")

                if not name:
                    name = "UNKNOWN"

                channel_names.append(name)

    return channel_names, meta


def infer_axes_and_shape(tf: tiff.TiffFile) -> Tuple[Optional[str], Tuple[int, ...], Optional[str]]:
    """
    Try to infer axes order and shape from tifffile series.
    Returns (axes_order, shape, dtype_str)
    """
    try:
        series = tf.series[0]
        axes = getattr(series, "axes", None)  # e.g. "TCZYX" or "CYX" or "YXC"
        shape = tuple(series.shape)
        dtype = str(series.dtype) if hasattr(series, "dtype") else None
        return axes, shape, dtype
    except Exception:
        return None, tuple(), None


def map_shape_to_image_size(axes: Optional[str], shape: Tuple[int, ...]) -> Dict[str, Any]:
    """
    Convert (axes, shape) to image size summary:
      - image_size: [Y, X] if available
      - n_channels: C if available
    """
    out: Dict[str, Any] = {}
    if not axes or not shape or len(axes) != len(shape):
        return out

    axis_to_dim = {ax: dim for ax, dim in zip(axes, shape)}
    # OME commonly uses X/Y; tifffile axes may use "YX" for spatial
    y = axis_to_dim.get("Y")
    x = axis_to_dim.get("X")
    c = axis_to_dim.get("C")

    if y is not None and x is not None:
        out["image_size"] = [int(y), int(x)]
    if c is not None:
        out["n_channels"] = int(c)

    # Keep full dims too (useful when Z/T exist)
    out["axes_dims"] = {ax: int(axis_to_dim[ax]) for ax in axis_to_dim}
    return out


def normalize_pixel_size_um(meta_from_xml: Dict[str, Any]) -> Optional[List[Optional[float]]]:
    """
    Try to return pixel size in micrometers [ps_y_um, ps_x_um].
    If units are not µm, we keep raw numbers but also note unit.
    """
    psx = meta_from_xml.get("physical_size_x")
    psy = meta_from_xml.get("physical_size_y")
    ux = meta_from_xml.get("physical_size_x_unit")
    uy = meta_from_xml.get("physical_size_y_unit")

    # If units are micrometer-like, treat as µm. Otherwise we still return raw values.
    # Many OME files use "µm" or "um" or "micrometer".
    def is_um(u: Optional[str]) -> bool:
        if not u:
            return False
        u2 = u.lower()
        return ("µm" in u2) or (u2 == "um") or ("micrometer" in u2)

    if psx is None and psy is None:
        return None

    # We return [y, x] ordering to match image_size [Y, X] convention.
    # If unit not µm, still return number but your pipeline should note the unit.
    return [psy, psx]


def write_channels_tsv(channel_names: List[str], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("channel_index\tmarker_name\n")
        for i, name in enumerate(channel_names):
            f.write(f"{i}\t{name}\n")


def main():
    ap = argparse.ArgumentParser(description="Extract channels.tsv and codex_meta.json from CODEX OME-TIFF.")
    ap.add_argument("ome_tiff", help="Path to .ome.tiff")
    ap.add_argument("--outdir", default=None, help="Output directory (default: alongside input file)")
    args = ap.parse_args()

    in_path = args.ome_tiff
    if args.outdir is None:
        outdir = os.path.dirname(os.path.abspath(in_path))
    else:
        outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    base = os.path.splitext(os.path.basename(in_path))[0]
    # Handle .ome.tiff -> base name cleanup
    if base.endswith(".ome"):
        base = base[:-4]

    channels_path = os.path.join(outdir, f"{base}.channels.tsv")
    meta_path = os.path.join(outdir, f"{base}.codex_meta.json")

    with tiff.TiffFile(in_path) as tf:
        ome_xml = tf.ome_metadata
        axes_order, shape, dtype_str = infer_axes_and_shape(tf)

    channel_names: List[str] = []
    meta_from_xml: Dict[str, Any] = {}
    if ome_xml:
        channel_names, meta_from_xml = parse_ome_xml(ome_xml)

    # Fallback if OME channel list is missing: infer C from axes and create placeholders
    if (not channel_names) and axes_order and ("C" in axes_order) and shape:
        axis_to_dim = {ax: dim for ax, dim in zip(axes_order, shape)}
        c = axis_to_dim.get("C", 0)
        channel_names = [f"CH{i}" for i in range(int(c))]

    # Build codex_meta.json
    meta: Dict[str, Any] = {}
    meta["input_file"] = os.path.abspath(in_path)
    meta["axes_order"] = axes_order
    meta["dtype"] = dtype_str

    # Size summary
    meta.update(map_shape_to_image_size(axes_order, shape))

    # Pixel size
    meta["pixel_size_um"] = normalize_pixel_size_um(meta_from_xml)
    meta["pixel_size_unit_note"] = {
        "x_unit": meta_from_xml.get("physical_size_x_unit"),
        "y_unit": meta_from_xml.get("physical_size_y_unit"),
        "raw_physical_size_x": meta_from_xml.get("physical_size_x"),
        "raw_physical_size_y": meta_from_xml.get("physical_size_y"),
    }

    # OME-declared sizes (if any)
    for k in ["sizex", "sizey", "sizec", "sizez", "sizet"]:
        if k in meta_from_xml and meta_from_xml[k] is not None:
            meta[f"ome_{k}"] = meta_from_xml[k]

    # Write outputs
    write_channels_tsv(channel_names, channels_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Wrote:")
    print("  ", channels_path)
    print("  ", meta_path)
    if not ome_xml:
        print("NOTE: No OME-XML metadata found. Channel names may be placeholders.")


if __name__ == "__main__":
    main()
