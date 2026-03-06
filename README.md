## 1. Introduction

This repository provides a minimal, explicit, and reproducible preprocessing pipeline that standardizes CODEX (CO-Detection by indEXing) multiplexed imaging data before modeling or biological assumptions are introduced.

The output of this repository is **model-ready tensors** that can be directly consumed by deep learning pipelines, including cell segmentation, phenotyping, and spatial analysis models.

The pipeline performs three sequential steps:

1. **Metadata extraction and semantic freezing** — parse OME-XML embedded in the TIFF to capture channel (marker) names, pixel sizes, and image dimensions.
2. **Conversion to chunked, random-access storage (Zarr)** — re-encode the OME-TIFF as a Zarr directory with configurable chunk sizes, enabling fast random patch reads without loading the entire image into memory.
3. **Per-channel robust intensity normalization** — clip, transform (arcsinh), and optionally min-max scale each channel independently so that downstream models receive consistent, well-conditioned input.

All scripts are designed for **HPC / SLURM** environments and ship with paired `.sh` job submission templates.

## 2. Dependencies


Readme · MD
Copy

# CODEX Spatial Preprocessing Pipeline

---

## 1. Introduction

This repository provides a minimal, explicit, and reproducible preprocessing pipeline that standardizes CODEX (CO-Detection by indEXing) multiplexed imaging data before modeling or biological assumptions are introduced.

The output of this repository is **model-ready tensors** that can be directly consumed by deep learning pipelines, including cell segmentation, phenotyping, and spatial analysis models.

The pipeline performs three sequential steps:

1. **Metadata extraction and semantic freezing** — parse OME-XML embedded in the TIFF to capture channel (marker) names, pixel sizes, and image dimensions.
2. **Conversion to chunked, random-access storage (Zarr)** — re-encode the OME-TIFF as a Zarr directory with configurable chunk sizes, enabling fast random patch reads without loading the entire image into memory.
3. **Per-channel robust intensity normalization** — clip, transform (arcsinh), and optionally min-max scale each channel independently so that downstream models receive consistent, well-conditioned input.

All scripts are designed for **HPC / SLURM** environments and ship with paired `.sh` job submission templates.

---

## 2. Dependencies

### 2.1 Create the Conda Environment

```
conda create -n spatial_preprocess python=3.10
conda activate spatial_preprocess
conda install -c conda-forge numpy zarr numcodecs tifffile lxml -y
```

### 2.2 Package Summary

| Package | Role |
|---------|------|
| `numpy` | Array computation and percentile estimation |
| `zarr` | Chunked array storage (read & write) |
| `numcodecs` | Blosc compression codec for Zarr |
| `tifffile` | OME-TIFF reading and Zarr-backed lazy access |
| `lxml` | OME-XML parsing |

## 3. Repository Structure

```
.
├── README.md
├── ome2meta.py        # Step 1: OME metadata extraction
├── ome2meta.sh        # Step 1: SLURM job script
├── ome2zarr.py        # Step 2: OME-TIFF → Zarr conversion
├── ome2zarr.sh        # Step 2: SLURM job script
├── raw2norm.py        # Step 3: Per-channel normalization
└── raw2norm.sh        # Step 3: SLURM job script
```

## 4. Pipeline Overview

The diagram below illustrates the data flow:

```
┌───────────────────┐
│  sample.ome.tiff  │   (raw CODEX acquisition, multi-channel OME-TIFF)
└────────┬──────────┘
         │
         ▼  Step 1: ome2meta.py
┌───────────────────────────────────┐
│  sample.channels.tsv              │   channel_index → marker_name mapping
│  sample.codex_meta.json           │   pixel size, axes, dimensions, dtype
└────────┬──────────────────────────┘
         │
         ▼  Step 2: ome2zarr.py
┌───────────────────────────────────┐
│  sample.codex_raw.zarr/           │   chunked Zarr (e.g. chunks=1,512,512)
│    └── 0  (dataset)               │   axes: CYX, dtype preserved from TIFF
└────────┬──────────────────────────┘
         │
         ▼  Step 3: raw2norm.py
┌───────────────────────────────────┐
│  sample.codex_norm.zarr/          │   normalized Zarr (float32)
│    └── 0  (dataset)               │   values in [0, 1] if --minmax enabled
└───────────────────────────────────┘
```

---

## 5. Step-by-Step Usage

### Step 1: OME Metadata Extraction (`ome2meta.py`)

Extracts channel names and image metadata from the OME-XML header embedded in the TIFF file. This step produces two files that are referenced by subsequent steps and serve as a frozen record of the acquisition parameters.

#### What it does

- Parses the OME-XML namespace to read `Channel/@Name` for each marker.
- Extracts `PhysicalSizeX`, `PhysicalSizeY` and their units (typically µm).
- Records image dimensions (`SizeX`, `SizeY`, `SizeC`, `SizeZ`, `SizeT`).
- Infers axes order and shape from `tifffile.series`.
- Falls back to generic channel names (`CH0`, `CH1`, …) when OME-XML is absent.

#### Usage

```bash
python ome2meta.py /path/to/sample.ome.tiff --outdir /path/to/meta
```

#### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `ome_tiff` | Yes | — | Path to the input `.ome.tiff` file |
| `--outdir` | No | Same directory as input file | Output directory for `.channels.tsv` and `.codex_meta.json` |

#### Outputs

**`<sample>.channels.tsv`** — tab-separated file mapping channel index to marker name:

```
channel_index	marker_name
0	DAPI
1	CD3
2	CD4
3	CD8
...
```

**`<sample>.codex_meta.json`** — structured metadata in JSON format:

```json
{
  "input_file": "/absolute/path/to/sample.ome.tiff",
  "axes_order": "CYX",
  "dtype": "uint16",
  "image_size": [14000, 10000],
  "n_channels": 48,
  "axes_dims": { "C": 48, "Y": 14000, "X": 10000 },
  "pixel_size_um": [0.377, 0.377],
  "pixel_size_unit_note": {
    "x_unit": "µm",
    "y_unit": "µm",
    "raw_physical_size_x": 0.377,
    "raw_physical_size_y": 0.377
  },
  "ome_sizex": 10000,
  "ome_sizey": 14000,
  "ome_sizec": 48
}
```

#### SLURM Submission

Edit `ome2meta.sh` to set your paths, then submit:

```bash
# Edit these variables in ome2meta.sh:
#   OME_TIFF, OUTDIR, CONDA_ENV, SCRIPT

sbatch ome2meta.sh
```

| Resource | Default |
|----------|---------|
| CPUs | 4 |
| Memory | 32 GB |
| Time limit | 1 hour |

---

### Step 2: OME-TIFF to Zarr Conversion (`ome2zarr.py`)

Converts the raw OME-TIFF into a Zarr directory store with configurable chunking and Blosc (zstd) compression. The conversion is performed **chunk-by-chunk** without loading the full image into RAM, making it safe for large (multi-GB) images on memory-constrained nodes.

#### What it does

- Opens the TIFF via `tifffile.series[i].aszarr()` for lazy, zero-copy access.
- Handles both `zarr.Array` and `zarr.Group` returns from `aszarr()` transparently.
- Creates a destination Zarr with the specified chunk shape and zstd compression.
- Copies data one chunk at a time to keep peak memory proportional to a single chunk.
- Stores full provenance (input path, axes, shape, dtype, compression settings) as Zarr root attributes.
- Optionally embeds the Step 1 metadata JSON into Zarr attributes for self-contained provenance.

#### Usage

```bash
python ome2zarr.py /path/to/sample.ome.tiff \
    --outdir /path/to/zarr \
    --chunks 1,512,512 \
    --clevel 3 \
    --meta_json /path/to/meta/sample.codex_meta.json \
    --overwrite
```

#### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `ome_tiff` | Yes | — | Path to input `.ome.tiff` |
| `--outdir` | No | Same directory as input | Output directory for the Zarr store |
| `--name` | No | Derived from filename | Custom output name prefix |
| `--series` | No | `0` | TIFF series index to convert |
| `--chunks` | No | `1,512,512` | Comma-separated chunk shape (must match data ndim) |
| `--clevel` | No | `3` | zstd compression level (1=fast, 9=small) |
| `--overwrite` | No | `False` | Overwrite existing Zarr directory |
| `--meta_json` | No | None | Path to `codex_meta.json` to embed in Zarr attrs |
| `--dataset_name` | No | `"0"` | Dataset key inside the Zarr group |

#### Choosing Chunk Shape

The chunk shape determines the smallest unit of I/O. For typical CODEX data with axes `CYX`:

| Scenario | Recommended `--chunks` | Rationale |
|----------|----------------------|-----------|
| Patch-based training (512×512) | `1,512,512` | One channel, one spatial patch per read |
| Patch-based training (256×256) | `1,256,256` | Smaller patches, more chunks |
| Whole-channel reads | `1,Y,X` | Minimal chunking along spatial dims |

> **Tip**: chunk sizes that evenly divide the spatial dimensions avoid partial chunks on boundaries and give more uniform I/O performance.

#### Output

```
sample.codex_raw.zarr/
├── .zgroup
├── .zattrs          ← provenance metadata (input path, axes, shape, dtype, etc.)
└── 0/               ← dataset array
    ├── .zarray
    └── <chunk files>
```

#### SLURM Submission

```bash
# Edit these variables in ome2zarr.sh:
#   OME_TIFF, OUTDIR, CONDA_ENV, SCRIPT, META_JSON, CHUNKS, CLEVEL

sbatch ome2zarr.sh
```

| Resource | Default |
|----------|---------|
| CPUs | 4 |
| Memory | 32 GB |
| Time limit | 2 hours |

---

### Step 3: Per-Channel Robust Normalization (`raw2norm.py`)

Applies per-channel intensity normalization to produce clean, model-ready tensors. Each channel is processed independently so that exposure differences and auto-fluorescence backgrounds do not leak between markers.

#### What it does

The normalization pipeline for each channel follows this sequence:

1. **Percentile clipping** — estimate the `p_low` and `p_high` percentiles by random pixel sampling across the full image, then clip all values to `[lo, hi]`.
2. **Background subtraction** — shift the clipped values by subtracting `lo`, so background intensity is near zero.
3. **Intensity transform** — apply `arcsinh(x / c)` (default), `log1p(x)`, or no transform. The arcsinh transform compresses high-dynamic-range fluorescence data while preserving relative ordering.
4. **Min-max scaling** (optional) — rescale each channel to `[0, 1]` based on the transformed range.

#### Why random sampling for percentiles?

Full percentile computation over tens of millions of pixels per channel is expensive. The script samples up to `--max_samples_per_channel` random `(y, x)` coordinates (default: 2M), which is more than sufficient for robust clipping bounds. Sampling is deterministic via `--seed`.

#### Usage

```bash
python raw2norm.py \
    --in_zarr  /path/to/zarr/sample.codex_raw.zarr \
    --out_zarr /path/to/zarr/sample.codex_norm.zarr \
    --p_low 1 --p_high 99 \
    --transform asinh --c 5 \
    --minmax \
    --max_samples_per_channel 2000000 \
    --out_dtype float32 \
    --clevel 3 \
    --overwrite
```

#### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--in_zarr` | Yes | — | Input raw Zarr path (from Step 2) |
| `--out_zarr` | Yes | — | Output normalized Zarr path |
| `--dataset` | No | `"0"` | Dataset name inside the Zarr group |
| `--overwrite` | No | `False` | Overwrite output Zarr if it exists |
| `--axes_order` | No | Read from attrs | Axes string (e.g. `"CYX"`). Auto-detected if stored by Step 2 |
| `--p_low` | No | `1.0` | Lower clipping percentile |
| `--p_high` | No | `99.0` | Upper clipping percentile |
| `--max_samples_per_channel` | No | `2,000,000` | Max random pixel samples per channel |
| `--seed` | No | `0` | RNG seed for reproducible sampling |
| `--transform` | No | `asinh` | Intensity transform: `asinh`, `log1p`, or `none` |
| `--c` | No | `5.0` | Scale factor for `arcsinh(x / c)` |
| `--minmax` | No | `False` | Apply per-channel min-max scaling to [0, 1] |
| `--out_dtype` | No | `float32` | Output dtype (`float32` or `float16`) |
| `--clevel` | No | `3` | Blosc zstd compression level |
| `--shuffle` | No | `bitshuffle` | Blosc shuffle mode: `bitshuffle`, `shuffle`, `noshuffle` |

#### Transform Options

| Transform | Formula | Best for |
|-----------|---------|----------|
| `asinh` (default) | `arcsinh(x / c)` | High dynamic range fluorescence; behaves like log for large values, linear near zero |
| `log1p` | `log(1 + x)` | General log compression; less tunable than asinh |
| `none` | identity | When no compression is desired (e.g., already pre-processed data) |

#### Output

```
sample.codex_norm.zarr/
├── .zgroup
├── .zattrs          ← inherited attrs from raw Zarr + normalization recipe + percentile values
└── 0/
    ├── .zarray
    └── <chunk files>
```

The Zarr root attributes include a `norm_recipe` block and `norm_percentiles` block for full reproducibility:

```json
{
  "norm_recipe": {
    "p_low": 1.0,
    "p_high": 99.0,
    "transform": "asinh",
    "c": 5.0,
    "minmax": true,
    "sampling": {
      "max_samples_per_channel": 2000000,
      "seed": 0
    }
  },
  "norm_percentiles": {
    "lo": [100.0, 85.0, ...],
    "hi": [4500.0, 3200.0, ...]
  }
}
```

#### SLURM Submission

```bash
# Edit these variables in raw2norm.sh:
#   IN_ZARR, OUT_ZARR, CONDA_ENV, SCRIPT
#   P_LOW, P_HIGH, TRANSFORM, ASINH_C, MAX_SAMPLES, OUT_DTYPE, CLEVEL

sbatch raw2norm.sh
```

| Resource | Default |
|----------|---------|
| CPUs | 4 |
| Memory | 32 GB |
| Time limit | 2 hours |

> **Performance tip**: the SLURM template uses `--clevel 1` and `--max_samples_per_channel 500000` for faster I/O and quicker percentile estimation. Increase these for production runs where compression ratio or estimation accuracy matters.

---

## 6. End-to-End Example

Below is a complete walkthrough for a sample named `tissue_A`:

```bash
# --- Environment ---
conda activate spatial_preprocess

SAMPLE="tissue_A"
OME_TIFF="/data/codex/${SAMPLE}.ome.tiff"
META_DIR="/data/codex/meta"
ZARR_DIR="/data/codex/zarr"

# --- Step 1: Extract metadata ---
python ome2meta.py "${OME_TIFF}" --outdir "${META_DIR}"
# → meta/tissue_A.channels.tsv
# → meta/tissue_A.codex_meta.json

# --- Step 2: Convert to Zarr ---
python ome2zarr.py "${OME_TIFF}" \
    --outdir "${ZARR_DIR}" \
    --chunks 1,512,512 \
    --clevel 3 \
    --meta_json "${META_DIR}/${SAMPLE}.codex_meta.json" \
    --overwrite
# → zarr/tissue_A.codex_raw.zarr/

# --- Step 3: Normalize ---
python raw2norm.py \
    --in_zarr  "${ZARR_DIR}/${SAMPLE}.codex_raw.zarr" \
    --out_zarr "${ZARR_DIR}/${SAMPLE}.codex_norm.zarr" \
    --p_low 1 --p_high 99 \
    --transform asinh --c 5 \
    --minmax \
    --overwrite
# → zarr/tissue_A.codex_norm.zarr/
```

