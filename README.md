## 1. Introduction

This repository provides a minimal, explicit, and reproducible preprocessing pipeline that standardizes codex data before modeling or biological assumptions are introduced.

The output of this repository is model-ready tensors that can be directly consumed by deep learning pipelines.

1. Metadata extraction and semantic freezing  
2. Conversion to chunked, random-access storage (Zarr)  
3. Per-channel robust intensity normalization  

## 2. Dependencies

```
conda create -n spatial_preprocess python=3.10
conda activate spatial_preprocess
conda install -c conda-forge numpy zarr numcodecs tifffile lxml -y
```
## 3. Pipeline Overview

### Step 1: OME Metadata Extraction

ome2meta.py

### Step 2: OME-TIFF to Zarr Conversion

ome2zarr.py

### Step 3: Per-Channel Robust Normalization

codex_step3_norm_zarr.py

