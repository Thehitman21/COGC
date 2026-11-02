# COGC

Official PyTorch implementation of **Controllable Optimizable Gamma Correction as Efficient Image Luminance Adapter**, accepted at **BMVC 2026**.

> **Controllable Optimizable Gamma Correction as Efficient Image Luminance Adapter**  
> Yimu Yang, Jiaming Chen, Zhenzhong Feng, and Zhenyuan Yang  
> British Machine Vision Conference (BMVC), 2026

[[Paper]](paper/COGC.pdf)

## Overview

COGC is a controllable and ultra-lightweight image-luminance adapter with only **0.13K parameters**. It uses a coordinate-based MLP to generate an optimizable spatial gamma map and iteratively adjusts image luminance without dataset-level network training.

The released inference code supports:

- low-light image enhancement;
- over-exposure correction;
- paper configurations for paired and unpaired benchmarks;
- optional BM3D post-processing;
- custom gamma ranges, residual settings, optimization direction, and random seed.

## Installation

COGC requires Python 3.9 or newer. We recommend using a virtual environment.

```bash
git clone https://github.com/Thehitman21/COGC.git
cd COGC

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell

pip install -r requirements.txt
```

## Quick Start

Enhance one image with automatic under/over-exposure detection:

```bash
python cogc_enhance.py path/to/input.png path/to/output.png --profile auto
```

Enhance every supported image directly inside a directory:

```bash
python cogc_enhance.py path/to/input_directory path/to/output_directory --profile unpaired
```

Reproduce the LOLv2 configuration reported in the paper:

```bash
python cogc_enhance.py path/to/low_light_images path/to/results --profile lolv2
```

Correct over-exposed LOM images:

```bash
python cogc_enhance.py path/to/over_exposed_images path/to/results --profile lom-over
```

## Paper Profiles

The following profiles implement the configurations reported in Table 1 of the paper.

| Profile | Scenario | Gamma range | Residual | BM3D |
| --- | --- | ---: | :---: | :---: |
| `lolv2` | LOLv2-Real | [0.3, 1.0] | On | On |
| `lsrw-huawei` | LSRW-Huawei | [0.3, 1.0] | On | On |
| `lsrw-nikon` | LSRW-Nikon | [0.3, 1.0] | Off | On |
| `lom-under-severe` | LOM under-exposure: Bike, Shrub, Sofa, Chair | [0.15, 1.0] | On | On |
| `lom-under-buu` | LOM under-exposure: Buu | [0.3, 1.0] | On | On |
| `lom-over` | LOM over-exposure | [3.0, 3.5] | Off | On |
| `unpaired` | DICM, VV, MEF, NPE, LIME | [0.3, 1.0] | Off | Off |
| `darkface` | DarkFace detection | [0.3, 1.0] | On | Off |
| `mfnet` | MFNet nighttime segmentation | [0.7, 1.0] | Off | Off |
| `auto` | General use with automatic exposure direction | [0.3, 1.0] | Automatic | Off |

## Advanced Options

The default optimization uses 25 iterations, a learning rate of 0.01, and the released initialization seed of 6272. All main controls can be overridden from the command line:

```bash
python cogc_enhance.py input.png output.png \
  --profile auto \
  --iterations 25 \
  --learning-rate 0.01 \
  --seed 6272 \
  --gamma-min 0.3 \
  --gamma-max 1.0 \
  --residual auto \
  --bm3d off \
  --mode brighten
```

Run the following command for the complete option list:

```bash
python cogc_enhance.py --help
```

## Reproducibility Notes

- The paper profiles fix task-level hyperparameters before test-time inference.
- BM3D uses `sigma = 20` by default and can be changed with `--bm3d-sigma`.
- `auto` is intended for convenient general use; benchmark reproduction should use the corresponding named profile.
- The scripts for benchmark metrics and downstream tasks are research utilities and may require their original datasets, models, and additional dependencies.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{Yang2026COGC,
  title        = {Controllable Optimizable Gamma Correction as Efficient Image Luminance Adapter},
  author       = {Yang, Yimu and Chen, Jiaming and Feng, Zhenzhong and Yang, Zhenyuan},
  booktitle    = {British Machine Vision Conference},
  year         = {2026},
  organization = {British Machine Vision Association}
}
```

## Contact

For questions about the paper or code, please open an issue in this repository.
