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

COGC requires Python 3.9 or newer. Python 3.10 is recommended.

```bash
git clone https://github.com/Thehitman21/COGC.git
cd COGC
```

### Conda (Recommended)

Create and activate an isolated Conda environment:

```bash
conda create --name cogc python=3.10 -y
conda activate cogc
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU inference, install the PyTorch build matching your CUDA version from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`. COGC automatically uses CUDA when it is available and otherwise runs on the CPU.

### Python venv (Alternative)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell
python -m pip install --upgrade pip
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

## Datasets

COGC is evaluated on paired enhancement, unpaired enhancement, novel-view synthesis, face detection, and semantic-segmentation benchmarks. Download each dataset from its official project page or author-maintained repository:

| Dataset | Task in the paper | Official resource |
| --- | --- | --- |
| LOLv2-Real | Paired low-light image enhancement | [CVPR 2020 Semi-Low-Light / LOLv2](https://github.com/flyywh/CVPR-2020-Semi-Low-Light#3-dataset) |
| LSRW-Huawei | Paired low-light image enhancement | [R2RNet / LSRW](https://github.com/JianghaiSCU/R2RNet#dataset) |
| LSRW-Nikon | Paired low-light image enhancement | [R2RNet / LSRW](https://github.com/JianghaiSCU/R2RNet#dataset) |
| LOM | Low-light and over-exposure novel-view synthesis | [Aleth-NeRF / LOM](https://github.com/cuiziteng/Aleth-NeRF#-usage) |
| DICM | Unpaired low-light image enhancement | [BIMEF dataset collection](https://github.com/baidut/BIMEF#datasets) |
| LIME | Unpaired low-light image enhancement | [BIMEF dataset collection](https://github.com/baidut/BIMEF#datasets) |
| MEF | Unpaired low-light image enhancement | [BIMEF dataset collection](https://github.com/baidut/BIMEF#datasets) |
| NPE | Unpaired low-light image enhancement | [BIMEF dataset collection](https://github.com/baidut/BIMEF#datasets) |
| VV | Unpaired low-light image enhancement | [BIMEF dataset collection](https://github.com/baidut/BIMEF#datasets) |
| DarkFace | Low-light face detection | [DARK FACE official project page](https://flyywh.github.io/CVPRW2019LowLight/) |
| MFNet | Nighttime RGB-thermal semantic segmentation | [MFNet official project page](https://www.mi.t.u-tokyo.ac.jp/projects/mil_multispectral/) |

The datasets are not redistributed in this repository. Please follow the original authors' licenses and usage terms, preserve the official train/test splits, and cite the corresponding dataset papers.

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
