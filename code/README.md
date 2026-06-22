# HSFM: Hard-Set-Guided Feature-Space Meta-Learning

Official PyTorch implementation of **HSFM: Hard-Set-Guided Feature-Space Meta-Learning for Robust Classification under Spurious Correlations** (ECCV 2026).

Paper: [arXiv:2603.29313](https://arxiv.org/abs/2603.29313)

## Overview

HSFM improves robustness under spurious correlations by optimizing a class-balanced support set of feature embeddings while keeping a pretrained backbone frozen. A linear head is adapted on the support set in an inner loop; support embeddings are updated in an outer loop to reduce loss on hard validation examples (top-K highest loss per class).

## Setup

```bash
pip install -r requirements.txt
```

## Data and ERM checkpoints

HSFM expects:

1. A dataset-specific ERM-trained ResNet-50 checkpoint (backbone + linear head).
2. Dataset files for one of: Waterbirds, CelebA, Dominoes, or MetaShift.

Place ERM checkpoints under `checkpoints/` (or pass `--model_checkpoint`). Dataset paths can be overridden via CLI flags; see `python train_hsfm.py --help`.

| Dataset    | Default data path                         | Default checkpoint              |
|------------|-------------------------------------------|---------------------------------|
| Waterbirds | `../DaC/waterbird_complete95_forest2water2` | `checkpoints/waterbirds_erm.pt` |
| CelebA     | `../DaC/celeba`                           | `checkpoints/celeba_erm.pt`     |
| Dominoes   | `Dominoes_SP90 2`                         | `checkpoints/dominoes_erm.pt`   |
| MetaShift  | `metashifts/MetaDatasetCatDog`            | `checkpoints/metashift_erm.pt`  |

## Training

Run HSFM with paper defaults (Table 11 in the appendix):

```bash
# Waterbirds
bash scripts/run_waterbirds.sh

# CelebA
bash scripts/run_celeba.sh

# Dominoes
bash scripts/run_dominoes.sh

# MetaShift
bash scripts/run_metashift.sh
```

Or directly:

```bash
python train_hsfm.py --dataset waterbirds --model_checkpoint checkpoints/waterbirds_erm.pt
```

### Main hyperparameters

| Flag | Paper symbol | Description |
|------|--------------|-------------|
| `--support_size_per_class` | \|S\| per class | Class-balanced support set size |
| `--T` | T | Inner-loop and head update steps |
| `--inner_lr` | α | Inner-loop learning rate |
| `--outer_lr` | η | Meta learning rate |
| `--KH` | K_H | Support embedding updates per epoch |
| `--K_hard` | K_hard | Hard validation samples per class |
| `--epochs` | epochs | Number of training epochs |

Defaults are set automatically per dataset from the paper. Validation worst-group accuracy is logged each epoch; test metrics are reported once for the best validation model.

To save optimized support embeddings for unCLIP visualization (Sec. 5.3):

```bash
python train_hsfm.py --dataset waterbirds --save_repr_checkpoints
```

## Project structure

```
HSFM/
├── train_hsfm.py      # Main HSFM training script
├── Waterbirds.py      # Waterbirds dataset loader
├── CelebA.py          # CelebA dataset loader
├── dominoes.py        # Dominoes dataset loader
├── Metashift.py       # MetaShift dataset loader
├── scripts/           # Example launch scripts
├── requirements.txt
└── README.md
```

## Citation

```bibtex
@inproceedings{parast2026hsfm,
  title={HSFM: Hard-Set-Guided Feature-Space Meta-Learning for Robust Classification under Spurious Correlations},
  author={Parast, Aryan Yazdan and Islam, Khawar and Won, Soyoun and Azam, Basim and Akhtar, Naveed},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## License

This code is released for research purposes. See the paper for dataset licenses and usage terms.
