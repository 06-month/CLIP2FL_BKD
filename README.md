# CLIP2FL-BKD

**Balanced Knowledge Distillation for long-tail federated learning, built on CLIP2FL.**

CLIP2FL distills knowledge from a frozen CLIP teacher into client models. This repository
replaces that distillation with **BKD**, which re-weights the teacher's output distribution by
class frequency before distillation, so that tail classes are not drowned out by head classes.

> Published as: Jun Jeon, Minu Baek, Sangkeum Lee\*, *"Balanced Knowledge Distillation (BKD) for
> Long-Tail Federated Learning Based on CLIP2FL"*, **KICS Winter Conference 2026**.
> Hanbat National University. (\*corresponding author)

I designed and implemented BKD, built the STL-10 long-tail pipeline and the experiment-tracking
layer, and ran all experiments reported here.

---

## Problem

Federated learning aggregates client updates without sharing client data. Two properties of
realistic deployments break it at the same time:

- **Non-IID.** Each client holds a different class distribution, so local models overfit to
  the classes they happen to own and their updates point in conflicting directions.
- **Long-tail.** Head classes hold far more samples than tail classes, so aggregation is
  dominated by head-class gradients and tail classes stay underfit.

CLIP2FL addresses this by using CLIP as a teacher: each client distills CLIP's image-text
similarity logits into its local model, injecting CLIP's representation into clients that have
too little data of their own.

**The limitation this repository targets:** CLIP2FL consumes the teacher distribution *as is*.
CLIP's logits are not balanced with respect to the client's own class distribution, so
distillation can transfer head-class bias straight into the student instead of correcting it.

---

## What This Repository Adds

The repository keeps the CLIP2FL pipeline intact and changes the client-side distillation term.
Both entry points are kept so the two settings can be run under identical conditions.

| | `main.py` (Base) | `main_bkd.py` (BKD) |
|---|---|---|
| Client distillation loss | `KDLoss` — KL against raw CLIP softmax | `BKD2Loss` — KL against **re-weighted** CLIP softmax |
| Teacher re-weighting | none | per-class `λ`, computed from client class counts |
| Datasets | CIFAR-10 / CIFAR-100 | CIFAR-10 / CIFAR-100 / **STL-10** |
| Checkpoint / resume | no | every 10 rounds, `--resume` |
| Run artifacts | log file | log + config / results / metadata JSON |

Additional changes made in this repository:

- **STL-10 long-tail support**, which upstream does not provide: a long-tail split
  (`Dataset/long_tailed_stl10.py`), dataset integrity checks and conditional download, and
  `AdaptiveAvgPool2d(1)` in the ResNet-8 backbone so the same student accepts both
  32×32 (CIFAR) and 96×96 (STL-10) inputs.
- **Reproducibility layer**: every run gets an `experiment_id`, and its configuration,
  per-round accuracies, and wall-clock metadata are written as separate JSON files.
- Removal of unused dataset paths (FEMNIST, SVHN, Caltech101) that were not part of the study.

---

## Method

### Class weights

At the start of local training, client *k* computes one weight per class from the number of
samples it holds for that class:

```python
# main_bkd.py — class Local.__init__
class_counts = torch.tensor(self.class_compose, dtype=torch.float32, device=self.device)
gamma = 0.5
lambda_c = 1.0 / class_counts.clamp(min=1.0) ** gamma
lambda_c = lambda_c / lambda_c.mean()          # normalized so the mean weight is 1
```

Two properties of this weighting:

- The exponent `γ = 0.5` makes the correction a **square-root inverse frequency**, not a full
  inverse.
- `λ` is computed from the **client's local class distribution**, not a global one.

### Re-weighted distillation

`BKD2Loss` applies `λ` to the teacher distribution and renormalizes it before the KL term, so
the result is still a probability distribution:

```python
pred_t = softmax(clip_logits / T)
pred_t = pred_t * lambda_c
pred_t = pred_t / pred_t.sum(1, keepdim=True)   # renormalize
kd     = KL(log_softmax(student / T), pred_t) * T * T
```

The client objective is unchanged in form:

```
L = L_CE + α · L_BKD
```

The design keeps the loss function itself untouched and only reshapes the teacher signal. This
avoids the training instability that direct loss re-weighting (e.g. Focal Loss) can introduce
in a federated setting, where each client already optimizes on a different distribution.

### Setup used for the reported results

| | |
|---|---|
| Teacher | CLIP ViT-B/32 (frozen) |
| Student | ResNet-8, 512-d features |
| Clients | 20 total, 8 sampled per round |
| Rounds | 200, 10 local epochs each, batch size 32 |
| Non-IID split | Dirichlet, `non_iid_alpha = 0.5` |
| Server | 100 synthetic federated features per class; 100 matching steps; 300 classifier re-training steps |
| Distillation | `T = 3.0`, `α = 1.0` |
| Seed | 7 |

---

## Results

Top-1 accuracy (%). `IF` is the imbalance factor; higher means more skewed.
Base is `main.py`, BKD is `main_bkd.py`, run under the settings above.

| Dataset | IF | Base | BKD | Δ |
|---|---:|---:|---:|---:|
| CIFAR-10 | 10 | 73.89 | 73.06 | −0.83 |
| CIFAR-10 | 50 | 76.38 | 76.04 | −0.34 |
| CIFAR-10 | 100 | 83.61 | 81.19 | −2.42 |
| STL-10 | 10 | 42.44 | **43.25** | **+0.81** |
| STL-10 | 50 | 43.63 | **45.31** | **+1.68** |
| STL-10 | 100 | 52.34 | **53.89** | **+1.55** |

**BKD improves STL-10 at every imbalance level and degrades CIFAR-10 at every imbalance level.**
The result is split, and the split is the interesting part.

---

## Analysis

The failure is not in the re-weighting; it is in the teacher.

CLIP was pretrained on high-resolution natural images. STL-10 images are 96×96 and stay close to
that domain, so CLIP's logits are informative and re-weighting them yields a better-balanced
target. CIFAR-10 images are 32×32, far outside CLIP's pretraining distribution, so its logits are
already a weak signal for this task, which limits what re-weighting it can achieve.

**The effect of BKD therefore depends on teacher–student domain agreement.**

---

## Setup

```bash
conda create -n clip2fl python=3.7.9 -y
conda activate clip2fl

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt      # includes CLIP from the official repo
```

CIFAR-10, CIFAR-100 and STL-10 are downloaded automatically on first run into `data/`.
Paths can be overridden with `--path_cifar10`, `--path_cifar100`, `--path_stl10`.

---

## Usage

Run the two entry points with identical arguments to reproduce a Base/BKD pair.
`IF = 1 / imb_factor`, so `IF = 10, 50, 100` corresponds to `--imb_factor 0.1, 0.02, 0.01`.

```bash
# BKD — STL-10, IF = 100
python main_bkd.py \
  --dataset stl10 --num_classes 10 \
  --imb_factor 0.01 --non_iid_alpha 0.5 \
  --num_clients 20 --num_online_clients 8 --num_rounds 200 \
  --num_epochs_local_training 10 \
  --match_epoch 100 --crt_epoch 300 \
  --alpha 1.0 --T 3.0 --contrast_alpha 0.001 \
  --lr_local_training 0.1 --lr_feature 0.1 --lr_net 0.01 \
  --gpu 0 --result_save results

# Base — same configuration
python main.py --dataset stl10 --num_classes 10 --imb_factor 0.01 ...
```

Switch datasets with `--dataset cifar10|cifar100|stl10` and set `--num_classes` accordingly
(CIFAR-100 uses 100).

### Key arguments

| Argument | Meaning | Default |
|---|---|---|
| `--imb_factor` | Long-tail severity; smaller is more imbalanced | `0.01` |
| `--non_iid_alpha` | Dirichlet concentration; smaller is more heterogeneous | `0.5` |
| `--alpha` | Weight of the distillation term | `1.0` |
| `--T` | Distillation temperature | `3.0` |
| `--contrast_alpha` | Weight of the server-side prototype contrastive loss | `0.001` |
| `--match_epoch` | Federated-feature optimization steps per round | `100` |
| `--crt_epoch` | Classifier re-training steps per round | `300` |
| `--resume` | Resume from a checkpoint | — |

---

## Experiment Tracking

Each run is identified by `{timestamp}_main_clip2fl_bkd_{alpha}_{contrast_alpha}_IF{imb_factor}`
and writes to `results/{dataset}/main_clip2fl_bkd/`:

| File | Contents |
|---|---|
| `{id}.log` | Full training log |
| `{id}_config.json` | Every argument used for the run |
| `{id}_results.json` | Final / best accuracy, per-round accuracy list, total runtime |
| `{id}_metadata.json` | Run metadata |
| `checkpoints/{id}_checkpoint_round_{N}.pth` | Saved every 10 rounds |
| `checkpoints/{id}_final_model.pth` | Final model |

Interrupted runs resume with `--resume <checkpoint path>`, restoring the round index,
accuracy history, synthetic features and labels.

---

## Limitations

- **BKD does not help on CIFAR-10.** The method is only validated as beneficial where the
  teacher's domain is close to the student's data.
- **The teacher can be corrupted by the correction.** `λ` amplifies tail-class probability mass
  regardless of whether CLIP's prediction for that sample was correct, so a confidently wrong
  teacher output is amplified along with a correct one.
- **The STL-10 input pipeline is inconsistent between student and teacher.** Local training
  applies `RandomCrop(32, padding=4)`, which was written for 32×32 CIFAR inputs, so on STL-10 the
  student is trained on 32×32 crops of a 96×96 image while the CLIP teacher receives the full
  image and evaluation uses the full 96×96 image. The backbone tolerates this through adaptive
  pooling, but it is a train/test and student/teacher resolution mismatch that was not intended,
  and the STL-10 numbers above should be read with that in mind.

---

## Repository Structure

```
.
├── main.py                       # Base: CLIP2FL with standard KD
├── main_bkd.py                   # BKD: re-weighted teacher distillation (+ checkpointing)
├── options.py                    # All hyperparameters
├── losses.py                     # Server-side prototype contrastive loss
├── Dataset/
│   ├── long_tailed_cifar10.py    # CIFAR long-tail split (upstream)
│   ├── long_tailed_stl10.py      # STL-10 long-tail split (added here)
│   ├── sample_dirichlet.py       # Non-IID client partitioning
│   ├── param_aug.py              # Differentiable Siamese augmentation
│   └── Gradient_matching_loss.py
├── Model/
│   ├── Resnet8.py / Resnet8_256.py   # ResNet-8 student (adaptive pooling for 32/96 px)
│   ├── ResNet50.py
│   └── nets.py
├── requirements.txt
└── README_base.md                # Original CLIP2FL README, kept unmodified
```

---

## Citation

```bibtex
@inproceedings{jeon2026bkd,
  title     = {Balanced Knowledge Distillation (BKD) for Long-Tail Federated Learning Based on CLIP2FL},
  author    = {Jeon, Jun and Baek, Minu and Lee, Sangkeum},
  booktitle = {KICS Winter Conference},
  year      = {2026}
}
```

---

## Acknowledgments

This work builds on the official implementation of **CLIP2FL**
(Shi et al., *CLIP-guided Federated Learning on Heterogeneous and Long-Tailed Data*, AAAI 2024),
which is itself based on [CReFF](https://github.com/shangxinyi/CReFF-FL).
The original project README is preserved in [`README_base.md`](README_base.md).
The CLIP encoder is the official [OpenAI CLIP](https://github.com/openai/CLIP) release.

Model, dataset-partitioning and augmentation code under `Model/`, `Dataset/sample_dirichlet.py`,
`Dataset/param_aug.py`, `Dataset/Gradient_matching_loss.py` and `losses.py` originate from those
projects. The BKD loss, the STL-10 long-tail pipeline, and the experiment-tracking layer are
specific to this repository.
