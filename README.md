# CLIP2FL-BKD

**Long-tail federated learning을 위한 Balanced Knowledge Distillation. CLIP2FL 위에 구현했다.**

CLIP2FL은 고정된 CLIP teacher의 지식을 client 모델로 distillation한다. 이 저장소는 그 distillation을
**BKD**로 교체한다. BKD는 teacher의 출력 분포를 class 빈도로 재가중한 뒤 distillation하므로,
head class가 tail class를 덮어쓰지 않는다.

> 게재: Jun Jeon, Minu Baek, Sangkeum Lee\*, *"Balanced Knowledge Distillation (BKD) for
> Long-Tail Federated Learning Based on CLIP2FL"*, **KICS 2026 동계학술대회**.
> 한밭대학교. (\*교신저자)

BKD 기법의 설계와 구현, STL-10 long-tail 파이프라인, 실험 추적 계층은 모두 저자 본인이 작성했으며,
아래에 보고된 실험도 직접 수행했다.

---

## 문제

Federated learning은 client의 데이터를 공유하지 않고 업데이트만 모은다. 현실적인 배포 환경에서는
다음 두 성질이 동시에 나타나 학습을 무너뜨린다.

- **Non-IID.** client마다 보유한 class 분포가 다르다. 그래서 local 모델은 자기가 가진 class에
  overfitting되고, 각 client의 업데이트는 서로 다른 방향을 가리킨다.
- **Long-tail.** head class의 샘플 수가 tail class보다 훨씬 많다. 그래서 aggregation이 head class의
  gradient에 지배되고 tail class는 계속 underfitting 상태로 남는다.

CLIP2FL은 CLIP을 teacher로 써서 이 문제에 접근한다. 각 client가 CLIP의 image-text similarity logit을
local 모델로 distillation하므로, 데이터가 부족한 client에 CLIP의 표현이 주입된다.

**이 저장소가 겨냥한 한계:** CLIP2FL은 teacher 분포를 *그대로* 사용한다. CLIP의 logit은 해당 client의
class 분포에 대해 balanced하지 않기 때문에, distillation이 head class의 편향을 교정하기는커녕 student로
그대로 전달할 수 있다.

---

## 이 저장소가 추가한 것

CLIP2FL 파이프라인은 그대로 두고 client 쪽 distillation 항만 바꿨다. 두 설정을 동일 조건에서 비교할 수
있도록 진입점을 둘 다 남겨두었다.

| | `main.py` (Base) | `main_bkd.py` (BKD) |
|---|---|---|
| Client distillation loss | `KDLoss` — raw CLIP softmax에 대한 KL | `BKD2Loss` — **재가중된** CLIP softmax에 대한 KL |
| Teacher 재가중 | 없음 | client의 class 개수로 계산한 class별 `λ` |
| 데이터셋 | CIFAR-10 / CIFAR-100 | CIFAR-10 / CIFAR-100 / **STL-10** |
| Checkpoint / resume | 없음 | 10 round마다 저장, `--resume` |
| 실행 산출물 | 로그 파일 | 로그 + config / results / metadata JSON |

그 밖에 이 저장소에서 변경한 것들이다.

- **STL-10 long-tail 지원.** upstream에는 없다. long-tail split(`Dataset/long_tailed_stl10.py`),
  데이터셋 무결성 검사와 조건부 다운로드를 추가했고, ResNet-8 backbone에 `AdaptiveAvgPool2d(1)`을 넣어
  같은 student가 32×32(CIFAR)와 96×96(STL-10) 입력을 모두 받도록 했다.
- **재현성 계층.** 모든 실행에 `experiment_id`를 부여하고, 설정과 round별 정확도, wall-clock 메타데이터를
  각각 별도 JSON으로 기록한다.
- 이번 연구에 쓰지 않은 데이터셋 경로(FEMNIST, SVHN, Caltech101)를 제거했다.

---

## 방법

### Class weight

local training 시작 시점에 client *k*는 자신이 보유한 class별 샘플 수로부터 class마다 weight 하나를
계산한다.

```python
# main_bkd.py — class Local.__init__
class_counts = torch.tensor(self.class_compose, dtype=torch.float32, device=self.device)
gamma = 0.5
lambda_c = 1.0 / class_counts.clamp(min=1.0) ** gamma
lambda_c = lambda_c / lambda_c.mean()          # 평균 weight가 1이 되도록 정규화
```

이 가중 방식의 성질 두 가지다.

- 지수 `γ = 0.5`이므로 교정 강도는 완전한 역빈도가 아니라 **역빈도의 제곱근**이다.
- `λ`는 전역 분포가 아니라 **해당 client의 local class 분포**로부터 계산된다.

### 재가중 distillation

`BKD2Loss`는 KL 항을 계산하기 전에 teacher 분포에 `λ`를 적용하고 다시 정규화한다. 그래서 결과가
여전히 확률 분포로 유지된다.

```python
pred_t = softmax(clip_logits / T)
pred_t = pred_t * lambda_c
pred_t = pred_t / pred_t.sum(1, keepdim=True)   # 재정규화
kd     = KL(log_softmax(student / T), pred_t) * T * T
```

client의 목적 함수는 형태가 그대로다.

```
L = L_CE + α · L_BKD
```

loss 함수 자체는 건드리지 않고 teacher 신호만 바꾸는 설계다. federated 환경에서는 각 client가 이미 서로
다른 분포 위에서 최적화하고 있기 때문에, loss를 직접 재가중하는 방식(예: Focal Loss)은 학습 불안정을
유발할 수 있다. 이 설계는 그 위험을 피한다.

### 보고된 결과의 실험 설정

| | |
|---|---|
| Teacher | CLIP ViT-B/32 (frozen) |
| Student | ResNet-8, 512-d feature |
| Client | 전체 20개, round당 8개 샘플링 |
| Round | 200, round당 local epoch 10, batch size 32 |
| Non-IID split | Dirichlet, `non_iid_alpha = 0.5` |
| Server | class당 synthetic federated feature 100개, matching 100 step, classifier 재학습 300 step |
| Distillation | `T = 3.0`, `α = 1.0` |
| Seed | 7 |

---

## 결과

Top-1 정확도(%). `IF`는 imbalance factor이며 클수록 분포가 치우쳐 있다.
Base는 `main.py`, BKD는 `main_bkd.py`이고 위 설정으로 실행했다.

| 데이터셋 | IF | Base | BKD | Δ |
|---|---:|---:|---:|---:|
| CIFAR-10 | 10 | 73.89 | 73.06 | −0.83 |
| CIFAR-10 | 50 | 76.38 | 76.04 | −0.34 |
| CIFAR-10 | 100 | 83.61 | 81.19 | −2.42 |
| STL-10 | 10 | 42.44 | **43.25** | **+0.81** |
| STL-10 | 50 | 43.63 | **45.31** | **+1.68** |
| STL-10 | 100 | 52.34 | **53.89** | **+1.55** |

**BKD는 STL-10에서 모든 imbalance 수준에서 성능을 올렸고, CIFAR-10에서는 모든 수준에서 떨어뜨렸다.**
결과가 갈렸고, 갈린 지점이 이 실험에서 볼 것이다.

---

## 분석

문제는 재가중 자체가 아니라 teacher에 있다.

CLIP은 고해상도 자연 이미지로 사전학습되었다. STL-10 이미지는 96×96이라 그 도메인에 가깝게 남아 있고,
따라서 CLIP의 logit이 유의미한 신호가 되어 재가중이 더 balanced한 target을 만들어낸다. CIFAR-10 이미지는
32×32로 CLIP의 사전학습 분포에서 크게 벗어나 있어서, logit 자체가 이미 이 과제에 대해 약한 신호다.
재가중으로 얻을 수 있는 것에 한계가 있다.

**즉 BKD의 효과는 teacher와 student의 도메인 일치도에 의존한다.**

---

## 설치

```bash
conda create -n clip2fl python=3.7.9 -y
conda activate clip2fl

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt      # 공식 저장소의 CLIP 포함
```

CIFAR-10, CIFAR-100, STL-10은 최초 실행 시 `data/`에 자동으로 다운로드된다.
경로는 `--path_cifar10`, `--path_cifar100`, `--path_stl10`으로 바꿀 수 있다.

---

## 사용법

Base/BKD 쌍을 재현하려면 두 진입점을 동일한 인자로 실행한다.
`IF = 1 / imb_factor`이므로 `IF = 10, 50, 100`은 각각 `--imb_factor 0.1, 0.02, 0.01`에 해당한다.

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

# Base — 동일한 설정
python main.py --dataset stl10 --num_classes 10 --imb_factor 0.01 ...
```

데이터셋은 `--dataset cifar10|cifar100|stl10`으로 바꾸고 `--num_classes`를 함께 맞춘다
(CIFAR-100은 100).

### 주요 인자

| 인자 | 의미 | 기본값 |
|---|---|---|
| `--imb_factor` | long-tail 강도. 작을수록 더 불균형 | `0.01` |
| `--non_iid_alpha` | Dirichlet 집중도. 작을수록 더 이질적 | `0.5` |
| `--alpha` | distillation 항의 weight | `1.0` |
| `--T` | distillation temperature | `3.0` |
| `--contrast_alpha` | server 쪽 prototype contrastive loss의 weight | `0.001` |
| `--match_epoch` | round당 federated feature 최적화 step 수 | `100` |
| `--crt_epoch` | round당 classifier 재학습 step 수 | `300` |
| `--resume` | checkpoint에서 이어서 학습 | — |

---

## 실험 추적

각 실행은 `{timestamp}_main_clip2fl_bkd_{alpha}_{contrast_alpha}_IF{imb_factor}`로 식별되며
`results/{dataset}/main_clip2fl_bkd/`에 기록된다.

| 파일 | 내용 |
|---|---|
| `{id}.log` | 전체 학습 로그 |
| `{id}_config.json` | 실행에 사용한 모든 인자 |
| `{id}_results.json` | 최종/최고 정확도, round별 정확도 리스트, 총 실행 시간 |
| `{id}_metadata.json` | 실행 메타데이터 |
| `checkpoints/{id}_checkpoint_round_{N}.pth` | 10 round마다 저장 |
| `checkpoints/{id}_final_model.pth` | 최종 모델 |

중단된 실행은 `--resume <checkpoint path>`로 이어서 진행하며, round 인덱스와 정확도 이력,
synthetic feature 및 label이 복원된다.

---

## 한계

- **BKD는 CIFAR-10에서 도움이 되지 않는다.** 이 기법은 teacher의 도메인이 student의 데이터와 가까운
  경우에 한해서만 이득이 검증되었다.
- **교정이 teacher를 오히려 망가뜨릴 수 있다.** `λ`는 해당 샘플에 대한 CLIP의 예측이 맞았는지와 무관하게
  tail class의 확률 질량을 키운다. 그래서 확신에 차서 틀린 teacher 출력도 함께 증폭된다.
- **STL-10에서 student와 teacher의 입력 파이프라인이 일치하지 않는다.** local training은 32×32 CIFAR
  입력용으로 작성된 `RandomCrop(32, padding=4)`를 적용한다. 그래서 STL-10에서는 student가 96×96 이미지의
  32×32 crop으로 학습되는 반면 CLIP teacher는 전체 이미지를 받고, 평가도 96×96 전체 이미지로 이뤄진다.
  adaptive pooling 덕분에 backbone은 이 상황을 견디지만, 의도하지 않은 train/test 및 student/teacher
  해상도 불일치다. 위의 STL-10 수치는 이 점을 감안하고 읽어야 한다.

---

## 저장소 구조

```
.
├── main.py                       # Base: 표준 KD를 쓰는 CLIP2FL
├── main_bkd.py                   # BKD: 재가중 teacher distillation (+ checkpointing)
├── options.py                    # 모든 하이퍼파라미터
├── losses.py                     # server 쪽 prototype contrastive loss
├── Dataset/
│   ├── long_tailed_cifar10.py    # CIFAR long-tail split (upstream)
│   ├── long_tailed_stl10.py      # STL-10 long-tail split (여기서 추가)
│   ├── sample_dirichlet.py       # Non-IID client 분할
│   ├── param_aug.py              # Differentiable Siamese augmentation
│   └── Gradient_matching_loss.py
├── Model/
│   ├── Resnet8.py / Resnet8_256.py   # ResNet-8 student (32/96 px 대응 adaptive pooling)
│   ├── ResNet50.py
│   └── nets.py
├── requirements.txt
└── README_base.md                # 원본 CLIP2FL README, 수정 없이 보존
```

---

## 인용

```bibtex
@inproceedings{jeon2026bkd,
  title     = {Balanced Knowledge Distillation (BKD) for Long-Tail Federated Learning Based on CLIP2FL},
  author    = {Jeon, Jun and Baek, Minu and Lee, Sangkeum},
  booktitle = {KICS Winter Conference},
  year      = {2026}
}
```

---

## 참고 및 출처

이 연구는 **CLIP2FL**의 공식 구현
(Shi et al., *CLIP-guided Federated Learning on Heterogeneous and Long-Tailed Data*, AAAI 2024)
위에 구축했으며, CLIP2FL 자체는 [CReFF](https://github.com/shangxinyi/CReFF-FL)를 기반으로 한다.
원본 프로젝트의 README는 [`README_base.md`](README_base.md)에 보존해두었다.
CLIP encoder는 [OpenAI CLIP](https://github.com/openai/CLIP) 공식 릴리스를 사용했다.

`Model/`, `Dataset/sample_dirichlet.py`, `Dataset/param_aug.py`,
`Dataset/Gradient_matching_loss.py`, `losses.py`의 모델·데이터셋 분할·augmentation 코드는 위 프로젝트에서
가져온 것이다. BKD loss, STL-10 long-tail 파이프라인, 실험 추적 계층은 이 저장소에서 작성했다.
