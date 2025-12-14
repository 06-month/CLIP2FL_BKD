# CLIP2FL_BKD
IoT 2차 프로젝트 논문 CLIP2FL에 BKD로직을 추가하여 재현한 레포지토리입니다.

## CLIP2FL 실행 가이드

### 1. 환경 세팅

1. Conda 가상환경 생성 및 활성화
   ```bash
   conda create -n clip2fl python=3.7.9 -y
   conda activate clip2fl
   ```

2. PyTorch 설치 (CUDA 11.0)
   ```bash
   pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 \
     -f https://download.pytorch.org/whl/torch_stable.html
   ```

3. 나머지 의존성 설치
   ```bash
   pip install -r requirements.txt
   ```
   *`requirements.txt`는 numpy, Pillow, tqdm, ftfy, regex, clip-by-openai 등을 포함*

### 2. 데이터 준비

#### CIFAR-10/100
- 데이터는 자동으로 다운로드됩니다 (`download=True` 옵션)
- 기본 경로: `data/CIFAR10/`, `data/CIFAR100/`
- `options.py`의 `--path_cifar10`, `--path_cifar100`로 경로 변경 가능

#### STL-10
- 데이터는 자동으로 다운로드됩니다
- 기본 경로: `data/STL10/`
- `options.py`의 `--path_stl10`로 경로 변경 가능

### 3. 실행 방법

#### 기본 실행 (main_bkd.py)

현재 메인 실행 파일은 `main_bkd.py`입니다. BKD (Balanced Knowledge Distillation) 방식을 사용합니다.

#### CIFAR-10 실행 예시

```bash
python main_bkd.py \
  --dataset cifar10 \
  --num_classes 10 \
  --num_clients 20 \
  --num_online_clients 8 \
  --num_rounds 200 \
  --num_epochs_local_training 10 \
  --match_epoch 100 \
  --crt_epoch 300 \
  --contrast_alpha 0.001 \
  --alpha 1.0 \
  --non_iid_alpha 0.5 \
  --imb_factor 0.01 \
  --lr_local_training 0.1 \
  --lr_feature 0.1 \
  --lr_net 0.01 \
  --gpu 0 \
  --result_save results
```

#### CIFAR-100 실행 예시

```bash
python main_bkd.py \
  --dataset cifar100 \
  --num_classes 100 \
  --num_clients 20 \
  --num_online_clients 8 \
  --num_rounds 200 \
  --num_epochs_local_training 10 \
  --match_epoch 100 \
  --crt_epoch 300 \
  --contrast_alpha 0.001 \
  --alpha 1.0 \
  --non_iid_alpha 0.5 \
  --imb_factor 0.01 \
  --lr_local_training 0.1 \
  --lr_feature 0.1 \
  --lr_net 0.01 \
  --gpu 0 \
  --result_save results
```

#### STL-10 실행 예시

```bash
python main_bkd.py \
  --dataset stl10 \
  --num_classes 10 \
  --num_clients 20 \
  --num_online_clients 8 \
  --num_rounds 200 \
  --num_epochs_local_training 10 \
  --match_epoch 100 \
  --crt_epoch 300 \
  --contrast_alpha 0.001 \
  --alpha 1.0 \
  --non_iid_alpha 0.5 \
  --imb_factor 0.01 \
  --lr_local_training 0.1 \
  --lr_feature 0.1 \
  --lr_net 0.01 \
  --gpu 0 \
  --result_save results
```

### 4. 주요 하이퍼파라미터

- `--dataset`: 데이터셋 선택 (`cifar10`, `cifar100`, `stl10`)
- `--num_classes`: 클래스 수 (CIFAR-10: 10, CIFAR-100: 100, STL-10: 10)
- `--num_clients`: 전체 클라이언트 수 (기본: 20)
- `--num_online_clients`: 각 라운드에 참여하는 클라이언트 수 (기본: 8)
- `--num_rounds`: 연방 학습 라운드 수 (기본: 200)
- `--num_epochs_local_training`: 클라이언트 로컬 학습 에폭 수 (기본: 10)
- `--match_epoch`: 서버 측 합성 특징 최적화 반복 수 (기본: 100)
- `--crt_epoch`: 서버 측 classifier 재훈련 반복 수 (기본: 300)
- `--alpha`: BKD2 distillation 계수 (기본: 1.0)
- `--contrast_alpha`: 서버 CLIP 손실 가중치 (기본: 0.001)
- `--non_iid_alpha`: 비동질성 제어 (기본: 0.5, 작을수록 더 비동질)
- `--imb_factor`: 불균형도 제어 (기본: 0.01, 작을수록 더 불균형)
- `--lr_local_training`: 클라이언트 학습률 (기본: 0.1)
- `--lr_feature`: 합성 특징 학습률 (기본: 0.1)
- `--lr_net`: 네트워크 파라미터 학습률 (기본: 0.01)
- `--gpu`: 사용할 GPU 번호 (기본: 1)
- `--result_save`: 결과 저장 디렉토리 (기본: `results`)
- `--resume`: 체크포인트에서 재개할 경로 (선택사항)

### 5. 결과 저장

실행 시 다음 파일들이 자동으로 저장됩니다:

- **로그 파일**: `results/{dataset}/main_clip2fl_bkd/{experiment_id}.log`
- **설정 파일**: `results/{dataset}/main_clip2fl_bkd/{experiment_id}_config.json`
- **메타데이터**: `results/{dataset}/main_clip2fl_bkd/{experiment_id}_metadata.json`
- **결과 파일**: `results/{dataset}/main_clip2fl_bkd/{experiment_id}_results.json`
- **체크포인트**: `results/{dataset}/main_clip2fl_bkd/checkpoints/{experiment_id}_checkpoint_round_{N}.pth`
- **최종 모델**: `results/{dataset}/main_clip2fl_bkd/checkpoints/{experiment_id}_final_model.pth`

체크포인트는 매 10라운드마다 자동 저장되며, 중단된 실험을 재개할 수 있습니다.

### 6. 체크포인트에서 재개

중단된 실험을 재개하려면:

```bash
python main_bkd.py \
  --dataset cifar10 \
  --resume results/cifar10/main_clip2fl_bkd/checkpoints/{experiment_id}_checkpoint_round_{N}.pth \
  ... (나머지 옵션 동일)
```

### 7. 여러 GPU 병렬 실행

#### 방법 1: 각 터미널에서 개별 실행

각 터미널을 열어서 다른 GPU로 실행:

**터미널 1 (GPU 0):**
```bash
conda activate clip2fl
cd /workspace/basic_code_semina_01/hanbat/CLIP2FL
python main_bkd.py --dataset cifar10 --gpu 0 --result_save results ...
```

**터미널 2 (GPU 1):**
```bash
conda activate clip2fl
cd /workspace/basic_code_semina_01/hanbat/CLIP2FL
python main_bkd.py --dataset cifar100 --gpu 1 --result_save results ...
```

**터미널 3 (GPU 2):**
```bash
conda activate clip2fl
cd /workspace/basic_code_semina_01/hanbat/CLIP2FL
python main_bkd.py --dataset stl10 --gpu 2 --result_save results ...
```

#### 방법 2: GPU 사용량 확인

```bash
# 실시간 GPU 사용량 모니터링
watch -n 1 nvidia-smi

# 또는
nvidia-smi -l 1
```

### 8. 실험 팁

- **빠른 테스트**: `--small_match_epoch 3`, `--small_crt_epoch 3`로 빠른 검증 가능
- **재현성**: `--seed` 고정 (기본: 7)으로 동일한 결과 재현 가능
- **불균형 조정**: `--imb_factor`를 조정하여 데이터 불균형도 변경 (0.01, 0.02, 0.1 등)
- **비동질성 조정**: `--non_iid_alpha`를 조정하여 클라이언트 간 데이터 분포 조정 (작을수록 더 비동질)
- **학습률 조정**: 데이터셋에 따라 `--lr_local_training`, `--lr_feature`, `--lr_net` 조정 필요

### 9. 결과 확인

실험 결과는 JSON 파일로 저장되며, 다음 정보를 포함합니다:

- `final_accuracy`: 최종 정확도
- `best_accuracy`: 최고 정확도
- `all_accuracies`: 모든 라운드의 정확도 리스트
- `total_rounds`: 총 라운드 수
- `total_time_seconds`: 총 실행 시간 (초)
- `total_time_hours`: 총 실행 시간 (시간)

결과 파일을 파싱하거나 직접 확인하여 실험 결과를 분석할 수 있습니다.
