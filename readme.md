# PepDiff: 펩타이드 3D 구조 생성을 위한 Transformer based Diffusion Model

## 1. 프로젝트 소개

**PepDiff**는 아미노산 서열 정보로부터 펩타이드의 3차원 백본(Backbone) 구조를 생성하는 Diffusion 모델 기반의 딥러닝 프로젝트입니다. 최신 Transformer 아키텍처를 활용하여 물리적으로 타당하고 정확한 3D 구조를 생성하는 것을 목표로 합니다.

모델은 직접 3D 좌표를 예측하는 대신, 구조를 결정하는 핵심 요소인 **이면각(torsion angles)**과 **결합각(bond angles)**을 예측합니다. 이후 **NERF(Natural Extension Reference Frame)** 알고리즘을 통해 이 각도 정보로부터 최종적인 backbone 3D 원자 좌표를 결정론적으로 재구성합니다.

## 2. 주요 특징

-   **Diffusion Model 기반 생성**: 노이즈로부터 점진적으로 펩타이드의 각도 정보를 복원하여 고품질의 다양한 구조를 생성합니다.
-   **Transformer 아키텍처**: `DiT(Diffusion Transformer)` 또는 `Cac(Cross-Attention Conditioning)` 아키텍처를 선택하여 사용할 수 있습니다.
-   **ESM Embedding 활용**: 펩타이드 서열의 진화적 정보를 담은 ESM(Evolutionary Scale Modeling) 임베딩을 조건(Condition)으로 사용하여 생성 성능을 높입니다.
-   **전체 구조 완성 파이프라인**: Diffusion 모델로 백본(Backbone) 구조를 생성한 후, ProteinMPNN을 이용해 아미노산 서열을 디자인하고 PyRosetta로 구조를 안정화(Relax)하는 전체 파이프라인을 제공합니다.
-   **재현 가능한 개발 환경**: **Docker**를 통해 모든 의존성을 포함한 개발 환경을 이미지로 제공하여, 어떤 머신에서든 동일한 환경에서 코드를 실행할 수 있습니다.

## 3. 프로젝트 구조

```
.
├── create_dataset.py         # PDB 파일로부터 학습용 데이터셋(.pkl) 생성
├── train.py                  # 모델 학습 메인 스크립트
├── generate_backbone_only_pdb.py   # 학습된 모델로 Backbone PDB 파일 생성
├── generate_complete_pdb.py      # Backbone PDB로부터 전체 구조를 완성하고 안정화
├── evaluation.py             # 단일 모델의 생성 결과 평가
├── comparison_evaluation.py  # 여러 모델의 생성 결과 비교 평가
│
├── model_config.py           # 모델, 학습, 데이터 경로 등 핵심 설정
├── create_dataset_config.py  # 데이터셋 생성 관련 설정
├── generate_backbone_only_pdb_config.py # Backbone PDB 생성 관련 설정
├── generate_complete_pdb_config.py    # 전체 구조 완성 관련 설정
├── evaluation_config.py      # 단일 모델 평가 관련 설정
├── comparison_evaluation_config.py # 비교 평가 관련 설정
│
├── model.py                  # 모델 아키텍처(DiT, Cac) 및 DiffusionTrainer 정의
├── utils/                    # NERF, 각도 계산 등 유틸리티 함수
│
└── readme.md                 # 프로젝트 설명 파일
```

## 4. 사용 방법

### 1단계: 개발 환경 설정

이 프로젝트는 모든 의존성이 설치된 Docker 이미지를 통해 실행하는 것을 권장합니다.

**사전 요구사항:**
-   NVIDIA Driver가 설치된 Linux 환경
-   [Docker](https://docs.docker.com/engine/install/)
-   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

**Docker 이미지 로드:**
(제공된 Docker 이미지 압축파일이 있는 경우에 아래 명령어를 실행하세요)
```bash
docker load < pepdiff-env.tar
```

**Docker 컨테이너 실행:**
```bash
# Docker 컨테이너를 시작하고 내부 쉘(bash)로 진입합니다.
docker run -it --rm --gpus all pepdiff-env
```

### 3단계: 컨테이너 내부에서 Git Clone
위 명령어를 실행하면 컨테이너 내부(`root@...:/app#`)로 접속됩니다. 이제 컨테이너 안에서 프로젝트 코드를 다운로드합니다.
```bash
# 컨테이너 내부에서 실행
git clone https://github.com/bsipl/pepdiff.git

```


1.  **PDB 파일 준비**: `data/pdbs/train` 및 `data/pdbs/test` 와 같은 폴더를 만들고, 학습 및 테스트에 사용할 원본 PDB 파일들을 각각 위치시킵니다.

2.  **설정 수정 (`create_dataset_config.py`)**: PDB 파일 경로 및 데이터셋이 저장될 경로를 설정합니다.
    ```python
    # 원본 PDB 파일들이 담긴 폴더 경로
    PDB_TRAIN_DIR = "/workspace/data/pdbs/train"
    PDB_TEST_DIR = "/workspace/data/pdbs/test"

    # 생성될 .pkl 데이터셋이 저장될 폴더 경로
    DATASET_OUTPUT_DIR = "/workspace/data/processed"
    ```

3.  **스크립트 실행**: 컨테이너 내부의 `codes` 폴더에서 아래 명령어를 실행합니다.
    ```bash
    cd /workspace/codes
    python create_dataset.py
    ```
    -   `DATASET_OUTPUT_DIR`에 `_train.pkl`, `_val.pkl`, `_test.pkl` 파일이 생성됩니다.

### 3단계: 모델 학습

1.  **설정 수정 (`model_config.py`)**: 학습에 사용할 데이터셋 경로, 모델 아키텍처, 하이퍼파라미터 등을 설정합니다.
    ```python
    # 학습/검증/테스트 데이터셋 경로
    "train_dataset_path": '/workspace/data/processed/peptide_feature_dataset_train.pkl',
    "val_dataset_path": '/workspace/data/processed/peptide_feature_dataset_val.pkl',
    "test_dataset_path": '/workspace/data/processed/peptide_feature_dataset_test.pkl',

    # 학습된 모델이 저장될 경로
    "save_path": "/workspace/trained_models/MyModel.pt",

    # 모델 아키텍처 선택 ('DiT' 또는 'Cac')
    "type": "Cac",
    # ... epochs, batch_size, learning_rate 등
    ```

2.  **학습 실행**: 컨테이너에서 아래 명령어를 실행합니다.
    ```bash
    python train.py
    ```
    -   학습된 모델(`.pt`)은 `save_path`에 지정된 경로에 저장됩니다.

### 4단계: Backbone 구조 생성

1.  **설정 수정 (`generate_backbone_only_pdb_config.py`)**: 학습된 모델 경로와 생성할 샘플 수 등을 설정합니다.
    ```python
    # 사용할 학습된 모델 경로
    "trained_model_path": "/workspace/trained_models/MyModel.pt",
    # PDB 생성을 위한 입력 데이터셋 (보통 test 셋)
    "dataset_path": "/workspace/data/processed/peptide_feature_dataset_test.pkl",
    # 생성된 Backbone PDB 파일이 저장될 디렉토리
    "output_pdb_dir": "./generated_backbone_only_pdbs",
    
    # 각 서열 당 생성할 샘플 PDB 개수
    "num_samples_per_sequence": 1000, 
    # ... batch_size, guidance_scale 등
    ```

2.  **생성 실행**: 컨테이너에서 아래 명령어를 실행합니다.
    ```bash
    python generate_backbone_only_pdb.py
    ```
    -   `output_pdb_dir`에 지정된 폴더에 Backbone 구조만 포함된 PDB 파일들이 생성됩니다.

### 5단계: 전체 구조 완성 및 안정화

1.  **설정 수정 (`generate_complete_pdb_config.py`)**: Backbone PDB가 있는 폴더와 최종 PDB를 저장할 폴더 경로를 확인합니다. 대부분의 경로는 자동으로 설정되므로 특별히 수정할 필요는 없습니다.
    ```python
    # 입력 디렉토리 (generate_backbone_only_pdb.py의 출력)
    "input_backbone_pdb_dir": "./generated_backbone_only_pdbs",

    # 최종 출력 디렉토리 (Relax까지 완료된 PDB)
    "output_complete_pdb_dir": "./generated_complete_pdbs/",
    ```

2.  **실행**: 컨테이너에서 아래 명령어를 실행합니다. 이 스크립트는 내부적으로 **`generate_backbone_only_pdb.py`를 먼저 실행**하므로, 4단계를 건너뛰고 이 스크립트만 실행해도 전체 파이프라인이 동작합니다.
    ```bash
    python generate_complete_pdb.py
    ```
    -   `output_complete_pdb_dir` 폴더에 Side-chain이 추가되고 PyRosetta Relax까지 완료된 최종 PDB 파일들과 에너지 로그 파일(`relax_energies.txt`)이 저장됩니다.

### 6단계: 성능 평가

1.  **설정 수정 (`evaluation_config.py`)**: 평가할 원본 PDB와 생성된 최종 PDB 샘플 폴더 경로 등을 설정합니다.
    ```python
    # 평가 결과(그래프, 캐시)가 저장될 디렉토리
    "OUTPUT_DIR": "/workspace/evaluation_results/MyModel_eval",
    # 원본 PDB 파일 경로
    "ORIGIN_PDB_DIR": "/workspace/data/pdbs/test",
    # 생성된 최종 PDB 샘플 경로
    "SAMPLES_PDB_DIR": "./generated_complete_pdbs",
    ```

2.  **평가 실행**: 컨테이너에서 아래 명령어를 실행합니다.
    ```bash
    python evaluation.py
    ```
    -   `OUTPUT_DIR`에 RMSD 계산 결과 캐시(`rmsd_cache.pkl`)와 다양한 평가 그래프(`.png`) 파일들이 생성됩니다.
