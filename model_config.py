import torch

# ==============================================================================
# 1. 경로 및 기본 설정
# ==============================================================================
PATHS = {
    "train_dataset": '/home/nohtaeil/Desktop/peptide_generation_AI/pep_diffusion/data/modpep_dataset/pdbs/splitted_peptide_pdbs/datasets_generated/peptide_feature_dataset_train.pkl',#train용 dataset이 위치하고있는 경로
    "test_dataset":  '/home/nohtaeil/Desktop/peptide_generation_AI/pep_diffusion/data/modpep_dataset/pdbs/splitted_peptide_pdbs/datasets_generated/peptide_feature_dataset_test.pkl',#test용 dataset이 위치하고있는 경로
    "save_path":     "DiffusionBERTDitSumModel.pt",#pretrained 학습 파라미터
}

# ==============================================================================
# 2. 학습 하이퍼파라미터
# ==============================================================================
TRAINING_PARAMS = {
    "epochs": 1500,
    "early_stop_patience": 50, # FAPE 도입기 변동성을 고려하여 patience 증가
    "batch_size": 128,
    "model_architecture": "dit",  # "dit" or "cac"
    "hidden_dim": 512,
    "feature_dim": 6,
    "learning_rate": 5e-5,
    "post_fape_learning_rate": 1e-4,  # FAPE 도입 이후 LR 상향으로 수렴 가속
    "warmup_ratio": 0.1,
    "condition_dropout": 0.1, # Classifier-Free Guidance를 위한 조건 드롭아웃 확률
    "fape_loss_coef": 0.5,  # FAPE loss 가중치
}

# ==============================================================================
# 3. 데이터 및 피처 설정
# ==============================================================================
# create_dataset.py에서 생성된 .pkl 파일의 키(key)와 일치해야 합니다.

# 이 리스트에 포함된 키의 데이터가 모델의 입력 피처(x)로 사용됩니다.
# 종류 : ['bb_torsions', 'bb_angles','bb_distances', 'sc_torsions']
FEATURE_KEYS = [
    'bb_torsions',  # (L, 3) - phi, psi, omega
    'bb_angles',    # (L, 3) - N-CA-C, CA-C-N, C-N-CA
]

# 각 피처의 차원을 정의 (데이터 로딩 시 검증 및 모델 구성에 사용)

FEATURE_DIMS = {
    'bb_torsions': 3,
    'bb_angles': 3,
    # 'sc_torsions': 6, # 예시: 사이드체인 추가 시
}

# 각 피처가 각도(angular)인지 여부를 정의 (주기성 손실 함수 적용 대상)
FEATURE_IS_ANGULAR = {
    'bb_torsions': True,
    'bb_angles': True,
}

# 이 리스트에 포함된 키의 데이터가 모델의 조건(condition)으로 사용됩니다.
CONDITION_KEYS = [
    'sequence_encoded', # (L, 20)
    'esm_embedding',    # (L, 320)
]

# 각 조건의 차원을 정의합니다.
CONDITION_DIMS = {
    'sequence_encoded': 20,
    'esm_embedding': 320, # esm2_t6_8M_UR50D 모델 기준
}

# 3D 구조 재구성에 필요한 실제 결합 길이 정보가 담긴 키
BOND_KEY = "bb_distances"

DATA_SPECS = {
    'data_dir': '/home/nohtaeil/Desktop/peptide_generation_AI/pep_diffusion/dataset/',
    'train_dataset_path': 'peptide_feature_dataset_train.pkl',
    'val_dataset_path': 'peptide_feature_dataset_val.pkl',
    'test_dataset_path': 'peptide_feature_dataset_test.pkl',
    'max_len': 30, # 펩타이드 최대 길이
}

# ==============================================================================
# 4. 모델 아키텍처 설정
# ==============================================================================
MODEL_ARCH = {
    "hidden_size": 512,
    "n_layers": 12,
    "n_heads": 8,
    "dit_modulation_strategy": "sum", # DiT 모델의 조건부 주입 방식: "sum", "separate", "sequential"
    "condition_merge_strategy": "gated", # CAC 모델의 조건 결합 방식: "sum", "concat", "gated", "film"
}

# ==============================================================================
# 5. Diffusion 모델 파라미터
# ==============================================================================
DIFFUSION_PARAMS = {
    "cond_prob": 0.9,       # Classifier-Free Guidance를 위한 조건부 학습 확률
    "timesteps": 1000,      # 총 디퓨전 타임스텝 수
    "guidance_scale": 1.5,  # 샘플링 시 조건부/비조건부 예측 혼합 비율
}

# ==============================================================================
# 6. 손실 함수 가중치
# ==============================================================================
# FEATURE_KEYS에 명시된 각 피처에 대한 손실 가중치
# 여기에 정의되지 않은 피처는 가중치 1.0으로 처리됩니다.
LOSS_WEIGHTS = {
    "bb_torsions": 1.0,
    "bb_angles":   1.0,
}

# ==============================================================================
# 손실 가중치 스케줄링 (Ramp-up)
# 에포크 기반 가중치 동적 조절 설정 (이제 feat_loss_weight 고정용으로만 사용)
# ==============================================================================
RAMP_CONFIG = {
    "feat_loss_end_weight": 1.0, # feat_loss에 적용할 고정 가중치
}

# ==============================================================================
# FAPE Loss 설정
# ==============================================================================
FAPE_LOSS_CONFIG = {
    "enabled": False,
    "weight_schedule": [0.01, 0.2],  # [min_weight_at_t=T-1, max_weight_at_t=0]
    "clamp_end": 10.0,                # FAPE loss 계산 시 사용할 고정된 clamp 거리
    "clip_value": 50.0,
    # 아래 파라미터들은 에포크 기반 스케줄링이 제거됨에 따라 더 이상 사용되지 않습니다.
    # "clamp_start": 10.0,
    # "ca_only_threshold": 0.15,
    # "local_window_threshold": 0.30
}


# ==============================================================================
# 7. 기타 설정
# ==============================================================================
MISC = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "multi_gpu": True, # 여러 GPU를 사용할지 여부 (DataParallel)
    "num_workers": 4 if torch.cuda.is_available() else 0, # 데이터 로딩 워커 수
    # Wandb 로깅 설정 (프로젝트 이름을 설정하면 활성화됨)
    "wandb_project": "peptide_diffusion_benchmark", # 예: "my_peptide_project"
    "wandb_entity": None,                # Wandb 사용자 또는 팀 이름 (선택 사항)
    "wandb_run_name": "CAC_Model_Baseline", # 이번 실행(run)에 부여할 이름
}

def to_dict():
    """Wandb 로깅을 위해 모든 설정을 하나의 딕셔너리로 변환합니다."""
    return {
        "PATHS": PATHS,
        "TRAINING_PARAMS": TRAINING_PARAMS,
        "FEATURE_KEYS": FEATURE_KEYS,
        "FEATURE_DIMS": FEATURE_DIMS,
        "CONDITION_KEYS": CONDITION_KEYS,
        "CONDITION_DIMS": CONDITION_DIMS,
        "BOND_KEY": BOND_KEY,
        "DATA_SPECS": DATA_SPECS,
        "MODEL_ARCH": MODEL_ARCH,
        "DIFFUSION_PARAMS": DIFFUSION_PARAMS,
        "LOSS_WEIGHTS": LOSS_WEIGHTS,
        "FAPE_LOSS_CONFIG": FAPE_LOSS_CONFIG,
        "RAMP_CONFIG": RAMP_CONFIG,
        "MISC": {k: v for k, v in MISC.items() if k != 'device'} # device 객체는 직렬화 불가
    }
