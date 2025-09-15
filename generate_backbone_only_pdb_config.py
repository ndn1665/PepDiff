import torch
import model_config

# ==============================================================================
# 1. 경로 설정
# ==============================================================================
PATHS = {
    # 생성에 사용할 학습 완료된 모델(.pt) 파일 경로
    "trained_model_path": model_config.PATHS["save_path"],
    
    # 구조 생성을 위한 조건(서열, ESM 임베딩 등)을 가져올 데이터셋 경로
    "dataset_path": model_config.PATHS["test_dataset"],
    
    # 생성된 PDB 파일이 저장될 디렉토리
    "output_pdb_dir": "./generated_backbone_only_pdbs",

}

# ==============================================================================
# 2. 생성 파라미터
# ==============================================================================
# 물리적으로 타당한 이상적인 백본 결합 길이 (Angstrom)
# 출처: AlphaFold2 및 일반적인 생화학적 값
IDEAL_BOND_LENGTHS = {
    "N_CA_LENGTH": 1.458,
    "CA_C_LENGTH": 1.525,
    "C_N_LENGTH": 1.329,
}

GENERATION_PARAMS = {
    # --- 모델 및 데이터 경로 ---
    "model_checkpoint_path": model_config.PATHS["save_path"],
    "test_dataset_path": model_config.PATHS["test_dataset"],

    # --- 구조 생성 방식 설정 ---
    "use_ideal_bond_lengths": True,  # True: 아래 정의된 이상적인 결합 길이 사용, False: 데이터셋의 실측값 사용
    
    # --- 생성 프로세스 설정 ---
    "num_samples_per_sequence": 1000,  # 각 서열 당 생성할 샘플 PDB 개수
    "guidance_scale": 1.0,  # Classifier-Free Guidance 강도 (0.0은 비활성화)
    "batch_size": 16,  # DataLoader에 사용할 배치 크기
    "samples_batch_size": 1000, # diffusion.sample()에 한번에 넣을 샘플 수 (GPU 메모리에 맞게 조절)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "multi_gpu": True, # 여러 GPU를 사용할지 여부 (DataParallel)
    "precision": "fp16", # "fp16" 또는 "fp32"
    
    # --- 출력 설정 ---
    "output_pdb_dir": "./generated_backbone_only_pdbs",
}
