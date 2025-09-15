# 파일 이름: create_dataset_config.py
# 설명: create_dataset.py 스크립트의 모든 설정을 관리합니다.

import os
from pathlib import Path

# --- 1. 피처 계산 선택 ---
# 각 피처를 계산할지 여부를 True/False로 설정합니다.
CALCULATE_ESM_EMBEDDING = True
CALCULATE_BB_TORSION_ANGLES = True  # (phi, psi, omega)
CALCULATE_BB_BOND_ANGLES = True     # (N-CA-C, CA-C-N, C-N-CA)
CALCULATE_BB_BOND_DISTANCES = True  # (N-CA, CA-C, C-N)
CALCULATE_SC_TORSION_ANGLES = False  # 사이드체인 피처는 더 이상 계산하지 않음
CALCULATE_BACKBONE_COORDS = True    # (N, CA, C 원자의 3D 좌표)

# --- 2. 데이터셋 구성 방식 선택 ---
# 'train_test' 또는 '10_fold' 중 하나를 선택합니다.
# 'train_test': PDB_TRAIN_DIR와 PDB_TEST_DIR 경로를 사용하여 train/test 데이터셋을 생성합니다.
# '10_fold': PDB_ROOT_DIR 아래의 'fold_0', 'fold_1', ... 'fold_9' 디렉토리를 찾아 10-fold 데이터셋을 생성합니다.
DATASET_MODE = 'train_test' # 또는 '10_fold'

# --- 3. 데이터셋 분할 설정 (train/validation) ---
# 'train_test' 모드에서만 사용됩니다.
# PDB_TRAIN_DIR의 데이터를 훈련(train)과 검증(validation) 데이터셋으로 나눌 비율입니다.
# 예: 0.1은 10%를 검증용으로 사용한다는 의미입니다.
VALIDATION_SPLIT_RATIO = 0.1

# --- 4. 입출력 경로 설정 ---
# 입력 PDB 파일이 있는 루트 디렉토리
# 10_fold 모드에서는 이 디렉토리 하위에 fold_n 폴더들이 있어야 합니다.
PDB_ROOT_DIR = Path("/home/nohtaeil/Desktop/peptide_generation_AI/pep_diffusion/data/modpep_dataset/pdbs/splitted_peptide_pdbs")

# 'train_test' 모드에서 사용할 경로
PDB_TRAIN_DIR = "/home/nohtaeil/Desktop/peptide_generation_AI/pep_diffusion/data/resolution2.5complexes/peptide/resolution2.5filtered/without_noncanonical_amino_acid"#res 2.5 로 train
PDB_TEST_DIR = PDB_ROOT_DIR / "test"

# 생성된 데이터셋(.pkl 파일)이 저장될 디렉토리
DATASET_OUTPUT_DIR = PDB_ROOT_DIR / "datasets_generated"
# 생성될 데이터셋 파일 이름 (접두사)
OUTPUT_FILENAME_PREFIX = "peptide_feature_dataset"

# --- 5. 데이터 처리 파라미터 ---
# 펩타이드의 최대 길이. 이 길이를 초과하는 펩타이드는 데이터셋에서 제외됩니다.
MAX_LEN = 30
# 한 번에 처리할 PDB 파일의 수 (배치 크기)
BATCH_SIZE = 16

# --- 6. ESM 모델 설정 ---
# 사용할 ESM-2 모델 이름
ESM_MODEL_NAME = 'esm2_t6_8M_UR50D'
# 임베딩을 추출할 ESM-2 모델의 레이어 번호
ESM_REPR_LAYER = 6 # esm2_t6_8M_UR50D의 경우 6

# --- 7. Rosetta 관련 설정 ---
# 표준 아미노산 목록 (3-letter code) - 원-핫 인코딩에 계속 사용됨
CANONICAL_AMINO_ACIDS = {
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"
}
