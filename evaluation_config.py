# 파일 이름: evaluation_config.py
# 설명: Rosetta 특징 기반 모델의 분석을 위한 설정 파일입니다.

from pathlib import Path

# --- 데이터셋 생성 및 모델 설정 파일 임포트 ---
import model_config as m_config
# generate_complete_pdb.py의 최종 출력 폴더를 가져오기 위함
import generate_complete_pdb_config as complete_config 

class Config:
    """Configuration settings for the rffeat_modpep dataset analysis."""

    # --- 분석 스위치 ---
    USE_IQR_FILTER: bool = False

    # --- 데이터셋 및 파일 경로 ---
    # 분석에 사용할 테스트 데이터셋 파일 (백본 생성 시 사용된 데이터셋과 동일)
    DATASET_FILE = Path(m_config.PATHS["test_dataset"])
    
    # 결과가 저장될 최상위 폴더
    OUTPUT_DIR = Path("/home/nohtaeil/Desktop/peptide_generation_AI/pep_diffusion/result/DiffusionDitModelResults")
    
    # 원본 PDB와 생성된 PDB가 있는 폴더 경로
    # (참고) create_dataset.py의 경로 설정과 일관성을 유지해야 함
    ORIGIN_PDB_DIR = Path("/home/nohtaeil/Desktop/peptide_generation_AI/pep_diffusion/data/modpep_dataset/pdbs/splitted_peptide_pdbs/test")
    SAMPLES_PDB_DIR = Path(complete_config.PATHS["output_complete_pdb_dir"])

    # --- 파일 이름 형식 ---
    ORIGIN_FNAME_TPL = "{pdb_id}.pdb"
    # generate_complete_pdb.py의 최종 파일 이름 형식: {pdb_id}_sample_{sample_idx}_complete.pdb
    SAMPLE_FNAME_TPL = "{pdb_id}_sample_{sample_idx}_complete.pdb"

    # --- 핵심 파라미터 ---
    # 생성된 PDB는 전체 아미노산를 포함하지만, backbone rmsd 평가를 위해 N/CA/C만 사용
    ATOM_TYPES = ("N", "CA", "C")
    
    MAX_SAMPLES = 100
    PEPTIDE_MAX_LEN = m_config.DATA_SPECS["max_len"]

    # --- 분석별 파라미터 ---
    IQR_K_VALUE: float = 1.5
    PEPTIDE_LEN_SAMPLE_COUNTS = [50, 100]
    SUCCESS_RATE_MAX_SAMPLES = 1000
    SUCCESS_RATE_STEP = 20
    SUCCESS_RATE_LEN_GROUP_SIZE = 5
    SUCCESS_RATE_N0 = 3
