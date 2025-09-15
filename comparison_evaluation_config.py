# comparison_config.py

from pathlib import Path
import model_config as m_config
import generate_pdb_config as g_config

class Config:
    """펩타이드 구조 생성 방법론 비교 분석을 위한 설정"""

    # --- 데이터셋 및 기본 경로 설정 (model_config, generate_pdb_config와 동기화) ---
    DATASET_FILE = Path(m_config.PATHS["test_dataset"])
    ORIGIN_PDB_DIR = Path("/home/nohtaeil/Desktop/peptide_generation_AI/pep_diffusion/data/modpep_dataset/pdbs/splitted_peptide_pdbs/test")
    OUTPUT_DIR = Path("/home/nohtaeil/Desktop/peptide_generation_AI/pep_diffusion/result/DiffusionDitModelResults")

    # --- 파일 이름 템플릿 ---
    ORIGIN_FNAME_TPL = "{pdb_id}.pdb"
    # (참고) 각 방법론별 샘플 파일 이름 형식은 다를 수 있으므로,
    # graph_generator에서 유연하게 처리하는 것이 좋습니다.
    # 아래는 우리 모델의 기본 형식을 나타냅니다.
    SAMPLE_FNAME_TPL = "gen_{pdb_id}_sample_{sample_idx}.pdb"

    # --- !중요! 비교할 방법론 정의 ---
    # 여기에 비교하고 싶은 방법론의 정보를 추가, 수정, 삭제하세요.
    METHODS = {
        "DIT (Ours)": { # 사용자 정의 이름
            "path": Path(g_config.PATHS["output_pdb_dir"]),
            "color": "blue",
            "marker": "^"
        },
        "MODPEP": {
            "path": Path("/home/nohtaeil/Desktop/peptide_generation_AI/pep_diffusion/generated_pdbs/pdb_by_modpep_program/samples"),
            "color": "red",
            "marker": "o"
        },
        "RDKit": {
            "path": Path("/home/nohtaeil/Desktop/peptide_generation_AI/pep_diffusion/generated_pdbs/pdb_by_rdkit/samples"),
            "color": "green",
            "marker": "s"
        },
    }

    # --- 공통 분석 파라미터 ---
    ATOM_TYPES = ("CA", "N", "C", "O")
    AA = [
        "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
        "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR",
    ]
    
    # --- 정확도(cRMSD) 분석 파라미터 ---
    ACCURACY_MAX_ENSEMBLE_SIZE = 200 # 예시 그래프와 유사하게 200으로 설정
    USE_IQR_FILTER = False # 이상치(outlier) 제거 여부
    IQR_K_VALUE = 1.5

    # --- 성공률 분석 파라미터 ---
    SUCCESS_MAX_ENSEMBLE_SIZE = 200 # 예시 그래프와 유사하게 200으로 설정
    SUCCESS_RATE_STEP = 5           # 앙상블 크기 스텝
    SUCCESS_RATE_N0 = 3           # 성공 기준(RMSD) 계산에 사용되는 파라미터 (조정 가능)