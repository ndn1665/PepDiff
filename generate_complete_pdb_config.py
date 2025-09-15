import os
import generate_backbone_only_pdb_config as backbone_config

# ==============================================================================
# 1. 경로 설정
# ==============================================================================
PATHS = {
    # 원본 서열 정보를 가져올 데이터셋 경로
    "dataset_path": backbone_config.PATHS["dataset_path"],

    # 입력 디렉토리: 백본만 있는 PDB 파일이 위치한 곳
    # generate_backbone_only_pdb.py의 출력 디렉토리를 사용합니다.
    "input_backbone_pdb_dir": backbone_config.PATHS["output_pdb_dir"],

    # 최종 출력 디렉토리: Relax까지 완료된 PDB 파일이 저장될 곳
    "output_complete_pdb_dir": "./generated_complete_pdbs/",

    # 스크립트 경로
    "add_oxygen_script": "bin/add_oxygen_to_backbone.py",
    "splice_script": "bin/splice_aa_onto_backbone.py",
    "relax_script": "utils/run_pyrosetta_relax.py",
}

# ==============================================================================
# 2. 실행 파라미터
# ==============================================================================
PARAMS = {
    # 병렬 처리를 위한 최대 프로세스 수 (0이면 사용 가능한 모든 코어 사용)
    "max_workers": os.cpu_count() -1,
}
