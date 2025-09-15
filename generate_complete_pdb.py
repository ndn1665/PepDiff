import os
import subprocess
import re
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from threading import Lock
from tqdm import tqdm
import generate_complete_pdb_config as config

# --- 설정값 불러오기 ---
DATASET_PATH = config.PATHS["dataset_path"]
INPUT_PDB_DIR = config.PATHS["input_backbone_pdb_dir"]
OUTPUT_COMPLETE_PDB_DIR = config.PATHS["output_complete_pdb_dir"]
MAX_WORKERS = config.PARAMS["max_workers"] if config.PARAMS["max_workers"] > 0 else os.cpu_count()

# 중간 파일 및 로그 경로 설정
OUTPUT_INTERMEDIATE_DIR = os.path.join(OUTPUT_COMPLETE_PDB_DIR, "intermediate_files")
ENERGY_LOG_FILE = os.path.join(OUTPUT_COMPLETE_PDB_DIR, "relax_energies.txt")

# 스크립트 경로
ADD_OXYGEN_SCRIPT_PATH = config.PATHS["add_oxygen_script"]
SPLICE_SCRIPT_PATH = config.PATHS["splice_script"]
RELAX_SCRIPT_PATH = config.PATHS["relax_script"]
# --- 설정 종료 ---

# 에너지 파일 쓰기를 위한 Lock 객체
energy_file_lock = Lock()

def run_pipeline_step(command, step_name=""):
    """파이프라인의 단일 스텝(subprocess)을 실행하고 결과를 출력합니다."""
    print(f"\n----- Starting: {step_name} -----")
    print(f"Running command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"--- ERROR in {step_name} ---")
        print("--- STDOUT ---")
        print(stdout)
        print("--- STDERR ---")
        print(stderr)
        raise subprocess.CalledProcessError(process.returncode, command, output=stdout, stderr=stderr)
    
    print(f"----- Finished: {step_name} -----")
    return stdout

def run_command_for_worker(command, log_file):
    """(Worker용) 주어진 명령어를 실행하고 stdout/stderr를 로그 파일에 기록하고 stdout을 반환합니다."""
    with open(log_file, 'a') as f:
        f.write(f"\n--- Running command: {' '.join(command)} ---\n")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        f.write("--- STDOUT ---\n")
        f.write(stdout)
        f.write("\n--- STDERR ---\n")
        f.write(stderr)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, output=stdout, stderr=stderr)
    return stdout

def parse_energies_from_output(output):
    """relax 스크립트의 출력에서 에너지 값을 파싱합니다."""
    before_energy = re.search(r"Energy before relax: ([-+]?\d*\.\d+|\d+)", output)
    after_energy = re.search(r"Energy after relax: ([-+]?\d*\.\d+|\d+)", output)
    
    before = float(before_energy.group(1)) if before_energy else None
    after = float(after_energy.group(1)) if after_energy else None
    
    return before, after

def process_single_pdb(args):
    """(Worker용) 단일 백본 PDB에 side chain 추가 및 relax 수행."""
    backbone_pdb_path, sequence = args
    try:
        pdb_id_with_sample = backbone_pdb_path.stem
        
        # 경로 설정
        oxygenated_pdb_path = os.path.join(OUTPUT_INTERMEDIATE_DIR, f"{pdb_id_with_sample}_with_O.pdb")
        temp_fasta_path = os.path.join(OUTPUT_INTERMEDIATE_DIR, f"{pdb_id_with_sample}.fasta")
        unrelaxed_pdb_path = os.path.join(OUTPUT_INTERMEDIATE_DIR, f"{pdb_id_with_sample}_unrelaxed.pdb")
        final_pdb_path = os.path.join(OUTPUT_COMPLETE_PDB_DIR, f"{pdb_id_with_sample}_complete.pdb")
        log_dir = os.path.join(OUTPUT_INTERMEDIATE_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"{pdb_id_with_sample}.log")

        if os.path.exists(final_pdb_path):
            return f"Skipped (already exists): {final_pdb_path}"
        
        open(log_file_path, 'w').close()

        # --- 파이프라인 실행 ---
        # 1. 백본에 산소(O) 추가
        add_oxygen_cmd = ["python", ADD_OXYGEN_SCRIPT_PATH, str(backbone_pdb_path), oxygenated_pdb_path]
        run_command_for_worker(add_oxygen_cmd, log_file_path)

        # 2. 원본 서열로 임시 FASTA 파일 생성
        with open(temp_fasta_path, 'w') as f:
            f.write(f">{pdb_id_with_sample}\n{sequence}\n")

        # 3. 생성된 FASTA를 이용해 서열을 백본에 결합
        splice_cmd = ["python", SPLICE_SCRIPT_PATH, "--backbone_pdb", oxygenated_pdb_path, "--fasta", temp_fasta_path, "--output_pdb", unrelaxed_pdb_path]
        run_command_for_worker(splice_cmd, log_file_path)
        
        # 4. PyRosetta Relax 실행
        relax_cmd = ["python", RELAX_SCRIPT_PATH, unrelaxed_pdb_path, final_pdb_path]
        relax_output = run_command_for_worker(relax_cmd, log_file_path)
        
        # 5. 에너지 파싱 및 파일에 기록
        before, after = parse_energies_from_output(relax_output)
        if before is not None and after is not None:
            with energy_file_lock:
                with open(ENERGY_LOG_FILE, 'a') as f:
                    f.write(f"{pdb_id_with_sample},{before:.4f},{after:.4f}\n")
        
        # 임시 FASTA 파일 삭제
        os.remove(temp_fasta_path)

        return f"Successfully processed: {pdb_id_with_sample}"
    except Exception as e:
        log_file_path = os.path.join(OUTPUT_INTERMEDIATE_DIR, "logs", f"{backbone_pdb_path.stem}.log")
        with open(log_file_path, 'a') as f:
            f.write(f"\n--- PIPELINE FAILED ---\n{e}\n")
        return f"Failed to process {backbone_pdb_path.name}: Check logs for details."

def main():
    """메인 실행 함수."""
    # 1. 백본 생성 단계
    run_pipeline_step(["python", "generate_backbone_only_pdb.py"], "Backbone Generation")

    # 2. Side Chain 추가 및 Relax 단계 준비
    print("\n--- Starting Side Chain Addition and Relaxation Pipeline ---")
    os.makedirs(OUTPUT_COMPLETE_PDB_DIR, exist_ok=True)
    os.makedirs(OUTPUT_INTERMEDIATE_DIR, exist_ok=True)
    
    # 에너지 로그 파일 헤더 작성
    if not os.path.exists(ENERGY_LOG_FILE):
        with open(ENERGY_LOG_FILE, 'w') as f:
            f.write("PDB_ID,Energy_Before_Relax,Energy_After_Relax\n")
            
    # 데이터셋에서 서열 정보 로드 및 매핑 생성
    print(f"Loading sequence data from {DATASET_PATH}...")
    try:
        dataset = pd.read_pickle(DATASET_PATH)
        # 데이터셋 형식에 따라 'pdb_id'와 'sequence' 컬럼명을 맞춰주세요.
        # 예: 'pdb' -> 'pdb_id', '1-letter' -> 'sequence'
        sequence_map = {item['pdb_id']: item['sequence_1letter'] for item in dataset}
        print(f"Successfully loaded and mapped {len(sequence_map)} sequences.")
    except Exception as e:
        print(f"Error loading or parsing dataset file: {e}")
        print("Please ensure the dataset is a pickle file containing a list of dicts with 'pdb_id' and 'sequence_1letter' keys.")
        return

    print(f"Input directory for this stage: {os.path.abspath(INPUT_PDB_DIR)}")
    print(f"Final output directory: {os.path.abspath(OUTPUT_COMPLETE_PDB_DIR)}")
    print(f"Energy log file: {os.path.abspath(ENERGY_LOG_FILE)}")
    print(f"Using a maximum of {MAX_WORKERS} processes.")

    # 처리할 PDB 파일과 서열 매핑
    pdb_files = list(Path(INPUT_PDB_DIR).glob("*.pdb"))
    if not pdb_files:
        print(f"No backbone PDB files found in {INPUT_PDB_DIR}. Exiting side chain addition stage.")
        return
        
    tasks = []
    for pdb_path in pdb_files:
        # 파일 이름에서 원본 PDB ID 추출 (e.g., "1a2b_sample_001" -> "1a2b")
        original_pdb_id = pdb_path.stem.split('_sample_')[0]
        if original_pdb_id in sequence_map:
            tasks.append((pdb_path, sequence_map[original_pdb_id]))
        else:
            print(f"Warning: Could not find sequence for {pdb_path.name}. Skipping.")

    if not tasks:
        print("No tasks to process after mapping sequences. Exiting.")
        return
        
    print(f"Found {len(tasks)} backbone PDB files with matching sequences to process.")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(process_single_pdb, tasks), total=len(tasks), desc="Adding Side Chains & Relaxing"))

    # 결과 요약
    success_count = sum(1 for r in results if "Successfully" in r)
    skipped_count = sum(1 for r in results if "Skipped" in r)
    failed_count = sum(1 for r in results if "Failed" in r)

    print("\n--- Side Chain Addition and Relaxation Finished ---")
    print(f"Total files: {len(tasks)}")
    print(f" - Successful: {success_count}")
    print(f" - Skipped: {skipped_count}")
    print(f" - Failed: {failed_count}")
    
    if failed_count > 0:
        print("\nFailed files:")
        for r in results:
            if "Failed" in r:
                print(f" - {r}")
        print(f"\nCheck logs in '{os.path.abspath(os.path.join(OUTPUT_INTERMEDIATE_DIR, 'logs'))}' for details.")


if __name__ == "__main__":
    main()
