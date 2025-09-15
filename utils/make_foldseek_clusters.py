import os
import shutil
import subprocess
import glob
from collections import defaultdict
import csv
# Biopython import 제거

# --- 0. 경로 변수 설정 ---
BASE_DIR = "resolution2.5complexes/peptide/resolution2.5filtered"
WORKSPACE_DIR = os.path.join(BASE_DIR, "foldseek_workspace_without_noncanonical_amino_acid")
PDB_SOURCE_DIR = os.path.join(BASE_DIR, "without_noncanonical_amino_acid")
PDB_INPUT_DIR = os.path.join(WORKSPACE_DIR, "pdbs")
CENTROID_DIR = os.path.join(WORKSPACE_DIR, "centroids")

PEPTIDE_DB = os.path.join(WORKSPACE_DIR, "peptideDB")
# easy-cluster의 출력 prefix
CLUSTER_PREFIX = os.path.join(WORKSPACE_DIR, 'clusters')
TMP_DIR = "tmp_results"

output_csv = os.path.join(WORKSPACE_DIR, 'cluster_summary.csv') # 저장될 CSV 파일

def run_command(command):
    """주어진 명령어를 실행하고 오류가 발생하면 스크립트를 중단합니다."""
    print(f"🚀 Executing: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing command: {' '.join(command)}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        exit(1)

# --- 새로운 FASTA 파싱 함수 ---
def parse_fasta_lengths(fasta_path):
    """FASTA 파일을 읽어 각 시퀀스의 ID와 길이를 딕셔너리로 반환합니다."""
    print(f"Reading sequence lengths from {fasta_path}...")
    lengths = {}
    try:
        with open(fasta_path, 'r') as f:
            current_id = None
            current_len = 0
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # 이전 시퀀스가 있었다면 길이를 저장
                    if current_id:
                        lengths[current_id] = current_len
                    # 새로운 시퀀스 정보 초기화
                    # 헤더에서 ID만 추출 (공백이 있을 경우 첫 단어만 사용)
                    current_id = line[1:].split()[0]
                    current_len = 0
                elif current_id:
                    # 시퀀스 라인이면 길이를 더함
                    current_len += len(line)
            # 파일의 마지막 시퀀스 정보 저장
            if current_id:
                lengths[current_id] = current_len
    except FileNotFoundError:
        print(f"❌ Error: FASTA 파일을 찾을 수 없습니다: {fasta_path}")
        return None
    
    print(f"✅ Found lengths for {len(lengths)} centroids.")
    return lengths


def main():
    """메인 워크플로우를 실행하는 함수"""
    print("🚀 Starting the Foldseek clustering workflow with Python...")

    # --- 1. 디렉토리 설정 및 PDB 파일 복사 ---
    print("\n--- 1. Setting up directories and copying PDB files ---")
    os.makedirs(PDB_INPUT_DIR, exist_ok=True)
    os.makedirs(CENTROID_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    
    print(f"Copying PDB files from {PDB_SOURCE_DIR} to {PDB_INPUT_DIR}...")
    # ... (복사 로직은 동일) ...
    pdb_files_to_copy = glob.glob(os.path.join(PDB_SOURCE_DIR, '*.pdb'))
    for pdb_file in pdb_files_to_copy:
        shutil.copy(pdb_file, PDB_INPUT_DIR)
    print(f"✅ {len(pdb_files_to_copy)} PDB files copied.")

    # --- 2. Foldseek 데이터베이스 생성 (easy-cluster는 DB가 아닌 폴더를 직접 받을 수도 있음) ---
    # 여기서는 폴더를 직접 사용하도록 변경하여 createdb 단계를 생략
    
    # --- 3. easy-cluster 워크플로우로 클러스터링 수행 ---
    print("\n--- 3. Clustering with easy-cluster workflow ---")
    run_command([
    'foldseek', 'easy-cluster',
    PDB_INPUT_DIR,
    CLUSTER_PREFIX, # 출력 prefix. 'clusters_cluster.tsv', 'clusters_rep_seq.fasta' 등이 생성됨
    TMP_DIR,
    '-e','0.0001',
    '--alignment-type', '1',
    '--tmscore-threshold', '0.7'
    ])
    print("✅ Easy-clustering complete.")

    # --- 4. FASTA 파일에서 Centroid 서열 길이 읽기 ---
    fasta_file = f"{CLUSTER_PREFIX}_rep_seq.fasta"
    centroid_lengths = parse_fasta_lengths(fasta_file)
    if centroid_lengths is None:
        exit(1)

    # --- 5. clusters_cluster.tsv 파일을 읽고 CSV로 저장 ---
    clu_tsv = f"{CLUSTER_PREFIX}_cluster.tsv"
    clusters = defaultdict(list)
    try:
        with open(clu_tsv, 'r') as f:
            for line in f:
                centroid, member = line.strip().split('\t')
                clusters[centroid].append(member)
    except FileNotFoundError:
        print(f"❌ Error: Cluster TSV 파일을 찾을 수 없습니다: {clu_tsv}")
        exit(1)

    print(f"\n--- 클러스터 정보를 {output_csv} 파일로 저장합니다 ---")
    try:
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Centroid', 'Sequence_Length', 'Member_Count', 'Members'])

            for centroid, members in sorted(clusters.items()):
                # 미리 읽어둔 딕셔너리에서 길이 조회
                seq_len = centroid_lengths.get(centroid, "N/A")
                
                row_to_write = [centroid, seq_len, len(members)] + members
                writer.writerow(row_to_write)
                
        print(f"✅ 성공적으로 파일을 저장했습니다.")
    except Exception as e:
        print(f"❌ Error: CSV 파일 저장 중 오류가 발생했습니다: {e}")

    # --- 6. Centroid PDB 파일 추출 ---
    print(f"\n--- 6. Centroid PDB 파일을 {CENTROID_DIR}에 저장합니다 ---")
    centroid_names = clusters.keys()
    copied_count = 0
    for name in centroid_names:
        source_pdb_path = os.path.join(PDB_INPUT_DIR, f"{name}.pdb")
        if os.path.exists(source_pdb_path):
            shutil.copy(source_pdb_path, CENTROID_DIR)
            copied_count += 1
        else:
            print(f"⚠️ Warning: 원본 PDB 파일을 찾을 수 없습니다: {source_pdb_path}")
            
    print(f"✅ {copied_count}개의 Centroid 파일을 성공적으로 복사했습니다.")
    
    # --- 완료 ---
    print("\n🎉 All tasks completed successfully! 🎉\n")

if __name__ == "__main__":
    main()