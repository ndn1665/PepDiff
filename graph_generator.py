# graph_generator.py

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import PDBParser, Superimposer
from tqdm import tqdm
import glob
from concurrent.futures import ProcessPoolExecutor

# utils 폴더에 있는 분석 유틸리티를 임포트합니다.
from utils.analysis_utils import rmsd_calculation, iqr_bounds, three_to_one, _extract_coords


def precalculate_all_rmsds(test_dataset: dict, config: object) -> dict:
    """
    모든 PDB의 모든 샘플에 대한 RMSD 값을 미리 계산하고 캐시합니다.
    이 단계는 계산 비용이 매우 높습니다.
    """
    print("모든 RMSD 값을 미리 계산합니다 (시간이 오래 걸릴 수 있습니다)...")
    all_rmsds_cache = {}
    parser = PDBParser(QUIET=True)

    # --- DEBUGGING: Add a flag to print details for the first PDB ID only ---
    is_first_pdb = True

    for pdb_path_str in tqdm(test_dataset["pdb_file"], desc="RMSD 사전 계산"):
        pdb_id = Path(pdb_path_str).stem
        origin_path, sample_paths = get_pdb_files(pdb_id, config)
        
        # --- More detailed debugging for the first item ---
        if is_first_pdb:
            print(f"\n--- [DEBUG] 첫 번째 PDB ID 상세 분석: {pdb_id} ---")
            print(f"1. 원본 PDB 경로 확인:")
            print(f"   - 경로: {origin_path.resolve()}")
            print(f"   - 존재 여부: {'✅ 존재함' if origin_path.exists() else '❌ 없음'}")
            
            print(f"2. 생성된 샘플 PDB 경로 확인:")
            sample_pattern = config.SAMPLE_FNAME_TPL.format(pdb_id=pdb_id, sample_idx='*')
            full_pattern = str(config.SAMPLES_PDB_DIR / sample_pattern)
            print(f"   - 검색 패턴: {full_pattern}")
            print(f"   - 찾은 파일 개수: {len(sample_paths)}")

        if not origin_path.exists() or not sample_paths:
            all_rmsds_cache[pdb_id] = []
            if is_first_pdb:
                 print("   - 원본 또는 샘플 파일이 없어 RMSD 계산을 건너뜁니다.")
                 print("--- [DEBUG] 분석 종료 ---")
            is_first_pdb = False
            continue

        rmsds = []
        for i, s_path in enumerate(sample_paths):
            s_path = Path(s_path)
            rms = rmsd_calculation(origin_path, s_path, parser, config)
            if rms is not None:
                rmsds.append(rms)
            
            # --- First sample detailed breakdown ---
            if is_first_pdb and i == 0:
                print(f"3. 첫 번째 샘플 RMSD 계산 과정 분석:")
                print(f"   - 샘플 경로: {s_path.resolve()}")
                try:
                    native = parser.get_structure("origin_debug", origin_path)
                    model = parser.get_structure("sample_debug", s_path)
                    
                    # --- [DEBUG] 추가: 실제 잔기 이름 확인 ---
                    model_residues_raw = [r.resname for r in model[0].get_residues()]
                    print(f"   - 생성된 PDB의 실제 잔기 목록 ({len(model_residues_raw)}개): {model_residues_raw[:15]}...")
                    
                    # --- [DEBUG] 수정: 필터링 로직을 실제 align 함수와 동일하게 변경 ---
                    native_residue_count = len(list(native[0].get_residues()))
                    model_residue_count = len(list(model[0].get_residues()))
                    
                    print(f"   - 원본 잔기 개수 (서열 길이): {native_residue_count}")
                    print(f"   - 생성된 잔기 개수 (서열 길이): {model_residue_count}")
                    
                    n_xyz = _extract_coords(native, config.ATOM_TYPES)
                    m_xyz = _extract_coords(model, config.ATOM_TYPES)
                    print(f"   - 원본 좌표 Shape: {n_xyz.shape}")
                    print(f"   - 생성된 좌표 Shape: {m_xyz.shape}")
                    
                    if n_xyz.shape != m_xyz.shape:
                         print("   - ❌ 진단: 좌표 Shape 불일치로 RMSD 계산 실패.")
                    elif native_residue_count != model_residue_count:
                         print("   - ❌ 진단: 서열 길이 불일치로 RMSD 계산 실패.")
                    else:
                         print(f"   - ✅ 진단: 조건 만족. 계산된 RMSD = {rms}")

                except Exception as e:
                    print(f"   - ❌ RMSD 계산 중 예외 발생: {e}")
                print("--- [DEBUG] 분석 종료 ---")

        # 정렬하지 않습니다. (샘플 순서 보존)
        all_rmsds_cache[pdb_id] = rmsds
        is_first_pdb = False # Only debug the first one
    
    print("RMSD 사전 계산 완료.")
    return all_rmsds_cache


# --- Helper Functions ---

def get_pdb_files(pdb_id: str, config: object) -> (Path, list):
    """주어진 PDB ID에 해당하는 원본 파일과 생성된 샘플 파일들의 경로를 반환합니다."""
    origin_path = config.ORIGIN_PDB_DIR / config.ORIGIN_FNAME_TPL.format(pdb_id=pdb_id)
    
    # 샘플 파일 이름 형식에 맞게 와일드카드(*)를 사용하여 모든 샘플을 찾습니다.
    sample_pattern = config.SAMPLE_FNAME_TPL.format(pdb_id=pdb_id, sample_idx='*')
    sample_paths = sorted(glob.glob(str(config.SAMPLES_PDB_DIR / sample_pattern)))
    
    return origin_path, sample_paths

# --- Analysis & Plotting Functions ---

def rmsd_by_samples(test_dataset: dict, config, all_rmsds_cache: dict):
    """
    Analyzes how the mean of the best RMSD changes with the number of generated samples.
    [MODIFIED] Uses a pre-calculated RMSD cache.
    """
    print("\n[1/4] '샘플별 RMSD' 분석 실행 중...")
    log_path = config.OUTPUT_DIR / "outliers_rmsd_by_samples.txt"
    fig_path = config.OUTPUT_DIR / "mean_min_rmsd_vs_samples.png"
    
    all_rmsds_per_pdb = all_rmsds_cache  # 캐시 사용
    
    mean_min_rmsd = {}
    with open(log_path, "w", encoding="utf-8") as log:
        if not config.USE_IQR_FILTER:
            log.write("IQR outlier filtering was disabled for this run.\n")
            
        for n in tqdm(range(config.MAX_SAMPLES), desc="[1/4] 샘플별 평균 계산"):
            best_per_pdb = []
            for pdb_id, rmsds in all_rmsds_per_pdb.items():
                if len(rmsds) > n:
                    best_per_pdb.append((pdb_id, min(rmsds[:n+1])))
            
            if not best_per_pdb: continue
            
            if config.USE_IQR_FILTER:
                (_, _), mask = iqr_bounds([v for _, v in best_per_pdb], k=config.IQR_K_VALUE)
                filtered_values = [v for (_, v), keep in zip(best_per_pdb, mask) if keep]
                outliers = [(pid, val) for (pid, val), keep in zip(best_per_pdb, mask) if not keep]
                for pid, val in outliers:
                    log.write(f"sample={n+1}\tPDB={pid}\tRMSD={val:.4f}\n")
            else:
                filtered_values = [v for _, v in best_per_pdb]

            if filtered_values:
                mean_min_rmsd[n+1] = float(np.mean(filtered_values))

    if mean_min_rmsd:
        x, y = zip(*sorted(mean_min_rmsd.items()))
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, marker='o')
        plt.xlabel('Number of Samples per Peptide')
        plt.ylabel('Mean of Best RMSD (Å)')
        plt.title('Mean of Best-fit RMSD vs. Number of Samples')
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
    print(f"-> [1/4] 분석 완료. 그림 저장 위치 → {fig_path}")

def rmsd_by_peptide_length(test_dataset: dict, config, all_rmsds_cache: dict):
    """
    Analyzes how the mean of the best RMSD changes with peptide length.
    [MODIFIED] Uses a pre-calculated RMSD cache.
    """
    print("\n[2/4] '펩타이드 길이별 RMSD' 분석 실행 중...")
    log_path = config.OUTPUT_DIR / "outliers_rmsd_by_length.txt"
    fig_path = config.OUTPUT_DIR / "mean_rmsd_by_length.png"

    results_by_len = {count: {} for count in config.PEPTIDE_LEN_SAMPLE_COUNTS}
    
    with open(log_path, "w", encoding="utf-8") as log:
        if not config.USE_IQR_FILTER:
            log.write("IQR outlier filtering was disabled for this run.\n")

        for length in tqdm(range(1, config.PEPTIDE_MAX_LEN + 1), desc="[2/4] 길이별 처리 중"):
            pdbs_at_len = [
                (idx, Path(pid_path).stem) for idx, pid_path in enumerate(test_dataset['pdb_file'])
                if len(test_dataset['sequence_1letter'][idx]) == length
            ]
            if not pdbs_at_len:
                continue

            best_rmsds_by_count = {count: [] for count in config.PEPTIDE_LEN_SAMPLE_COUNTS}
            for _, pdb_id in pdbs_at_len:
                cached_rmsds = all_rmsds_cache.get(pdb_id)
                if not cached_rmsds:
                    continue

                for count in sorted(config.PEPTIDE_LEN_SAMPLE_COUNTS):
                    if len(cached_rmsds) >= count:
                        best_rmsd = min(cached_rmsds[:count])
                        best_rmsds_by_count[count].append((pdb_id, best_rmsd))

            for count, best_per_pdb in best_rmsds_by_count.items():
                if not best_per_pdb: continue
                
                # [수정] IQR 필터링 전 데이터가 비어있는지 명시적으로 확인
                values_for_iqr = [v for _, v in best_per_pdb]
                if config.USE_IQR_FILTER and values_for_iqr:
                    (_, _), mask = iqr_bounds(values_for_iqr, k=config.IQR_K_VALUE)
                    filtered_values = [v for v, keep in zip(values_for_iqr, mask) if keep]
                    outliers = [(pid, v) for (pid, v), keep in zip(best_per_pdb, mask) if not keep]
                    for pid, v in outliers:
                        log.write(f"len={length}\tsamples={count}\tPDB={pid}\tRMSD={v:.4f}\n")
                else:
                    filtered_values = values_for_iqr

                if filtered_values:
                    results_by_len[count][length] = float(np.mean(filtered_values))
    
    if any(results_by_len.values()):
        plt.figure(figsize=(8, 6))
        for count, means in results_by_len.items():
            if not means: continue
            x, y = zip(*sorted(means.items()))
            plt.plot(x, y, marker='o', linestyle='-', label=f'{count} Samples')
        
        plt.xlabel('Peptide Length')
        plt.ylabel('Mean of Best RMSD (Å)')
        plt.title('Mean of Best-fit RMSD vs. Peptide Length')
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
    print(f"-> [2/4] 분석 완료. 그림 저장 위치 → {fig_path}")


def plot_success_rate_vs_ensemble(test_dataset: dict, config: object, all_rmsds_cache: dict):
    """
    ModPEP 논문(Fig 3b)과 동일한 방식으로 앙상블 크기에 따른 성공률을 계산하고 그래프를 생성합니다.
    [MODIFIED] Uses a pre-calculated RMSD cache.
    """
    print("\n[3/4] '성공률 vs 앙상블 크기' 분석 실행 중...")
    
    ensemble_sizes = range(
        config.SUCCESS_RATE_STEP, 
        config.SUCCESS_RATE_MAX_SAMPLES + 1, 
        config.SUCCESS_RATE_STEP
    )
    total_peptides = len(test_dataset["pdb_file"])
    success_rates = []

    # 앙상블 크기를 늘려가며 성공률을 계산합니다.
    for k in tqdm(ensemble_sizes, desc="[3/4] 성공률 계산 중"):
        successful_peptides_count = 0
        for i, pdb_path_str in enumerate(test_dataset["pdb_file"]):
            pdb_id = Path(pdb_path_str).stem
            
            cached_rmsds = all_rmsds_cache.get(pdb_id)
            if not cached_rmsds:
                continue

            ensemble_rmsds = cached_rmsds[:k]
            
            peptide_len = len(test_dataset['sequence_1letter'][i])
            rmsd_threshold = config.SUCCESS_RATE_N0 * np.log(peptide_len)
            
            if ensemble_rmsds and min(ensemble_rmsds) < rmsd_threshold:
                successful_peptides_count += 1
        
        rate = (successful_peptides_count / total_peptides) * 100 if total_peptides > 0 else 0
        success_rates.append(rate)

    # 그래프 생성
    plt.figure(figsize=(8, 6))
    plt.plot(list(ensemble_sizes), success_rates, marker='o', linestyle='-')
    plt.xlabel("Ensemble Size", fontsize=14)
    plt.ylabel("Success Rate (%)", fontsize=14)
    plt.title("Success Rate vs. Ensemble Size", fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(0, 101)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "success_rate_vs_ensemble.png", dpi=300)
    plt.close()
    print("-> [3/4] '성공률 vs 앙상블 크기' 그림 저장 완료.")


def rmsd_by_ensemble_and_length(test_dataset: dict, config, all_rmsds_cache: dict):
    """
    Analyzes how the mean of the best RMSD changes with ensemble size, 
    grouping results by peptide length. This plot is analogous to Fig. 4.
    [MODIFIED] Uses a pre-calculated RMSD cache.
    """
    print("\n[4/4] '앙상블/길이별 RMSD' 분석 실행 중...")
    fig_path = config.OUTPUT_DIR / "rmsd_vs_ensemble_by_length.png"

    ensemble_sizes = [1] + list(range(
        config.SUCCESS_RATE_STEP, 
        config.SUCCESS_RATE_MAX_SAMPLES + 1, 
        config.SUCCESS_RATE_STEP
    ))
    max_len = config.PEPTIDE_MAX_LEN
    group_size = config.SUCCESS_RATE_LEN_GROUP_SIZE
    
    buckets = [(lo, min(max_len, lo + group_size - 1)) for lo in range(1, max_len + 1, group_size)]
    bucket_labels = [f"{lo}-{hi}" for lo, hi in buckets]
    
    bucket_rmsd_means = {lab: [] for lab in bucket_labels}
    average_rmsd_means = []

    for ens in tqdm(ensemble_sizes, desc="[4/4] 앙상블별 처리 중"):
        best_rmsds_by_bucket = {lab: [] for lab in bucket_labels}

        for idx, pdb_path_str in enumerate(test_dataset["pdb_file"]):
            pdb_id = Path(pdb_path_str).stem
            seq_len = len(test_dataset["sequence_1letter"][idx])
            if not (1 <= seq_len <= max_len): continue
            
            bucket_idx = (seq_len - 1) // group_size
            if bucket_idx >= len(bucket_labels): continue
            bucket_lab = bucket_labels[bucket_idx]

            cached_rmsds = all_rmsds_cache.get(pdb_id)
            if not cached_rmsds or len(cached_rmsds) < ens:
                continue

            best_rmsd_for_pdb = min(cached_rmsds[:ens])

            if np.isfinite(best_rmsd_for_pdb):
                best_rmsds_by_bucket[bucket_lab].append(best_rmsd_for_pdb)

        # [수정] 각 버킷에 대해 IQR 필터링 적용
        total_rmsd_sum, total_pdbs = 0, 0
        for lab in bucket_labels:
            results = best_rmsds_by_bucket[lab]
            
            if config.USE_IQR_FILTER and results:
                (_, _), mask = iqr_bounds(results, k=config.IQR_K_VALUE)
                filtered_results = [r for r, keep in zip(results, mask) if keep]
            else:
                filtered_results = results

            if not filtered_results:
                bucket_rmsd_means[lab].append(np.nan)
                continue
            
            mean_val = np.mean(filtered_results)
            bucket_rmsd_means[lab].append(mean_val)
            
            total_rmsd_sum += sum(filtered_results)
            total_pdbs += len(filtered_results)

        if total_pdbs > 0:
            average_rmsd_means.append(total_rmsd_sum / total_pdbs)
        else:
            average_rmsd_means.append(np.nan)
            
    # --- 그래프 생성 ---
    plt.style.use('default')
    plt.figure(figsize=(8, 6))
    
    markers = itertools.cycle(['o', 's', '^', 'v', 'D', 'P'])
    linestyles = itertools.cycle(['--', '-.', ':', (0, (3, 1, 1, 1))])
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(bucket_labels)))

    for i, lab in enumerate(bucket_labels):
        plt.plot(ensemble_sizes, bucket_rmsd_means[lab],
                 marker=next(markers),
                 linestyle='--',
                 color=colors[i],
                 label=lab)

    plt.plot(ensemble_sizes, average_rmsd_means, color='black', lw=2.5, solid_capstyle='round', label='Average')

    plt.xlabel("Ensemble size", fontsize=14)
    plt.ylabel("cRMSD (Å)", fontsize=14)
    plt.title("Average Accuracies vs. Ensemble Size", fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title="Peptide Length", loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"-> [4/4] 분석 완료. 그림 저장 위치 → {fig_path}")