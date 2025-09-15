# comparison_graph_generator.py

import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import PDBParser
from pathlib import Path
from tqdm import tqdm
import glob
import copy
from concurrent.futures import ProcessPoolExecutor

# utils 폴더에 있는 분석 유틸리티를 임포트합니다.
from utils.analysis_utils import rmsd_calculation, iqr_filter

# --- Helper Functions ---

def get_pdb_files(pdb_id: str, config: object) -> (Path, list):
    """주어진 PDB ID에 해당하는 원본 파일과 생성된 샘플 파일들의 경로를 반환합니다."""
    origin_path = config.ORIGIN_PDB_DIR / config.ORIGIN_FNAME_TPL.format(pdb_id=pdb_id)
    sample_pattern = config.SAMPLE_FNAME_TPL.format(pdb_id=pdb_id, sample_idx='*')
    # SAMPLES_PDB_DIR은 config 객체 내에서 동적으로 변경될 수 있습니다.
    sample_paths = sorted(glob.glob(str(config.SAMPLES_PDB_DIR / sample_pattern)))
    return origin_path, sample_paths

# --- Comparison Plotting Functions ---

def plot_accuracy_vs_ensemble(test_dataset: dict, config):
    """
    [그래프 1] 여러 방법론에 대해 앙상블 크기에 따른 cRMSD(정확도)를 비교합니다.
    """
    print("\n[1/2] 정확도(cRMSD) vs 앙상블 크기 분석 중...")
    fig_path = config.OUTPUT_DIR / "comparison_accuracy_vs_ensemble.png"
    all_methods_results = {}
    p = PDBParser(QUIET=True)

    # 설정 파일에 정의된 모든 방법론에 대해 반복
    for method_name, params in config.METHODS.items():
        print(f"  - 처리 중인 방법론: {method_name}")
        
        best_rmsds_for_all_pdbs = [[] for _ in range(config.ACCURACY_MAX_ENSEMBLE_SIZE)]

        for pdb_id in tqdm(test_dataset['pdb_file'], desc=f"  Calculating RMSD for {method_name}", leave=False):
            origin_path = config.ORIGIN_PDB_DIR / config.ORIGIN_FNAME_TPL.format(pdb_id=pdb_id)
            
            min_rmsd_so_far = np.inf
            for i in range(config.ACCURACY_MAX_ENSEMBLE_SIZE):
                sample_path = params["path"] / config.SAMPLE_FNAME_TPL.format(sample_idx=i, pdb_id=pdb_id)
                rms = rmsd_calculation(origin_path, sample_path, p, config)
                if rms is not None:
                    min_rmsd_so_far = min(min_rmsd_so_far, rms)
                
                if np.isfinite(min_rmsd_so_far):
                    best_rmsds_for_all_pdbs[i].append(min_rmsd_so_far)

        # 앙상블 크기별로 평균 cRMSD 계산
        mean_best_rmsds = []
        for i in range(config.ACCURACY_MAX_ENSEMBLE_SIZE):
            rmsds_at_size_i = best_rmsds_for_all_pdbs[i]
            if config.USE_IQR_FILTER:
                rmsds_at_size_i = iqr_filter(rmsds_at_size_i, config.IQR_K_VALUE)
            
            valid_rmsds = [r for r in rmsds_at_size_i if r is not None]
            if valid_rmsds:
                mean_best_rmsds.append(np.mean(valid_rmsds))
            else:
                mean_best_rmsds.append(np.nan) # 데이터가 없으면 NaN 처리
        
        all_methods_results[method_name] = mean_best_rmsds

    # --- 모든 방법론에 대한 그래프 그리기 ---
    plt.figure(figsize=(10, 8))
    for method_name, y_values in all_methods_results.items():
        x_values = range(1, len(y_values) + 1)
        params = config.METHODS[method_name]
        plt.plot(x_values, y_values,
                 label=method_name,
                 color=params['color'],
                 marker=params['marker'],
                 markersize=4, linestyle='-')

    plt.xlabel("Ensemble size", fontsize=14)
    plt.ylabel("cRMSD (Å)", fontsize=14)
    plt.title("Accuracy vs. Ensemble Size Comparison", fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[성공] 정확도 비교 그래프 저장 완료: {fig_path}")

def plot_success_rate_vs_ensemble(test_dataset: dict, config: object):
    """
    여러 방법론에 대해 앙상블 크기에 따른 성공률을 비교하는 그래프를 생성합니다.
    (ModPEP 논문 Fig 3b 방식)
    """
    print("Analyzing 'Success Rate vs. Ensemble Size' for comparison...")
    
    ensemble_sizes = range(
        config.SUCCESS_RATE_STEP,
        config.SUCCESS_MAX_ENSEMBLE_SIZE + 1,
        config.SUCCESS_RATE_STEP
    )
    total_peptides = len(test_dataset["pdb_file"])
    
    plt.figure(figsize=(10, 8))
    parser = PDBParser(QUIET=True)

    # 각 방법론에 대해 분석 실행
    for method_name, method_info in config.METHODS.items():
        print(f"  -> Processing method: {method_name}")
        
        # config 객체를 임시로 복사하여 현재 방법론의 경로를 사용하도록 설정
        temp_config = copy.deepcopy(config)
        temp_config.SAMPLES_PDB_DIR = method_info["path"]
        
        success_rates = []
        
        # RMSD 값 캐싱
        all_rmsds_cache = {}
        with ProcessPoolExecutor() as executor:
            future_to_pdb_id = {}
            for i, pdb_path_str in enumerate(test_dataset["pdb_file"]):
                pdb_id = Path(pdb_path_str).stem
                origin_path, sample_paths = get_pdb_files(pdb_id, temp_config)
                if not sample_paths:
                    all_rmsds_cache[pdb_id] = []
                    continue
                futures = [executor.submit(rmsd_calculation, origin_path, s_path, parser, temp_config) for s_path in sample_paths]
                future_to_pdb_id[pdb_id] = futures

            for pdb_id, futures in tqdm(future_to_pdb_id.items(), desc=f"    -> RMSD for {method_name}", leave=False):
                rmsds = [f.result() for f in futures if f.result() is not None]
                all_rmsds_cache[pdb_id] = sorted(rmsds)

        # 앙상블 크기별 성공률 계산
        for k in ensemble_sizes:
            successful_peptides_count = 0
            for i, pdb_path_str in enumerate(test_dataset["pdb_file"]):
                pdb_id = Path(pdb_path_str).stem
                cached_rmsds = all_rmsds_cache.get(pdb_id)
                if not cached_rmsds:
                    continue
                
                ensemble_rmsds = cached_rmsds[:k]
                
                peptide_len = len(test_dataset['sequences_1letter'][i])
                rmsd_threshold = temp_config.SUCCESS_RATE_N0 * np.log(peptide_len)
                
                if ensemble_rmsds and min(ensemble_rmsds) < rmsd_threshold:
                    successful_peptides_count += 1
            
            rate = (successful_peptides_count / total_peptides) * 100 if total_peptides > 0 else 0
            success_rates.append(rate)
            
        # 현재 방법론의 결과를 그래프에 추가
        plt.plot(
            list(ensemble_sizes), 
            success_rates, 
            marker=method_info["marker"], 
            color=method_info["color"], 
            label=method_name
        )

    # 그래프 최종 설정 및 저장
    plt.xlabel("Ensemble Size", fontsize=14)
    plt.ylabel("Success Rate (%)", fontsize=14)
    plt.title("Comparison of Success Rate vs. Ensemble Size", fontsize=16)
    plt.legend(title="Method")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(0, 101)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "comparison_success_rate_vs_ensemble.png", dpi=300)
    plt.close()
    print(" -> 'Comparison of Success Rate vs. Ensemble Size' plot saved.")