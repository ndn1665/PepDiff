# 파일 이름: evaluation.py

import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# --- 1. 새로운 설정 파일 임포트 ---
from evaluation_config import Config

# --- 2. 분석 함수 임포트 (graph_generator.py에서 명시적으로 가져옴) ---
try:
    from graph_generator import (
        precalculate_all_rmsds,
        rmsd_by_samples,
        rmsd_by_peptide_length,
        plot_success_rate_vs_ensemble,
        rmsd_by_ensemble_and_length
    )
except ImportError:
    print("❌ 오류: 'graph_generator.py' 파일을 찾을 수 없거나, 필요한 함수가 없습니다.")
    exit(1)


def load_dataset(data_path: Path):
    """지정된 경로에서 피클 데이터셋을 로드합니다."""
    if not data_path.exists():
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {data_path}")
    with data_path.open("rb") as f:
        return pickle.load(f)

def main():
    """분석 스크립트의 메인 실행 함수."""
    
    config = Config()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"결과는 다음 폴더에 저장됩니다: {config.OUTPUT_DIR.resolve()}")

    try:
        test_dataset = load_dataset(config.DATASET_FILE)
        print(f"데이터셋을 성공적으로 불러왔습니다: {config.DATASET_FILE}")
    except FileNotFoundError as e:
        print(f"[오류] {e}")
        exit(1)

    # --- [수정] RMSD 캐시 파일 확인 및 로드/생성 로직 ---
    cache_save_path = config.OUTPUT_DIR / "rmsd_cache.pkl"
    
    if cache_save_path.exists():
        print(f"✅ 기존 RMSD 캐시 파일을 발견했습니다. 파일을 로드합니다: {cache_save_path}")
        try:
            with open(cache_save_path, "rb") as f:
                all_rmsds_cache = pickle.load(f)
        except Exception as e:
            print(f"⚠️ 캐시 파일 로드 중 오류 발생: {e}. RMSD를 새로 계산합니다.")
            all_rmsds_cache = None
    else:
        all_rmsds_cache = None

    if all_rmsds_cache is None:
        # 모든 분석 전에 RMSD 값을 미리 한 번만 계산합니다.
        all_rmsds_cache = precalculate_all_rmsds(test_dataset, config)

        # 계산된 RMSD 캐시를 pkl 파일로 저장합니다.
        try:
            with open(cache_save_path, "wb") as f:
                pickle.dump(all_rmsds_cache, f)
            print(f"✅ 계산된 RMSD 값을 다음 파일에 저장했습니다: {cache_save_path}")
        except Exception as e:
            print(f"⚠️ RMSD 캐시 저장 중 오류 발생: {e}")

    # 병렬 처리를 사용하여 분석 함수들 실행 (미리 계산된 캐시 전달)
    print("\n분석을 시작합니다...")
    futures = []
    with ProcessPoolExecutor() as executor:
        futures.append(executor.submit(rmsd_by_samples, test_dataset, config, all_rmsds_cache))
        futures.append(executor.submit(rmsd_by_peptide_length, test_dataset, config, all_rmsds_cache))
        futures.append(executor.submit(plot_success_rate_vs_ensemble, test_dataset, config, all_rmsds_cache))
        futures.append(executor.submit(rmsd_by_ensemble_and_length, test_dataset, config, all_rmsds_cache))
        
        print("모든 분석 작업이 제출되었습니다. 완료될 때까지 기다려주세요...")
        
        # 각 작업의 결과를 확인하여 예외가 발생했는지 확인합니다.
        for future in futures:
            try:
                future.result()  # 작업이 끝날 때까지 기다리고, 예외가 있었다면 여기서 발생시킵니다.
            except Exception as e:
                print(f"🚨 병렬 분석 작업 중 오류가 발생했습니다: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
