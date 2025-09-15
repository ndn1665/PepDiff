# run_comparison_analysis.py

import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# --- Step 1: 새로운 비교용 설정 파일 임포트 ---
from comparison_evaluation_config import Config

# --- Step 2: 새로운 비교 그래프 생성 함수 임포트 ---
from comparison_graph_generator import (
    plot_accuracy_vs_ensemble,
    plot_success_rate_vs_ensemble,
)


def load_dataset(data_path: Path):
    """지정된 경로에서 피클된 데이터셋을 로드합니다."""
    if not data_path.exists():
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {data_path}")
    with data_path.open("rb") as f:
        return pickle.load(f)

def main():
    """
    여러 펩타이드 구조 생성 방법론의 성능 비교 분석을 실행하는 메인 스크립트입니다.
    설정 파일을 로드하고, 정의된 각 방법론에 대한 분석을 수행하여
    하나의 통합된 그래프로 결과를 저장합니다.
    """
    # 설정 클래스 초기화
    config = Config()

    # 결과 저장 디렉토리 생성
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"결과가 다음 디렉토리에 저장됩니다: {config.OUTPUT_DIR.resolve()}")

    # 데이터셋 로드
    try:
        test_dataset = load_dataset(config.DATASET_FILE)
        print(f"데이터셋을 성공적으로 로드했습니다: {config.DATASET_FILE}")
    except FileNotFoundError as e:
        print(f"[오류] {e}")
        exit(1)

    # --- 분석 함수 실행 (병렬 처리) ---
    with ProcessPoolExecutor() as executor:
        print("\n분석을 시작합니다...")
        # 1. 정확도(cRMSD) vs 앙상블 크기 그래프 생성
        executor.submit(plot_accuracy_vs_ensemble, test_dataset, config)
        
        # 2. 성공률 vs 앙상블 크기 그래프 생성
        executor.submit(plot_success_rate_vs_ensemble, test_dataset, config)
    
    print("\n모든 분석이 완료되었습니다.")


if __name__ == "__main__":
    main()