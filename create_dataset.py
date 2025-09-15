"""
파일 이름: create_dataset.py
설명: Bio.PDB 기반으로 백본(N, CA, C)만을 사용해 데이터셋을 생성합니다.
      - 사이드체인 피처는 계산하지 않습니다.
      - 백본 각도(φ/ψ/ω), 결합각, 결합거리, 백본 좌표(N,CA,C)만 저장합니다.
"""
import numpy as np
import os
import torch
import esm
import pickle
import sys
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from Bio.PDB import PDBParser, is_aa
from sklearn.model_selection import train_test_split
import pandas as pd

# --- 1. 설정 임포트 ---
try:
    import create_dataset_config as config
except ImportError:
    print("❌ 오류: 'create_dataset_config.py' 파일을 찾을 수 없습니다.", file=sys.stderr)
    sys.exit(1)


# --- 2. 기하 유틸리티 (NumPy 기반) ---
def _unit(v):
    """벡터를 단위 벡터로 변환합니다."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

def _angle(a, b, c):
    """세 점 a, b, c 사이의 각도(∠abc)를 계산합니다."""
    v1 = _unit(a - b)
    v2 = _unit(c - b)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.arccos(dot))

def _dihedral(p1, p2, p3, p4):
    """네 점 p1, p2, p3, p4 사이의 비틀림각(dihedral)을 계산합니다."""
    b0 = p2 - p1
    b1 = p3 - p2
    b2 = p4 - p3
    b1_u = _unit(b1)
    v = b0 - np.dot(b0, b1_u) * b1_u
    w = b2 - np.dot(b2, b1_u) * b1_u
    x = np.dot(_unit(v), _unit(w))
    y = np.dot(np.cross(b1_u, _unit(v)), _unit(w))
    return float(np.arctan2(y, x))


# --- 3. 백본 특징 계산 클래스 ---
class BackboneFeatureCalculator:
    """Bio.PDB를 사용하여 PDB 파일로부터 백본 관련 피처만 계산하는 클래스."""
    def __init__(self):
        self.parser = PDBParser(QUIET=True)

    def _pick_chain(self, structure, pdb_path):
        """PDB 구조에서 가장 긴 아미노산 체인을 선택합니다."""
        models = list(structure.get_models())
        if not models:
            tqdm.write(f"  - ⚠️ {os.path.basename(pdb_path)} 파일에서 모델을 찾을 수 없어 건너뜁니다.")
            return None
        model = models[0] # ID에 상관없이 첫 번째 모델을 사용

        chains = list(model.get_chains())
        if not chains: return None

        best_chain, best_len = None, -1
        for ch in chains:
            residues = [r for r in ch.get_residues() if is_aa(r, standard=False)]
            if len(residues) > best_len:
                best_len = len(residues)
                best_chain = ch
        return best_chain

    def calculate_sequence_and_features(self, pdb_path: str):
        """PDB 파일로부터 서열과 백본 피처를 계산합니다."""
        try:
            structure = self.parser.get_structure(Path(pdb_path).stem, pdb_path)
        except Exception as e:
            tqdm.write(f"  - ⚠️ Bio.PDB 파싱 오류 {os.path.basename(pdb_path)}: {e}")
            return None, None

        chain = self._pick_chain(structure, pdb_path)
        if chain is None: return None, None

        # N, CA, C 백본 원자가 모두 존재하는 아미노산 잔기만 필터링합니다.
        residues = []
        for r in chain.get_residues():
            if is_aa(r, standard=False) and 'N' in r and 'CA' in r and 'C' in r:
                residues.append(r)
        
        L = len(residues)
        if L == 0: return None, None

        # --- 피처 계산 ---
        seq3 = [r.get_resname().strip() for r in residues]
        n = np.stack([r['N'].coord.astype(np.float32) for r in residues])
        ca = np.stack([r['CA'].coord.astype(np.float32) for r in residues])
        c = np.stack([r['C'].coord.astype(np.float32) for r in residues])

        # 1. 백본 좌표 (L, 3, 3)
        backbone_coords = np.stack([n, ca, c], axis=1)

        # 2. 결합 거리 (L, 3): N-CA, CA-C, C-N(+1)
        bb_dist = np.full((L, 3), np.nan, dtype=np.float32)
        bb_dist[:, 0] = np.linalg.norm(n - ca, axis=1)
        bb_dist[:, 1] = np.linalg.norm(ca - c, axis=1)
        bb_dist[:-1, 2] = np.linalg.norm(c[:-1] - n[1:], axis=1)

        # 3. 결합 각 (L, 3): ∠N-CA-C, ∠CA-C-N(+1), ∠C(-1)-N-CA
        bb_ang = np.full((L, 3), np.nan, dtype=np.float32)
        for i in range(L):
            bb_ang[i, 0] = _angle(n[i], ca[i], c[i])
            if i < L - 1: bb_ang[i, 1] = _angle(ca[i], c[i], n[i+1])
            if i > 0: bb_ang[i, 2] = _angle(c[i-1], n[i], ca[i])

        # 4. Torsions (L, 3): φ, ψ, ω
        bb_tor = np.full((L, 3), np.nan, dtype=np.float32)
        for i in range(L):
            if i > 0: bb_tor[i, 0] = _dihedral(c[i-1], n[i], ca[i], c[i])
            if i < L - 1: bb_tor[i, 1] = _dihedral(n[i], ca[i], c[i], n[i+1])
            if i > 0: bb_tor[i, 2] = _dihedral(ca[i-1], c[i-1], n[i], ca[i])

        # 최종 피처 딕셔너리 생성
        features_dict = {}
        if config.CALCULATE_BB_TORSION_ANGLES: features_dict["bb_torsions"] = bb_tor
        if config.CALCULATE_BB_BOND_ANGLES: features_dict["bb_angles"] = bb_ang
        if config.CALCULATE_BB_BOND_DISTANCES: features_dict["bb_distances"] = bb_dist
        if config.CALCULATE_BACKBONE_COORDS: features_dict["backbone_coords"] = backbone_coords
        
        return seq3, features_dict

# --- 4. ESM 모델 로드 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

esm_model, alphabet = None, None
if config.CALCULATE_ESM_EMBEDDING:
    print(f"Loading ESM-2 model ({config.ESM_MODEL_NAME})...")
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet(config.ESM_MODEL_NAME)
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval().to(device)
    if str(device) == 'cuda': esm_model.half()
    print("ESM-2 model loaded.")

# --- 5. 헬퍼 함수 ---
ALL_AMINO_ACIDS = sorted(list(config.CANONICAL_AMINO_ACIDS))
AA_MAP_20D = {aa: i for i, aa in enumerate(ALL_AMINO_ACIDS)}
AA3_TO_1 = {
    "ALA":"A", "CYS":"C", "ASP":"D", "GLU":"E", "PHE":"F", "GLY":"G", "HIS":"H",
    "ILE":"I", "LYS":"K", "LEU":"L", "MET":"M", "ASN":"N", "PRO":"P", "GLN":"Q",
    "ARG":"R", "SER":"S", "THR":"T", "VAL":"V", "TRP":"W", "TYR":"Y"
}

def encode_sequence(seq_list, max_len, aa_map):
    encoding = np.zeros((max_len, len(aa_map)), dtype=np.float32)
    for i, aa in enumerate(seq_list):
        if i >= max_len: break
        if aa in aa_map: encoding[i, aa_map[aa]] = 1.0
    return encoding

def pad_array(arr, target_len, pad_value=np.nan):
    current_len = arr.shape[0]
    if current_len >= target_len: return arr[:target_len]
    pad_shape = (target_len - current_len,) + arr.shape[1:]
    pad = np.full(pad_shape, pad_value, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)

def get_esm_embedding(batch_samples, layers):
    _batch_labels, batch_strs, batch_tokens = batch_converter(batch_samples)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[layers], return_contacts=False)
    token_representations = results["representations"][layers]
    embedding_list = []
    for i, (_, seq_str) in enumerate(batch_samples):
        seq_len = len(seq_str)
        embedding_list.append(token_representations[i, 1 : seq_len + 1, :].cpu())
    return embedding_list

# --- 6. 데이터셋 생성 함수 ---
def create_dataset(pdb_root_dir, feature_calculator):
    dataset = {'pdb_file': [], 'sequences_3letter': [], 'sequence_1letter': [], 'sequence_encoded': []}
    # 설정에 따라 데이터셋 딕셔너리 키 초기화
    if config.CALCULATE_BB_TORSION_ANGLES: dataset["bb_torsions"] = []
    if config.CALCULATE_BB_BOND_ANGLES: dataset["bb_angles"] = []
    if config.CALCULATE_BB_BOND_DISTANCES: dataset["bb_distances"] = []
    if config.CALCULATE_SC_TORSION_ANGLES: dataset["sc_torsions"] = []
    if config.CALCULATE_BACKBONE_COORDS: dataset["backbone_coords"] = []
    if config.CALCULATE_ESM_EMBEDDING: dataset["esm_embedding"] = []
    
    pdb_directory = Path(pdb_root_dir)
    if not pdb_directory.exists():
        print(f"❌ PDB 디렉토리를 찾을 수 없습니다: {pdb_directory}")
        return dataset

    pdb_files = sorted([f for f in pdb_directory.glob("*.pdb")])
    skipped_files_info = {}

    pbar = tqdm(range(0, len(pdb_files), config.BATCH_SIZE), desc=f"Processing {pdb_directory.name}")
    for start in pbar:
        batch_pdb = pdb_files[start : start + config.BATCH_SIZE]
        batch_meta_info = []

        for pdb_file in batch_pdb:
            seq_list, features_dict = feature_calculator.calculate_sequence_and_features(str(pdb_file))
            
            if seq_list is None:
                skipped_files_info[pdb_file] = "유효 백본 잔기 없음/처리 오류"
                continue
            if not (0 < len(seq_list) <= config.MAX_LEN):
                skipped_files_info[pdb_file] = f"길이 필터링됨 (길이: {len(seq_list)})"
                continue

            batch_meta_info.append({
                "pdb_file": pdb_file, "seq_list": seq_list, "features": features_dict
            })

        if not batch_meta_info: continue
            
        if config.CALCULATE_ESM_EMBEDDING:
            try:
                batch_samples_for_esm = [
                    (info["pdb_file"].name, "".join(AA3_TO_1.get(aa, "X") for aa in info["seq_list"]))
                    for info in batch_meta_info
                ]
                batch_embeddings = get_esm_embedding(batch_samples_for_esm, config.ESM_REPR_LAYER)
                for info, emb in zip(batch_meta_info, batch_embeddings):
                    info["esm_embedding"] = emb
            except RuntimeError as e:
                print(f"\nESM 임베딩 오류 (배치 건너뜀): {e}", file=sys.stderr)
                for info in batch_meta_info: skipped_files_info[info['pdb_file']] = "ESM 처리 오류"
                continue

        for info in batch_meta_info:
            dataset["pdb_file"].append(str(info["pdb_file"]))
            dataset["sequences_3letter"].append(info["seq_list"])
            
            seq_1letter = "".join(AA3_TO_1.get(aa, "X") for aa in info["seq_list"])
            dataset["sequence_1letter"].append(seq_1letter)
            dataset["sequence_encoded"].append(encode_sequence(info["seq_list"], config.MAX_LEN, AA_MAP_20D))
            
            for key, arr in info["features"].items():
                dataset[key].append(pad_array(arr, config.MAX_LEN))
            
            if config.CALCULATE_ESM_EMBEDDING and "esm_embedding" in info:
                dataset["esm_embedding"].append(pad_array(info["esm_embedding"].numpy().astype(np.float16), config.MAX_LEN))

    if skipped_files_info:
        with open("create_dataset_error.log", "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"--- Log at {timestamp} for [{pdb_directory.name}] ---\n")
            for fname, reason in sorted(skipped_files_info.items()):
                f.write(f"  - 파일: {fname}, 원인: {reason}\n")
            f.write("-" * 50 + "\n\n")

    return dataset


def split_and_save_dataset(full_dataset, output_dir, filename_prefix):
    """데이터셋을 길이 기준으로 계층화하여 train/validation으로 분할하고 저장합니다."""
    if not any(full_dataset.values()):
        print("⚠️ 분할할 데이터가 없습니다. 데이터셋이 비어있습니다.")
        return None, None

    # 데이터셋의 모든 값을 DataFrame으로 변환하여 다루기 쉽게 만듭니다.
    df = pd.DataFrame(full_dataset)
    # 계층화의 기준이 될 각 펩타이드의 실제 길이를 계산하여 새 열로 추가합니다.
    df['length'] = df['sequences_3letter'].apply(len)
    
    # 각 길이별 샘플 수를 계산합니다.
    length_counts = df['length'].value_counts()
    
    # 샘플이 1개뿐인 '외톨이' 길이들을 찾아냅니다.
    lone_sample_lengths = length_counts[length_counts < 2].index.tolist()
    
    if lone_sample_lengths:
        print(f"[*] 정보: 샘플이 1개뿐인 길이들 {sorted(lone_sample_lengths)} 은(는) 모두 훈련 데이터셋에 포함됩니다.")
        # 외톨이 샘플과 분할 가능한 샘플로 데이터프레임을 나눕니다.
        df_lone = df[df['length'].isin(lone_sample_lengths)]
        df_splittable = df[~df['length'].isin(lone_sample_lengths)]
    else:
        # 외톨이 샘플이 없으면 전체를 분할 대상으로 합니다.
        df_lone = pd.DataFrame()
        df_splittable = df

    # 분할 가능한 데이터가 있을 경우에만 계층 분할을 수행합니다.
    if not df_splittable.empty:
        train_df_main, val_df = train_test_split(
            df_splittable,
            test_size=config.VALIDATION_SPLIT_RATIO,
            stratify=df_splittable['length'],  # 길이 분포를 동일하게 유지
            random_state=42    # 재현성을 위한 시드 고정
        )
    else:
        # 분할 가능한 데이터가 하나도 없는 경우 (모든 샘플이 외톨이인 경우)
        train_df_main, val_df = pd.DataFrame(), pd.DataFrame()

    # 외톨이 샘플들을 훈련 데이터셋에 합칩니다.
    train_df = pd.concat([train_df_main, df_lone], ignore_index=True)
    
    # 더 이상 필요없는 'length' 열을 삭제합니다.
    train_df = train_df.drop(columns=['length'])
    val_df = val_df.drop(columns=['length'], errors='ignore') # val_df가 비어있을 수 있으므로 오류 무시

    # DataFrame을 다시 딕셔너리 형태로 변환합니다.
    train_dataset = {col: train_df[col].tolist() for col in train_df.columns}
    val_dataset = {col: val_df[col].tolist() for col in val_df.columns}

    # 분할된 데이터셋을 각각의 .pkl 파일로 저장합니다.
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / f"{filename_prefix}_train.pkl"
    with open(train_path, 'wb') as f:
        pickle.dump(train_dataset, f)
    print(f"==> ✅ Train dataset saved to {train_path} ({len(train_df)} samples)")

    val_path = None
    if not val_df.empty:
        val_path = output_dir / f"{filename_prefix}_val.pkl"
        with open(val_path, 'wb') as f:
            pickle.dump(val_dataset, f)
        print(f"==> ✅ Validation dataset saved to {val_path} ({len(val_df)} samples)")
        
    return str(train_path), str(val_path) if val_path else None

# --- 7. 데이터셋 검증 함수 ---
def verify_dataset_entry(dataset_path: str):
    print("\n--- 데이터셋 검증 시작 ---")
    if not Path(dataset_path).exists():
        print(f"❌ 검증할 파일을 찾을 수 없습니다: {dataset_path}")
        return

    print(f"🔍 '{dataset_path}' 파일의 첫 번째 항목을 분석합니다...")
    try:
        with open(dataset_path, 'rb') as f: data = pickle.load(f)
    except Exception as e:
        print(f"❌ 파일을 불러오는 중 오류가 발생했습니다: {e}")
        return

    if not isinstance(data, dict) or not all(isinstance(v, list) for v in data.values()):
        print("❌ 데이터 형식이 예상과 다릅니다. (key: list of values 형태가 아님)")
        return

    first_entry_data = {key: (value_list[0] if value_list else "N/A") for key, value_list in data.items()}
    
    print("\n[첫 번째 샘플 데이터]")
    for key, value in first_entry_data.items():
        if isinstance(value, np.ndarray): print(f"- {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, list): print(f"- {key}: list of {len(value)} items")
        else: print(f"- {key}: {value}")
    print("\n--- 데이터셋 검증 완료 ---\n")

# --- 8. 메인 실행 블록 ---
if __name__ == "__main__":
    feature_calculator = BackboneFeatureCalculator()

    print(f"\n🚀 Starting dataset creation in '{config.DATASET_MODE}' mode.")
    if config.DATASET_MODE == 'train_test':
        print("\n[1/2] 전체 학습/검증용 PDB로부터 피처를 추출합니다...")
        # 우선 모든 데이터를 하나의 데이터셋으로 불러옵니다.
        full_train_val_dataset = create_dataset(config.PDB_TRAIN_DIR, feature_calculator)
        
        print("\n[2/2] 데이터셋을 Train/Validation으로 분할하여 저장합니다...")
        train_pkl_path, val_pkl_path = split_and_save_dataset(
            full_train_val_dataset,
            config.DATASET_OUTPUT_DIR,
            config.OUTPUT_FILENAME_PREFIX
        )

        print("\n[+] TEST 데이터셋 생성을 시작합니다...")
        test_dataset = create_dataset(config.PDB_TEST_DIR, feature_calculator)

        test_pkl_path = None
        if test_dataset and any(len(v) > 0 for v in test_dataset.values()):
            test_path = config.DATASET_OUTPUT_DIR / f"{config.OUTPUT_FILENAME_PREFIX}_test.pkl"
            config.DATASET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(test_path, 'wb') as f: pickle.dump(test_dataset, f)
            print(f"==> ✅ Test dataset saved to {test_path}")
            test_pkl_path = str(test_path)
        
        if test_pkl_path: verify_dataset_entry(test_pkl_path)
            
    elif config.DATASET_MODE == '10_fold':
        print("10-fold 모드는 현재 이 스크립트에서 지원되지 않습니다.")
    else:
        print(f"❌ Unknown mode in config: {config.DATASET_MODE}")
