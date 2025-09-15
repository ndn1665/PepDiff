# 파일 이름: generate_backbone_only_pdb.py
# 설명: 학습된 Diffusion 모델과 NERF 알고리즘을 사용하여 펩타이드 서열로부터 3D 백본 구조(PDB)를 생성합니다.
#      PyRosetta 의존성을 제거하고, 예측된 각도와 결합 길이를 직접 3D 좌표로 변환합니다.

import os
# TensorFlow의 로그 레벨을 설정하여 INFO 및 WARNING 메시지를 비활성화합니다.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from torch.cuda.amp import autocast

# --- 1. 설정 및 모델/데이터 클래스 임포트 ---
import model_config as train_config
import generate_backbone_only_pdb_config as gen_config
from model import DiffusionTrainer, PeptideDataset, get_model
from utils.nerf import nerf_build_batch

# --- 2. PDB 파일 저장 유틸리티 ---
def save_coords_to_pdb(coords: np.ndarray, sequence_3_letter: list, output_path: str):
    """
    주어진 3D 좌표와 아미노산 서열 정보를 사용하여 PDB 파일을 생성합니다.
    Args:
        coords (np.ndarray): 원자 좌표 배열. shape: (num_atoms, 3)
        sequence_3_letter (list): 3-letter 아미노산 코드의 리스트.
        output_path (str): 저장될 PDB 파일의 경로.
    """
    atom_names = ["N", "CA", "C"]
    num_residues = len(sequence_3_letter)
    
    with open(output_path, "w") as f:
        atom_idx = 1
        for res_idx in range(num_residues):
            res_name = sequence_3_letter[res_idx]
            for i in range(3):  # N, CA, C
                atom_name = atom_names[i]
                coord_idx = res_idx * 3 + i
                if coord_idx >= len(coords):
                    tqdm.write(f"⚠️ 경고: 좌표 배열의 끝에 도달했습니다. {res_name} 잔기의 일부 원자가 누락될 수 있습니다.")
                    break
                
                x, y, z = coords[coord_idx]
                
                # PDB ATOM record format
                line = (
                    f"ATOM  {atom_idx:5d} {atom_name:<4s}{res_name:3s} A{res_idx+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {atom_name[0]:>2s}  \n"
                )
                f.write(line)
                atom_idx += 1
        f.write("TER\n")

# --- 3. 메인 생성 함수 ---
def generate_pdbs():
    """학습된 모델을 로드하여 NERF 기반 PDB 생성을 수행하는 메인 함수."""
    
    # --- 설정 로드 ---
    # 장치 선택 최적화: 'cuda' 요청 시 가능하면 GPU 사용, 'auto' 지원
    cfg_device = str(gen_config.GENERATION_PARAMS.get("device", "cpu")).lower()
    if cfg_device in ("cuda", "gpu", "cuda:0", "auto") and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # [최적화] CUDA 환경에서 커널 자동 튜닝 및 matmul 정밀도 향상
    if device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    model_path = Path(gen_config.PATHS["trained_model_path"])
    dataset_path = Path(gen_config.PATHS["dataset_path"])
    output_dir = Path(gen_config.PATHS["output_pdb_dir"])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generated PDBs will be saved in: {output_dir.resolve()}")

    # --- 2. 모델 로드 ---
    # 먼저 CPU에서 모델을 생성합니다.
    model = get_model(train_config)
    
    # 체크포인트 로드 (CPU 매핑)
    checkpoint_path = gen_config.GENERATION_PARAMS["model_checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        print(f"🚨 Fatal Error: Model checkpoint not found at {checkpoint_path}")
        return
        
    print(f"Loading model checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # state_dict 추출 및 'module.' 접두사 처리 (DataParallel 호환성)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') # 'module.' 접두사 제거
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    print("Model loaded successfully.")

    # --- 2a. Multi-GPU 설정 및 장치로 모델 이동 ---
    if gen_config.GENERATION_PARAMS.get("multi_gpu", False) and torch.cuda.device_count() > 1:
        print(f"🚀 Let's use {torch.cuda.device_count()} GPUs for generation!")
        model = torch.nn.DataParallel(model)
    model.to(device)
    
    model.eval()
    diffusion = DiffusionTrainer(model)

    # --- 3. 데이터셋 준비 ---
    test_dataset_path = gen_config.GENERATION_PARAMS["test_dataset_path"]
    
    # [수정] PeptideDataset 객체 생성이 누락된 부분 추가
    if not os.path.exists(test_dataset_path):
        print(f"🚨 Fatal Error: Dataset file not found at {test_dataset_path}")
        return
    dataset = PeptideDataset(test_dataset_path)

    # 배치 생성으로 가속화
    gen_batch_size = int(gen_config.GENERATION_PARAMS.get("batch_size", 8))
    loader_workers = int(gen_config.GENERATION_PARAMS.get("num_workers", 0))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=gen_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=loader_workers,
        pin_memory=(device.type == "cuda")
    )
    print(f"✅ Loaded dataset for conditions from '{test_dataset_path}'.")

    # --- PDB 생성 루프 ---
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Generating PDBs")
    
    num_samples_per_sequence = gen_config.GENERATION_PARAMS["num_samples_per_sequence"]
    # [최적화] 샘플링 청크 사이즈 로드
    samples_batch_size = gen_config.GENERATION_PARAMS.get("samples_batch_size", 50) # 기본값 50
    # [최적화] 자동 혼합정밀도 설정
    precision = str(gen_config.GENERATION_PARAMS.get("precision", "fp32")).lower()
    autocast_dtype = None
    autocast_enabled = False
    if device.type == "cuda":
        if precision == "fp16":
            autocast_dtype = torch.float16
            autocast_enabled = True
        elif precision == "bf16":
            autocast_dtype = torch.bfloat16
            autocast_enabled = True

    # [최적화] 동적 피처 인덱싱(고정) — 루프 밖에서 계산
    feature_indices = {}
    current_idx = 0
    for key in train_config.FEATURE_KEYS:
        feature_indices[key] = slice(current_idx, current_idx + train_config.FEATURE_DIMS[key])
        current_idx += train_config.FEATURE_DIMS[key]

    def get_feature(tensor, key):
        return tensor[:, :, feature_indices.get(key)]

    for batch_idx, batch_data in pbar:
        # --- 원본 데이터 준비 ---
        _, cond_dict_cpu, seq_mask_cpu, _, length_tensor, bond_len_cpu = batch_data
        original_batch_size = seq_mask_cpu.shape[0]

        # [최적화] 샘플링을 청크 단위로 나누어 실행
        for s_chunk_start in range(0, num_samples_per_sequence, samples_batch_size):
            s_chunk_end = min(s_chunk_start + samples_batch_size, num_samples_per_sequence)
            current_samples_in_chunk = s_chunk_end - s_chunk_start

            # [최적화] 청크 전체가 이미 생성되어 있으면 스킵
            if gen_config.GENERATION_PARAMS.get("skip_existing", False):
                all_exist = True
                for original_item_idx in range(original_batch_size):
                    global_item_idx = batch_idx * original_batch_size + original_item_idx
                    if 'pdb_file' in dataset.data and global_item_idx < len(dataset.data['pdb_file']):
                        pdb_id = Path(dataset.data['pdb_file'][global_item_idx]).stem
                    else:
                        pdb_id = f"idx{global_item_idx}"
                    prefix = gen_config.GENERATION_PARAMS.get("filename_prefix", "gen")
                    for sample_sub_idx in range(current_samples_in_chunk):
                        global_sample_idx = s_chunk_start + sample_sub_idx
                        out_path = output_dir / f"{prefix}_{pdb_id}_sample_{global_sample_idx}.pdb"
                        if not out_path.exists():
                            all_exist = False
                            break
                    if not all_exist:
                        break
                if all_exist:
                    continue

            # --- [최적화] 현재 청크에 맞게 조건 텐서 확장 ---
            mega_batch_size = original_batch_size * current_samples_in_chunk
            cond_dict = {
                k: v.repeat_interleave(current_samples_in_chunk, dim=0).to(device, non_blocking=True)
                for k, v in cond_dict_cpu.items()
            }
            seq_mask = seq_mask_cpu.repeat_interleave(current_samples_in_chunk, dim=0).to(device, non_blocking=True)

            # --- 결합 길이 텐서 준비 (설정에 따라 분기) ---
            if gen_config.GENERATION_PARAMS.get("use_ideal_bond_lengths", False):
                # 이상적인 결합 길이 사용
                max_len = train_config.DATA_SPECS["max_len"]
                ideal_lengths = torch.tensor([
                    gen_config.IDEAL_BOND_LENGTHS["N_CA_LENGTH"],
                    gen_config.IDEAL_BOND_LENGTHS["CA_C_LENGTH"],
                    gen_config.IDEAL_BOND_LENGTHS["C_N_LENGTH"],
                ], device=device)
                bond_len = ideal_lengths.view(1, 1, 3).expand(mega_batch_size, max_len, -1)
            else:
                # 데이터셋의 실측 결합 길이 사용
                bond_len = bond_len_cpu.repeat_interleave(current_samples_in_chunk, dim=0).to(device, non_blocking=True)
            
            # --- [최적화] 확장된 배치로 모델 샘플링 (단 한번 호출) ---
            shape = (mega_batch_size, train_config.DATA_SPECS["max_len"], diffusion.feature_input_dim)
            with torch.no_grad():
                with autocast(dtype=autocast_dtype, enabled=autocast_enabled):
                    predicted_x0 = diffusion.sample(
                        cond_dict,
                        guidance_scale=gen_config.GENERATION_PARAMS["guidance_scale"],
                        shape=shape,
                        mask=seq_mask
                    )

            # --- 결과 처리 및 저장 (이후 로직은 동일) ---
            bb_torsions = get_feature(predicted_x0, 'bb_torsions')
            bb_angles = get_feature(predicted_x0, 'bb_angles')
            coords_batch = nerf_build_batch(
                phi=bb_torsions[:, :, 0], psi=bb_torsions[:, :, 1], omega=bb_torsions[:, :, 2],
                bond_angle_n_ca_c=bb_angles[:, :, 0], bond_angle_ca_c_n=bb_angles[:, :, 1], bond_angle_c_n_ca=bb_angles[:, :, 2],
                bond_len_n_ca=bond_len[:, :, 0], bond_len_ca_c=bond_len[:, :, 1], bond_len_c_n=bond_len[:, :, 2]
            ).detach().cpu().numpy()

            # 배치의 각 항목에 대해 파일 저장
            for i in range(mega_batch_size):
                original_item_idx = i // current_samples_in_chunk
                sample_sub_idx = i % current_samples_in_chunk
                
                global_sample_idx = s_chunk_start + sample_sub_idx
                global_item_idx = batch_idx * original_batch_size + original_item_idx
                
                seq_len = int(length_tensor[original_item_idx].item())

                if 'pdb_file' in dataset.data and global_item_idx < len(dataset.data['pdb_file']):
                    pdb_id = Path(dataset.data['pdb_file'][global_item_idx]).stem
                else:
                    pdb_id = f"idx{global_item_idx}"

                sequence_3_letter = dataset.data['sequences_3letter'][global_item_idx][:seq_len]
                pbar.set_postfix_str(f"{pdb_id}_sample_{global_sample_idx}")
                coords_sample = coords_batch[i, :seq_len * 3, :]

                try:
                    prefix = gen_config.GENERATION_PARAMS.get("filename_prefix", "gen")
                    output_filename = f"{prefix}_{pdb_id}_sample_{global_sample_idx}.pdb"
                    out_path = output_dir / output_filename
                    if gen_config.GENERATION_PARAMS.get("skip_existing", False) and out_path.exists():
                        continue
                    # [최적화] 파일 쓰기 최소화 — 문자열 누적 후 일괄 쓰기
                    atom_names = ["N", "CA", "C"]
                    lines = []
                    for res_idx in range(seq_len):
                        res_name = sequence_3_letter[res_idx]
                        base = res_idx * 3
                        for i_atom in range(3):
                            atom_name = atom_names[i_atom]
                            coord_idx = base + i_atom
                            x, y, z = coords_sample[coord_idx]
                            lines.append(
                                f"ATOM  {len(lines)+1:5d} {atom_name:<4s}{res_name:3s} A{res_idx+1:4d}    "
                                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {atom_name[0]:>2s}  \n"
                            )
                    lines.append("TER\n")
                    with open(out_path, "w") as f:
                        f.writelines(lines)
                except Exception as e:
                    tqdm.write(f"‼️ ERROR generating PDB for {pdb_id}_sample_{global_sample_idx}: {e}")

    print("\n✅ Generation complete.")

# --- 4. 실행 ---
if __name__ == "__main__":
    generate_pdbs()
