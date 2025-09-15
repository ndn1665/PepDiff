# analysis_utils.py
# 설명: 분석 및 그래프 생성을 위한 공통 유틸리티 함수 모음

from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

from Bio.PDB.Structure import Structure
from Bio.SVDSuperimposer import SVDSuperimposer

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# model.py는 상위 디렉토리에, nerf.py는 현재 디렉토리에 위치해야 합니다.
from model import AbstractDiffusionModel
from .nerf import nerf_build_batch

AA_THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def three_to_one(resname):
    """3글자 아미노산 코드를 1글자 코드로 변환"""
    return AA_THREE_TO_ONE.get(resname, 'X')


def iqr_bounds(values, k: float):
    """
    IQR 기반 이상치 탐지를 위한 경계와 마스크를 계산합니다.
    (하한, 상한), 그리고 이상치가 아닌 값에 대한 boolean 마스크를 반환합니다.
    """
    # 숫자형이고 유한한 값만 필터링하여 통계 계산
    v_finite = np.array([x for x in values if isinstance(x, (int, float)) and np.isfinite(x)])
    
    if v_finite.size < 4:
        # 데이터가 부족하여 이상치를 판단할 수 없으므로 모두 정상으로 간주
        return (-np.inf, np.inf), [True] * len(values)
        
    q1, q3 = np.percentile(v_finite, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    
    # 원본 리스트의 각 값에 대해 경계 내에 있는지 확인하는 마스크 생성
    mask = [isinstance(x, (int, float)) and lower <= x <= upper for x in values]
    return (lower, upper), mask


def iqr_filter(values, k: float):
    """
    IQR 기반으로 리스트에서 이상치를 필터링하고, 이상치는 None으로 대체합니다.
    """
    v_finite = np.array([x for x in values if x is not None and isinstance(x, (int, float)) and np.isfinite(x)])
    
    if v_finite.size < 4:
        return values
        
    q1, q3 = np.percentile(v_finite, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    
    return [x if x is not None and isinstance(x, (int, float)) and lower <= x <= upper else None for x in values]


def _extract_coords(struct: Structure, atom_types):
    """구조체에서 특정 원자의 좌표를 추출합니다."""
    coords = [
        a.coord
        for a in struct[0].get_atoms()
        if a.name in atom_types and np.isfinite(a.coord).all()
    ]
    return np.asarray(coords, dtype=float)


def align(native: Structure, model: Structure, config) -> float:
    """
    두 구조를 정렬하고 RMSD를 계산합니다.
    [수정] 두 구조에 공통적으로 존재하는 최대 백본 원자 집합을 찾아 RMSD를 계산합니다.
           이를 통해 원본 PDB에 일부 원자가 누락되어 있어도, 비교 가능한 부분만으로
           RMSD를 계산하여 통계에서 누락되는 데이터 포인트를 최소화합니다.
    """
    native_residues = list(native[0].get_residues())
    model_residues = list(model[0].get_residues())
    
    # 1. 전체 잔기 개수가 동일한지 먼저 확인
    if len(native_residues) != len(model_residues):
        return np.inf

    native_coords = []
    model_coords = []

    # 2. 잔기 쌍(residue pair)을 순회하며 공통 백본 원자 좌표 추출
    for native_res, model_res in zip(native_residues, model_residues):
        # 각 잔기 쌍에서 공통으로 존재하는 원자들의 좌표만 추출
        for atom_name in config.ATOM_TYPES:
            if atom_name in native_res and atom_name in model_res:
                native_coords.append(native_res[atom_name].coord)
                model_coords.append(model_res[atom_name].coord)

    # 3. 추출된 좌표가 RMSD 계산에 충분한지 확인 (최소 1개 아미노산, 즉 3개 원자)
    if len(native_coords) < 3:
        return np.inf
        
    n_xyz = np.asarray(native_coords, dtype=float)
    m_xyz = np.asarray(model_coords, dtype=float)

    # 4. SVDSuperimposer를 사용하여 RMSD 계산
    si = SVDSuperimposer()
    si.set(n_xyz, m_xyz)
    try:
        si.run()
    except np.linalg.LinAlgError:
        return np.inf

    rot, tran = si.get_rotran()
    diff = n_xyz - (m_xyz @ rot + tran)
    # 공통 원자 개수(n_xyz.shape[0])로 나누어 RMSD를 계산합니다.
    return float(np.sqrt((diff ** 2).sum() / n_xyz.shape[0]))


def rmsd_calculation(origin_path: Path, sample_path: Path, p: PDBParser, config):
    """두 PDB 파일 간의 RMSD를 계산합니다."""
    if not (origin_path.exists() and sample_path.exists()):
        return None
    try:
        native = p.get_structure("origin", origin_path)
        model = p.get_structure("sample", sample_path)
        rms = align(native, model, config)
        return rms if np.isfinite(rms) else None
    except Exception:
        return None


def add_backbone_carbonyl_oxygens(coords_n_ca_c: np.ndarray) -> np.ndarray:
    """
    N, C-alpha, C 좌표만 있는 배치 데이터에 카르보닐 산소(O) 원자를 추가합니다.
    Args:
        coords_n_ca_c (np.ndarray): (Batch, Length * 3, 3) 모양의 좌표 배열.
    Returns:
        np.ndarray: (Batch, (Length-1) * 4, 3) 모양의 N, CA, C, O 좌표 배열.
                    마지막 잔기의 산소는 정의할 수 없으므로 (Length-1)이 됩니다.
    """
    batch_size = coords_n_ca_c.shape[0]
    num_atoms = coords_n_ca_c.shape[1]
    seq_len = num_atoms // 3
    
    # 이상적인 기하학적 상수 (Amber ff)
    C_O_LENGTH = 1.231
    
    output_coords = []

    for i in range(batch_size):
        coords = coords_n_ca_c[i]
        full_backbone_atoms = []
        for res_idx in range(seq_len - 1):
            # 현재 잔기와 다음 잔기의 원자 좌표 추출
            n_i = coords[res_idx * 3]
            ca_i = coords[res_idx * 3 + 1]
            c_i = coords[res_idx * 3 + 2]
            n_ip1 = coords[(res_idx + 1) * 3]
            
            # C-CA, C-N 벡터 계산
            v_c_ca = ca_i - c_i
            v_c_n_next = n_ip1 - c_i
            
            # 두 벡터를 정규화
            u_c_ca = v_c_ca / np.linalg.norm(v_c_ca)
            u_c_n_next = v_c_n_next / np.linalg.norm(v_c_n_next)
            
            # C=O 벡터 방향은 CA-C-N 각의 이등분선과 거의 반대 방향
            v_co_dir = -(u_c_ca + u_c_n_next)
            u_co = v_co_dir / np.linalg.norm(v_co_dir)
            
            # 산소 원자 좌표 계산
            o_i = c_i + u_co * C_O_LENGTH
            
            full_backbone_atoms.extend([n_i, ca_i, c_i, o_i])
        
        output_coords.append(np.array(full_backbone_atoms))

    return np.array(output_coords)


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Kabsch 알고리즘을 사용하여 두 좌표 집합 P와 Q를 최적으로 정렬한 후,
    주어진 모든 원자 간의 RMSD를 계산합니다. SVD 수렴 실패 시 np.nan을 반환합니다.
    Args:
        P (np.ndarray): 첫 번째 구조의 원자 좌표. shape: (N, 3)
        Q (np.ndarray): 두 번째 구조의 원자 좌표. shape: (N, 3)
    Returns:
        float: RMSD 값 또는 계산 실패 시 np.nan.
    """
    # 0. 입력값에 NaN이 있는지 확인
    if np.any(np.isnan(P)) or np.any(np.isnan(Q)):
        return np.nan

    # 1. 중심을 원점으로 이동
    P_centered = P - P.mean(axis=0)
    Q_centered = Q - Q.mean(axis=0)

    # 2. 공분산 행렬 계산
    C = np.dot(Q_centered.T, P_centered)

    # 3. 최적의 회전 행렬 계산 (SVD 사용)
    try:
        V, S, W = np.linalg.svd(C)
    except np.linalg.LinAlgError:
        # SVD가 수렴하지 않는 경우 (예: 초기 학습 단계의 비정상적인 구조)
        return np.nan

    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    
    U = np.dot(V, W)

    # 4. P를 Q에 정렬
    P_aligned = np.dot(P_centered, U.T)
    
    # 5. RMSD 계산
    return np.sqrt(np.mean(np.sum((P_aligned - Q_centered) ** 2, axis=1)))


def plot_ramachandran(phi_pred, psi_pred, phi_true, psi_true, epoch: int, output_dir: str):
    """
    예측된 phi/psi 각도와 실제 각도의 분포를 라마찬드란 플롯으로 비교하여 저장합니다.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 실제 분포
    axes[0].hist2d(phi_true, psi_true, bins=100, cmap='viridis', range=[[-np.pi, np.pi], [-np.pi, np.pi]])
    axes[0].set_title(f'True Ramachandran Distribution')
    axes[0].set_xlim([-np.pi, np.pi])
    axes[0].set_ylim([-np.pi, np.pi])
    ticks = np.linspace(-np.pi, np.pi, 5)
    tick_labels = ['-180', '-90', '0', '90', '180']
    axes[0].set_xticks(ticks, labels=tick_labels)
    axes[0].set_yticks(ticks, labels=tick_labels)

    # 예측 분포
    axes[1].hist2d(phi_pred, psi_pred, bins=100, cmap='viridis', range=[[-np.pi, np.pi], [-np.pi, np.pi]])
    axes[1].set_title(f'Predicted Ramachandran - Epoch {epoch}')
    axes[1].set_xlim([-np.pi, np.pi])
    axes[1].set_ylim([-np.pi, np.pi])
    axes[1].set_xticks(ticks, labels=tick_labels)
    axes[1].set_yticks(ticks, labels=tick_labels)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ramachandran_epoch_{epoch}.png", dpi=150)
    plt.close()


@torch.no_grad()
def run_validation_metrics(
    model: AbstractDiffusionModel,
    diffusion,
    val_loader,
    epoch: int,
    output_dir: str = "./validation_outputs"
):
    """
    Validation 데이터셋에 대해 3D 구조를 생성하고,
    전체 백본(N, CA, C, O) RMSD와 라마찬드란 플롯을 계산/저장합니다.
    """
    model.eval()
    device = next(model.parameters()).device
    # DataParallel로 래핑된 경우 내부 실제 모델로 언랩
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    all_rmsds = []
    all_phi_pred, all_psi_pred = [], []
    all_phi_true, all_psi_true = [], []
    num_rmsd_attempts = 0
    
    print("\nRunning validation metrics (Backbone RMSD, Ramachandran)...")
    pbar = tqdm(val_loader, desc="Validation Metrics")
    for batch_data in pbar:
        x0_cpu, c_dict_cpu, _, _, lengths_cpu, bond_len_cpu = batch_data
        
        # --- 1. 데이터 준비 ---
        c = {key: val.to(device) for key, val in c_dict_cpu.items()}
        bond_len = bond_len_cpu.to(device)
        num_samples = x0_cpu.shape[0]

        # --- 1.1. 학습과 동일한 attention mask 생성 ---
        seq_mask = torch.zeros((num_samples, base_model.max_len), dtype=torch.float32, device=device)
        for bi in range(num_samples):
            L = int(lengths_cpu[bi].item())
            if L > 0:
                seq_mask[bi, :L] = 1.0

        # --- 2. 모델을 통해 구조(피처) 예측 ---
        shape = (num_samples, base_model.max_len, base_model.feature_input_dim)
        predicted_x0 = diffusion.sample(c, guidance_scale=1.0, shape=shape, mask=seq_mask)
        
        # --- 3. NERF를 사용하여 3D 좌표 생성 ---
        # 예측된 구조
        pred_torsions = predicted_x0[:, :, 0:3]
        pred_angles = predicted_x0[:, :, 3:6]
        coords_pred_batch_n_ca_c = nerf_build_batch(
            phi=pred_torsions[:, :, 0], psi=pred_torsions[:, :, 1], omega=pred_torsions[:, :, 2],
            bond_angle_n_ca_c=pred_angles[:, :, 0], bond_angle_ca_c_n=pred_angles[:, :, 1], bond_angle_c_n_ca=pred_angles[:, :, 2],
            bond_len_n_ca=bond_len[:, :, 0], bond_len_ca_c=bond_len[:, :, 1], bond_len_c_n=bond_len[:, :, 2]
        )

        # 실제 구조
        true_torsions = x0_cpu[:, :, 0:3]
        true_angles = x0_cpu[:, :, 3:6]
        coords_true_batch_n_ca_c = nerf_build_batch(
            phi=true_torsions[:, :, 0], psi=true_torsions[:, :, 1], omega=true_torsions[:, :, 2],
            bond_angle_n_ca_c=true_angles[:, :, 0], bond_angle_ca_c_n=true_angles[:, :, 1], bond_angle_c_n_ca=true_angles[:, :, 2],
            bond_len_n_ca=bond_len_cpu[:, :, 0], bond_len_ca_c=bond_len_cpu[:, :, 1], bond_len_c_n=bond_len_cpu[:, :, 2]
        )

        # --- 3.5. 카르보닐 산소(O) 원자 추가 ---
        coords_pred_full_bb_batch = add_backbone_carbonyl_oxygens(coords_pred_batch_n_ca_c.cpu().numpy())
        coords_true_full_bb_batch = add_backbone_carbonyl_oxygens(coords_true_batch_n_ca_c.cpu().numpy())

        # --- 4. RMSD 및 각도 분포 데이터 수집 ---
        for i in range(num_samples):
            # 마지막 잔기는 산소 원자를 정의할 수 없으므로 seq_len-1 길이의 백본을 비교
            seq_len = lengths_cpu[i].item()
            if seq_len <= 1:
                continue
            
            num_rmsd_attempts += 1
            num_bb_atoms = (seq_len - 1) * 4
            
            coords_pred_bb = coords_pred_full_bb_batch[i, :num_bb_atoms, :]
            coords_true_bb = coords_true_full_bb_batch[i, :num_bb_atoms, :]
            
            if coords_pred_bb.shape[0] > 0 and coords_pred_bb.shape == coords_true_bb.shape:
                rmsd_val = kabsch_rmsd(coords_pred_bb, coords_true_bb)
                if not np.isnan(rmsd_val):
                    all_rmsds.append(rmsd_val)

            # 라마찬드란 플롯을 위한 phi, psi 각도 수집 (기존과 동일)
            all_phi_pred.extend(pred_torsions[i, :seq_len, 0].cpu().numpy())
            all_psi_pred.extend(pred_torsions[i, :seq_len, 1].cpu().numpy())
            all_phi_true.extend(true_torsions[i, :seq_len, 0].numpy())
            all_psi_true.extend(true_torsions[i, :seq_len, 1].numpy())

    # --- 5. 최종 결과 계산 및 저장 ---
    avg_rmsd = np.mean(all_rmsds) if all_rmsds else 0.0
    num_converged = len(all_rmsds)
    
    print(
        f"Validation Backbone RMSD: {avg_rmsd:.4f} "
        f"(calculated on {num_converged}/{num_rmsd_attempts} convergent samples)"
    )
    
    # 라마찬드란 플롯 저장 (비유한 값 필터링 및 래핑 보정)
    import os
    os.makedirs(output_dir, exist_ok=True)
    phi_pred_arr = np.array(all_phi_pred, dtype=float)
    psi_pred_arr = np.array(all_psi_pred, dtype=float)
    phi_true_arr = np.array(all_phi_true, dtype=float)
    psi_true_arr = np.array(all_psi_true, dtype=float)

    # 유한값만 사용
    pred_finite = np.isfinite(phi_pred_arr) & np.isfinite(psi_pred_arr)
    true_finite = np.isfinite(phi_true_arr) & np.isfinite(psi_true_arr)

    # --- 추가: 진단을 위한 로그 ---
    num_total_preds = len(phi_pred_arr)
    num_finite_preds = np.sum(pred_finite)
    print(
        f"Ramachandran plot data for epoch {epoch}: "
        f"Found {num_finite_preds} finite predictions out of {num_total_preds} total."
    )
    # -------------------------

    # 각도 래핑 보정 (-pi~pi)
    wrap = lambda a: np.arctan2(np.sin(a), np.cos(a))
    phi_pred_arr, psi_pred_arr = wrap(phi_pred_arr[pred_finite]), wrap(psi_pred_arr[pred_finite])
    phi_true_arr, psi_true_arr = wrap(phi_true_arr[true_finite]), wrap(psi_true_arr[true_finite])

    plot_ramachandran(
        phi_pred_arr, psi_pred_arr,
        phi_true_arr, psi_true_arr,
        epoch, output_dir
    )
    print(f"Ramachandran plot saved to '{output_dir}/ramachandran_epoch_{epoch}.png'")
    
    return avg_rmsd
