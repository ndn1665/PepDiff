import torch

def robust_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """벡터를 안전하게 정규화합니다."""
    norms = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v / (norms + eps)


class Rigid:
    """
    Rigid transformation (회전 + 평행이동)을 나타내는 클래스.
    모든 연산은 벡터화되어 배치 처리를 지원합니다.

    rotation: (..., 3, 3)
    translation: (..., 3)
    """
    def __init__(self, rotation: torch.Tensor, translation: torch.Tensor):
        """
        Args:
            rotation (Tensor): (..., 3, 3) 회전 행렬 (로컬->글로벌)
            translation (Tensor): (..., 3) 평행이동 벡터 (글로벌 좌표계에서의 원점)
        """
        self.rotation = rotation
        self.translation = translation

    def apply(self, points: torch.Tensor) -> torch.Tensor:
        """
        Global -> Local 변환: R^T * (X_global - t)

        points: (B, M, 3)
        rotation/translation: (B, L, 3, 3), (B, L, 3)
        반환: (B, L, M, 3)  # 각 잔기 프레임(i)에서 모든 점(j)의 로컬 좌표
        """
        # (B, 1, M, 3) - (B, L, 1, 3) -> (B, L, M, 3)
        centered = points.unsqueeze(1) - self.translation.unsqueeze(2)
        # (B, L, 3, 3)^T x (B, L, M, 3) -> (B, L, M, 3)
        return torch.einsum('...ji,...aj->...ai', self.rotation, centered)

    def inverse_apply(self, points: torch.Tensor) -> torch.Tensor:
        """
        Local -> Global 변환: R * X_local + t

        points: (B, L, M, 3)
        반환: (B, L, M, 3)
        """
        rotated = torch.einsum('...ij,...aj->...ai', self.rotation, points)
        return rotated + self.translation.unsqueeze(2)

    def inverse(self) -> "Rigid":
        """역변환 Rigid 반환."""
        inv_R = self.rotation.transpose(-1, -2)
        inv_t = -torch.einsum('...ij,...j->...i', inv_R, self.translation)
        return Rigid(inv_R, inv_t)


def build_local_frames(coords: torch.Tensor) -> Rigid:
    """
    각 아미노산 잔기의 N, CA, C 좌표로부터 로컬 좌표계(Rigid)를 생성합니다.
    AF2 관행:
      - 원점: CA
      - y축: (N - CA) 정규화
      - x축: (C - CA)에서 y축 성분 제거 후 정규화
      - z축: x × y 정규화
      - (선택) 재직교화: y' = normalize(z × x)

    Args:
        coords: (B, L*3, 3)  # 원자 순서는 [N, CA, C]가 잔기마다 반복

    Returns:
        Rigid(rotation=(B, L, 3, 3), translation=(B, L, 3))
    """
    B, _, _ = coords.shape
    L = coords.shape[1] // 3

    # (B, L*3, 3) -> (B, L, 3, 3) : [N, CA, C]
    residue_coords = coords.reshape(B, L, 3, 3)
    n = residue_coords[:, :, 0, :]   # (B, L, 3)
    ca = residue_coords[:, :, 1, :]
    c  = residue_coords[:, :, 2, :]

    v_y = robust_normalize(n - ca)  # (B, L, 3)
    v_xu = c - ca
    v_x = robust_normalize(v_xu - (v_xu * v_y).sum(-1, keepdim=True) * v_y)
    v_z = robust_normalize(torch.cross(v_x, v_y, dim=-1))
    # 재직교화로 수치적 안정성 보강
    v_y = robust_normalize(torch.cross(v_z, v_x, dim=-1))

    # 회전 행렬을 [v_x, v_y, v_z] 열로 구성 (로컬->글로벌)
    rotation = torch.stack([v_x, v_y, v_z], dim=-1)  # (B, L, 3, 3)
    translation = ca                                  # (B, L, 3)
    return Rigid(rotation, translation)


def fape_loss(
    pred_coords: torch.Tensor,   # (B, L*3, 3)
    target_coords: torch.Tensor, # (B, L*3, 3)
    lengths: torch.Tensor,       # (B,)  실제 residue 길이(패딩 제외)
    clamp_dist: float = 10.0,
    eps: float = 1e-8,
    use_ca_only: bool = False,   # True면 CA 원자만 사용
    local_window_k: int = None,  # 정수 k면 |i-j|<=k 윈도우만 사용
    return_stats: bool = False   # True면 (loss, stats) 반환
) -> torch.Tensor:
    """
    Frame-Aligned Point Error (FAPE) 손실을 벡터화로 계산합니다.

    절차:
      1) 예측/실제 좌표로 각 잔기 로컬 프레임 생성
      2) 모든 원자 좌표를 모든 잔기 프레임으로 로컬화 (i×j)
      3) L2 거리 -> 클램프
      4) 패딩 마스크 (i-프레임 × j-원자) 외적 적용 후 평균

    Returns:
        스칼라 텐서: 평균 FAPE
    """
    B, _, _ = pred_coords.shape
    L = pred_coords.shape[1] // 3
    device = pred_coords.device

    # 1) 로컬 프레임
    pred_frames   = build_local_frames(pred_coords)
    target_frames = build_local_frames(target_coords)

    # 2) Global -> Local (핵심: inverse() 쓰지 말고 apply() 사용)
    #    결과: (B, L, 3L, 3)
    pred_local   = target_frames.apply(pred_coords)
    target_local = target_frames.apply(target_coords)

    # 3) L2 거리
    dist_err = torch.linalg.norm(pred_local - target_local, dim=-1)  # (B, L, 3L)

    # 4) 클램프
    clamped = torch.clamp(dist_err, max=clamp_dist)

    # 5) i×j 외적 마스크
    # i(잔기) 유효: (B, L)
    res_mask = (torch.arange(L, device=device).unsqueeze(0) < lengths.unsqueeze(1))
    # j(원자) 유효: (B, 3L)
    atom_mask = res_mask.repeat_interleave(3, dim=1)

    # CA-only 선택 시 j 마스크를 CA 위치만 True로 제한
    if use_ca_only:
        ca_mask_1d = torch.zeros(3 * L, dtype=torch.bool, device=device)
        ca_mask_1d[1::3] = True  # CA 인덱스: 1, 4, 7, ...
        atom_mask = atom_mask & ca_mask_1d.unsqueeze(0)

    # (B, L, 3L)
    pair_mask = res_mask.unsqueeze(-1) & atom_mask.unsqueeze(1)

    # 로컬 윈도우(|i-j|<=k) 적용
    if local_window_k is not None and local_window_k >= 0:
        res_idx = torch.arange(L, device=device)
        j_res_idx = torch.arange(L, device=device).repeat_interleave(3)  # (3L,)
        # (L, 3L): 각 i에 대해 허용되는 j 원자 위치
        local_mask_2d = (res_idx.unsqueeze(1) - j_res_idx.unsqueeze(0)).abs() <= local_window_k
        pair_mask = pair_mask & local_mask_2d.unsqueeze(0)

    # (선택) 수치 안전장치
    finite_mask = torch.isfinite(clamped)
    pair_mask = pair_mask & finite_mask

    if pair_mask.any():
        loss_val = clamped[pair_mask].mean()
    else:
        loss_val = torch.tensor(0.0, device=device)

    if return_stats:
        with torch.no_grad():
            over_clamp = (dist_err > clamp_dist) & pair_mask
            clamp_ratio = over_clamp.float().mean() if pair_mask.any() else torch.tensor(0.0, device=device)
        return loss_val, {"clamp_ratio": clamp_ratio}
    return loss_val
