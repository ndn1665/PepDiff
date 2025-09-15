# energy_utils.py (S 동적 처리 + 정규화 최종본)

import torch
import torchani
import warnings

# ── 원소 목록(종 이름)만 쓰므로 'C'가 두 번 나와도 무방: CA(탄소), C(탄소) 모두 'C'
RESIDUE_BACKBONE_ATOMS = {
    'A': ['N', 'C', 'C', 'O'], 'R': ['N', 'C', 'C', 'O'], 'N': ['N', 'C', 'C', 'O'],
    'D': ['N', 'C', 'C', 'O'], 'Q': ['N', 'C', 'C', 'O'],
    'E': ['N', 'C', 'C', 'O'], 'G': ['N', 'C', 'C', 'O'], 'H': ['N', 'C', 'C', 'O'],
    'I': ['N', 'C', 'C', 'O'], 'L': ['N', 'C', 'C', 'O'], 'K': ['N', 'C', 'C', 'O'],
    'F': ['N', 'C', 'C', 'O'], 'P': ['N', 'C', 'C', 'O'],
    'S': ['N', 'C', 'C', 'O'], 'T': ['N', 'C', 'C', 'O'], 'W': ['N', 'C', 'C', 'O'],
    'Y': ['N', 'C', 'C', 'O'], 'V': ['N', 'C', 'C', 'O'],
    'C': ['N', 'C', 'C', 'O'],
    'M': ['N', 'C', 'C', 'O'],
    'X': []
}

class EnergyCalculator:
    """TorchANI를 사용하여 분자 배치의 포텐셜 에너지를 계산(정규화 포함)."""
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        print(f"Initializing EnergyCalculator with TorchANI on {self.device}...")
        try:
            self.model = torchani.models.ANI2x().to(self.device)
            self.model.eval()
            print("TorchANI ANI-2x model loaded successfully.")
        except Exception as e:
            warnings.warn(f"❌ Failed to load TorchANI model. Energy calculation will be skipped. Error: {e}")
            self.model = None

        # 런닝 통계(유효 샘플만 반영; z-score에 사용)
        self.register_running_norm = True
        self._eps = 1e-6
        self._count = 0
        self._mean = torch.zeros(1, device=self.device)
        self._M2 = torch.zeros(1, device=self.device)

    def _update_running_stats(self, values: torch.Tensor):
        for v in values.detach():
            self._count += 1
            delta = v - self._mean
            self._mean = self._mean + delta / self._count
            delta2 = v - self._mean
            self._M2 = self._M2 + delta * delta2

    def _current_std(self):
        if self._count < 2:
            return torch.ones(1, device=self.device)
        var = self._M2 / (self._count - 1)
        return torch.sqrt(torch.clamp(var, min=self._eps))

    def compute(self, sequences, coords_batch, update_stats: bool = True):
        """
        백본 원자(N, C, C, O) 4개로 구성된 배치에 대해 에너지를 계산.
        - update_stats=True일 때만 런닝 정규화 통계를 갱신.
        - 모든 잔기는 4개의 원자를 갖는다고 가정하여 로직을 단순화.
        """
        if self.model is None:
            return torch.zeros(len(sequences), device=self.device)

        energies_raw = []
        valid_flags = []

        for i in range(len(sequences)):
            try:
                coords = coords_batch[i]  # (max_len*4, 3)

                # 1. 유효 원자 마스크 및 개수 확인
                valid_atom_mask = (coords.abs().sum(dim=-1) > 0)
                n_valid_atoms = int(valid_atom_mask.sum().item())

                # 유효 원자가 없거나, 4의 배수가 아니면 비정상으로 간주하고 스킵
                if n_valid_atoms == 0 or n_valid_atoms % 4 != 0:
                    energies_raw.append(torch.tensor(0.0, device=self.device))
                    valid_flags.append(False)
                    continue
                
                # 2. 유효 잔기 수 계산 및 species 리스트 생성 (단순화)
                n_res_valid = n_valid_atoms // 4
                species_list = ['N', 'C', 'C', 'O'] * n_res_valid
                
                # 3. 좌표 슬라이싱 및 에너지 계산
                coords_use = coords[:n_valid_atoms, :].unsqueeze(0)  # (1, natoms, 3)
                species_tensor = self.model.species_to_tensor(species_list).to(self.device).unsqueeze(0)
                
                _, energy = self.model((species_tensor, coords_use))
                energies_raw.append(energy.squeeze())
                valid_flags.append(True)

            except Exception as e:
                warnings.warn(f"TorchANI energy calculation failed for sequence {i}. Error: {e}")
                energies_raw.append(torch.tensor(0.0, device=self.device))
                valid_flags.append(False)

        energies_raw = torch.stack(energies_raw)

        # 런닝 정규화 갱신
        if self.register_running_norm and update_stats:
            valid_mask = torch.tensor(valid_flags, device=self.device, dtype=torch.bool)
            if valid_mask.any():
                self._update_running_stats(energies_raw[valid_mask])

        # z-score 정규화
        mean = self._mean
        std = self._current_std()
        energies_norm = torch.where(
            torch.tensor(valid_flags, device=self.device),
            (energies_raw - mean) / std,
            torch.zeros_like(energies_raw)
        )
        return energies_norm

