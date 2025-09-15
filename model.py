import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertEncoder, BertLMHeadModel, BertPreTrainedModel, BertSelfAttention, BertIntermediate, BertOutput
from typing import Optional, Tuple, Union, Dict
from abc import ABC, abstractmethod

# 펩타이드 구조를 3D 좌표로 재구성하기 위한 NERF 알고리즘
from utils.nerf import nerf_build_batch
from utils.fape import fape_loss, build_local_frames
from utils.custom_metrics import radian_smooth_l1_loss
# 새로 생성한 설정 파일 임포트
import model_config as config


# ==============================================================================
# 유틸리티 함수
# ==============================================================================
def wrap_pi(x: torch.Tensor) -> torch.Tensor:
    """
    각도(라디안)를 -pi에서 +pi 사이의 값으로 래핑(wrapping)합니다.
    주기적인 특성을 가진 각도 데이터를 일관된 범위로 정규화하는 데 사용됩니다.
    """
    return (x + math.pi) % (2 * math.pi) - math.pi

def pad_array(arr, target_len, pad_value=0):
    """
    NumPy 배열을 지정된 길이로 패딩하거나 자릅니다.
    데이터셋 내 모든 샘플의 길이를 통일시켜 배치 처리를 용이하게 합니다.
    """
    current_len = arr.shape[0]
    if current_len >= target_len:
        return arr[:target_len]
    pad_shape = (target_len - current_len,) + arr.shape[1:]
    pad = np.full(pad_shape, pad_value, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


# ==============================================================================
# 데이터셋 클래스
# ==============================================================================
class PeptideDataset(Dataset):
    """
    Pickle 파일로부터 펩타이드 데이터를 로드하고, model_config에 따라
    동적으로 피처와 조건을 구성하여 모델 학습에 필요한 형태로 전처리합니다.
    """
    def __init__(self, dataset_path: str):
        with open(dataset_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.max_len = config.DATA_SPECS["max_len"]
        self.feature_keys = config.FEATURE_KEYS
        self.condition_keys = config.CONDITION_KEYS
        self.bond_key = config.BOND_KEY

        # 데이터셋에 필요한 모든 키가 존재하는지 확인하여 설정과 데이터의 불일치를 방지
        for key in self.feature_keys + self.condition_keys + [self.bond_key]:
            if key not in self.data:
                raise KeyError(f"'{key}' not found in the dataset. Please check model_config.py and the dataset generation script.")
        
        print(f"Dataset using feature keys: {self.feature_keys}")
        print(f"Dataset using condition keys: {self.condition_keys}")

    def __len__(self):
        """데이터셋의 총 샘플 수를 반환합니다."""
        # 'sequences_3letter' 키의 존재를 보장하여 데이터셋의 무결성 강화
        if 'sequences_3letter' not in self.data:
            raise KeyError("'sequences_3letter' key not found in the dataset, which is required for determining dataset length.")
        return len(self.data['sequences_3letter'])

    def __getitem__(self, idx):
        """
        주어진 인덱스(idx)에 해당하는 샘플을 전처리하여 텐서 형태로 반환합니다.
        """
        length = min(len(self.data['sequences_3letter'][idx]), self.max_len)
        
        # 설정된 FEATURE_KEYS에 따라 피처 데이터를 동적으로 결합
        # 예: ['bb_torsions', 'bb_angles'] -> (L, 3)과 (L, 3)을 합쳐 (L, 6) 텐서 생성
        features_list = [self.data[key][idx][:length] for key in self.feature_keys]
        features_rad = np.concatenate(features_list, axis=-1)
        
        # NaN 값을 처리하고 최대 길이에 맞춰 패딩
        feature_validity_mask = ~np.isnan(features_rad)
        features_no_nan = np.nan_to_num(features_rad, nan=0.0)
        features_padded = pad_array(features_no_nan, self.max_len, 0.0)
        feature_validity_mask_padded = pad_array(feature_validity_mask, self.max_len, False)

        # 3D 구조 재구성에 필요한 실제 결합 길이 정보
        bond_lengths = self.data[self.bond_key][idx][:length]
        bond_lengths_padded = pad_array(np.nan_to_num(bond_lengths, nan=1.45), self.max_len, 1.45)

        # 설정된 CONDITION_KEYS에 따라 조건 데이터를 동적으로 결합
        # 예: ['sequence_encoded', 'esm_embedding'] -> (L, 20)과 (L, 320)을 합쳐 (L, 340) 텐서 생성
        # --- 수정: 조건을 딕셔너리 형태로 유지 ---
        conditions_dict = {
            key: torch.from_numpy(
                pad_array(np.nan_to_num(self.data[key][idx], nan=0.0), self.max_len, 0.0)
            ).float()
            for key in self.condition_keys
        }
        
        # (추가) 데이터셋에 backbone 좌표가 있으면 N,CA,C만 선택해 전달 (학습에서 타깃 좌표로 사용)
        if 'backbone_coords' in self.data:
            try:
                bb_coords = self.data['backbone_coords'][idx][:length]  # (L, 4, 3) with order [N, CA, C, O]
                bb_nca_c = bb_coords[:, :3, :]                          # (L, 3, 3)
                bb_padded = pad_array(bb_nca_c, self.max_len, np.nan)   # (max_len, 3, 3)
                # (max_len*3, 3)로 펼침
                bb_flat = bb_padded.reshape(self.max_len * 3, 3)
                conditions_dict['gt_backbone_coords'] = torch.from_numpy(np.nan_to_num(bb_flat, nan=0.0)).float()
            except Exception:
                pass
        
        # 실제 서열 길이에 대한 마스크 (패딩된 부분은 0, 실제 부분은 1)
        seq_len_mask = np.zeros(self.max_len, dtype=np.float32)
        seq_len_mask[:length] = 1.0

        # 모델 학습에 필요한 모든 데이터를 텐서로 변환하여 반환
        return (
            torch.from_numpy(features_padded).float(),           # (L, F_dim) - 모델 입력 피처
            conditions_dict,                                     # (dict)     - 모델 조건 정보 (딕셔너리 형태)
            torch.from_numpy(seq_len_mask).float(),              # (L,)       - 어텐션 마스크용
            torch.from_numpy(feature_validity_mask_padded),      # (L, F_dim) - 손실 계산 시 유효 피처 마스크
            torch.tensor(length, dtype=torch.long),              # (스칼라)    - 실제 길이
            torch.from_numpy(bond_lengths_padded).float()        # (L, 3)     - NERF용 결합 길이
        )

# ==============================================================================
# 모델 아키텍처
# ==============================================================================
class SinusoidalPositionEmbeddings(nn.Module):
    """
    디퓨전 타임스텝(t)을 위한 Sinusoidal 위치 임베딩을 생성합니다.
    트랜스포머의 위치 인코딩과 동일한 원리를 사용하여, 연속적인 시간 정보를
    고차원 벡터 공간에 표현하여 모델이 시간의 흐름을 이해할 수 있도록 돕습니다.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1 if half_dim > 1 else 1) # 스케일링 팩터
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_scale) # 주파수 계산
        out = time.float()[:, None] * freqs[None, :]
        out = torch.cat([out.sin(), out.cos()], dim=-1) # sin, cos 함수 적용
        if self.dim % 2 == 1: # 차원이 홀수일 경우 처리
            out = torch.cat([out, torch.zeros_like(out[:, :1])], dim=-1)
        return out

class BertEmbeddingsSimple(nn.Module):
    """
    BERT 모델을 위한 간단한 임베딩 모듈.
    입력 임베딩에 '학습 가능한 위치 임베딩(position embeddings)'을 더하고
    Layer Normalization과 Dropout을 적용하여 최종 입력 텐서를 생성합니다.
    """
    def __init__(self, bert_cfg: BertConfig):
        super().__init__()
        # 각 위치(0부터 max_len-1)에 대한 고유한 임베딩 벡터를 학습
        self.position_embeddings = nn.Embedding(bert_cfg.max_position_embeddings, bert_cfg.hidden_size)
        self.register_buffer("position_ids", torch.arange(bert_cfg.max_position_embeddings).expand((1, -1)))
        self.LayerNorm = nn.LayerNorm(bert_cfg.hidden_size, eps=bert_cfg.layer_norm_eps)
        self.dropout = nn.Dropout(bert_cfg.hidden_dropout_prob)

    def forward(self, input_embeds: torch.Tensor, position_ids: Optional[torch.LongTensor] = None):
        B, L, H = input_embeds.shape
        if position_ids is None:
            position_ids = self.position_ids[:, :L].expand(B, -1)
        # 입력 임베딩에 위치 임베딩을 더하여 위치 정보를 주입
        out = input_embeds + self.position_embeddings(position_ids)
        out = self.LayerNorm(out)
        out = self.dropout(out)
        return out

class AnglesPredictor(nn.Module):
    """
    BERT 인코더의 최종 출력으로부터 노이즈(각도 정보)를 예측하는 MLP 헤드입니다.
    BERT가 추출한 고차원 표현을 다시 원래 피처의 차원으로 매핑하는 역할을 합니다.
    """
    def __init__(self, d_model: int, d_out: int):
        super().__init__()
        self.dense1 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dense2 = nn.Linear(d_model, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = self.act(x)
        x = self.layer_norm(x)
        return self.dense2(x)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DitBlock(nn.Module):
    """
    DiT(Diffusion Transformer) 블록.
    adaLN-Zero (Adaptive Layer Normalization with zero initialization) 방식을 구현하며,
    세 가지 조건부 주입 전략(sum, separate, sequential)을 지원합니다.
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1, strategy="sum"):
        super().__init__()
        self.strategy = strategy
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = BertSelfAttention(BertConfig(hidden_size=hidden_size, num_attention_heads=num_heads, attention_probs_dropout_prob=dropout, _attn_implementation="eager"))
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # 전략에 따라 adaLN 레이어들을 조건부로 생성
        if self.strategy == "sum":
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        elif self.strategy == "separate":
            self.adaLN_modulation_attn = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True))
            self.adaLN_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True))
        elif self.strategy == "sequential":
            self.adaLN_modulation_A = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
            self.adaLN_modulation_B = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        else:
            raise ValueError(f"Unknown modulation strategy: {self.strategy}")

    def forward(self, x, c, attention_mask=None):
        # --- Self-Attention 파트 ---
        x_res = x
        x_norm = self.norm1(x)

        if self.strategy == "sum":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            x_mod_attn = modulate(x_norm, shift_msa, scale_msa)
        elif self.strategy == "separate":
            c_attn, c_mlp = c
            shift_msa, scale_msa, gate_msa = self.adaLN_modulation_attn(c_attn).chunk(3, dim=1)
            shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation_mlp(c_mlp).chunk(3, dim=1)
            x_mod_attn = modulate(x_norm, shift_msa, scale_msa)
        elif self.strategy == "sequential":
            c_A, c_B = c
            shift_msa_A, scale_msa_A, gate_msa_A, shift_mlp_A, scale_mlp_A, gate_mlp_A = self.adaLN_modulation_A(c_A).chunk(6, dim=1)
            shift_msa_B, scale_msa_B, gate_msa_B, shift_mlp_B, scale_mlp_B, gate_mlp_B = self.adaLN_modulation_B(c_B).chunk(6, dim=1)
            
            x_mod_attn_A = modulate(x_norm, shift_msa_A, scale_msa_A)
            x_mod_attn = modulate(x_mod_attn_A, shift_msa_B, scale_msa_B)
            gate_msa = gate_msa_B # 최종 gate 사용

        attn_output = self.attn(x_mod_attn, attention_mask=attention_mask)[0]
        x = x_res + gate_msa.unsqueeze(1) * attn_output

        # --- MLP 파트 ---
        x_res = x
        x_norm = self.norm2(x)
        
        if self.strategy == "sum" or self.strategy == "separate":
            x_mod_mlp = modulate(x_norm, shift_mlp, scale_mlp)
        elif self.strategy == "sequential":
            x_mod_mlp_A = modulate(x_norm, shift_mlp_A, scale_mlp_A)
            x_mod_mlp = modulate(x_mod_mlp_A, shift_mlp_B, scale_mlp_B)
            gate_mlp = gate_mlp_B # 최종 gate 사용
            
        mlp_output = self.mlp(x_mod_mlp)
        x = x_res + gate_mlp.unsqueeze(1) * mlp_output
        
        return x

class AbstractDiffusionModel(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_input_dim = sum(self.config.FEATURE_DIMS[key] for key in self.config.FEATURE_KEYS)
        self.condition_per_residue = sum(self.config.CONDITION_DIMS[key] for key in self.config.CONDITION_KEYS)
        self.hidden_size = self.config.MODEL_ARCH["hidden_size"]
        self.max_len = self.config.DATA_SPECS["max_len"]

    @abstractmethod
    def forward(self, x_t: torch.Tensor, c: Dict[str, torch.Tensor], t: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None):
        """
        Args:
            x_t (Tensor): (B, L, F_dim) - Noisy features at timestep t
            c (dict):   (dict of Tensors) - Conditions (sequence, etc.)
            t (Tensor):   (B,)          - Timestep
            attn_mask (Tensor): (B, L)  - Attention mask
        Returns:
            Tensor: (B, L, F_dim) - Predicted noise
        """
        pass


class DiffusionBertDitModel(AbstractDiffusionModel):
    """
    DiT(Diffusion Transformer) 아키텍처를 구현한 디퓨전 모델.
    - 다양한 조건부 주입 전략("sum", "separate", "sequential")을 지원합니다.
    """
    def __init__(self, config):
        super().__init__(config)
        
        H = self.hidden_size
        n_layers = self.config.MODEL_ARCH["n_layers"]
        n_heads = self.config.MODEL_ARCH["n_heads"]
        self.strategy = self.config.MODEL_ARCH.get("dit_modulation_strategy", "sum")
        print(f"Initializing DiffusionBertDitModel with '{self.strategy}' modulation strategy.")

        # 1. 입력 프로젝션 및 임베딩
        self.proj_feat = nn.Linear(self.feature_input_dim, H)
        # 각 조건에 대한 별도의 프로젝션 레이어 생성
        self.cond_proj_layers = nn.ModuleDict({
            key: nn.Linear(self.config.CONDITION_DIMS[key], H)
            for key in self.config.CONDITION_KEYS
        })
        self.time_embed = SinusoidalPositionEmbeddings(H)
        self.pos_embed = nn.Embedding(self.max_len, H)

        # 2. DitBlock 스택
        self.blocks = nn.ModuleList([
            DitBlock(hidden_size=H, num_heads=n_heads, strategy=self.strategy) for _ in range(n_layers)
        ])

        # 3. 출력 헤드
        self.final_norm = nn.LayerNorm(H, elementwise_affine=False, eps=1e-6)
        self.head = AnglesPredictor(H, self.feature_input_dim)
        
        # adaLN-Zero를 위한 파라미터 초기화
        self.initialize_weights()

    def initialize_weights(self):
        # 0으로 끝나는 Linear 레이어 초기화 (adaLN-Zero)
        for block in self.blocks:
            if self.strategy == "sum":
                nn.init.zeros_(block.adaLN_modulation[1].weight)
                nn.init.zeros_(block.adaLN_modulation[1].bias)
            elif self.strategy == "separate":
                nn.init.zeros_(block.adaLN_modulation_attn[1].weight)
                nn.init.zeros_(block.adaLN_modulation_attn[1].bias)
                nn.init.zeros_(block.adaLN_modulation_mlp[1].weight)
                nn.init.zeros_(block.adaLN_modulation_mlp[1].bias)
            elif self.strategy == "sequential":
                nn.init.zeros_(block.adaLN_modulation_A[1].weight)
                nn.init.zeros_(block.adaLN_modulation_A[1].bias)
                nn.init.zeros_(block.adaLN_modulation_B[1].weight)
                nn.init.zeros_(block.adaLN_modulation_B[1].bias)

            nn.init.zeros_(block.mlp[2].weight)
            nn.init.zeros_(block.mlp[2].bias)
        nn.init.zeros_(self.head.dense2.weight)
        nn.init.zeros_(self.head.dense2.bias)

    def forward(self, x_t, c, t, attn_mask=None, position_ids=None):
        B, L, _ = x_t.shape
        
        # 1. 입력 토큰 생성
        x = self.proj_feat(x_t) 
        pos_ids = torch.arange(L, device=x_t.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(pos_ids)

        # 2. Condition Vector(s) 생성
        t_embed = self.time_embed(t.view(-1))
        
        cond_tensors = [self.cond_proj_layers[key](c[key]) for key in self.config.CONDITION_KEYS if key in c]
        if len(cond_tensors) != 2 and self.strategy in ["separate", "sequential"]:
             raise ValueError(f"'{self.strategy}' strategy requires exactly 2 conditions, but got {len(cond_tensors)}.")
        
        if self.strategy == "sum":
            # 각 조건 텐서를 독립적으로 평균내어 리스트에 담습니다.
            mean_cond_embeds = [tensor.mean(dim=1) for tensor in cond_tensors]
            
            # 시간 임베딩과 모든 평균 조건 임베딩을 합산합니다.
            condition_vector = t_embed
            for cond_embed in mean_cond_embeds:
                condition_vector = condition_vector + cond_embed
        else: # separate or sequential
            # config.CONDITION_KEYS의 순서를 따름. 0: A, 1: B
            c_embed_mean_A = cond_tensors[0].mean(dim=1)
            c_embed_mean_B = cond_tensors[1].mean(dim=1)
            condition_vector_A = t_embed + c_embed_mean_A
            condition_vector_B = t_embed + c_embed_mean_B
            if self.strategy == "separate":
                # (Attention용, MLP용)
                condition_vector = (condition_vector_A, condition_vector_B)
            else: # sequential
                # (첫번째 적용, 두번째 적용)
                condition_vector = (condition_vector_A, condition_vector_B)

        # 3. DitBlock 스택 통과
        ext_attn_mask = None
        if attn_mask is not None:
            ext_attn_mask = attn_mask[:, None, None, :].to(x.dtype)
            ext_attn_mask = (1.0 - ext_attn_mask) * -10000.0

        for block in self.blocks:
            x = block(x, condition_vector, attention_mask=ext_attn_mask)
            
        # 4. 최종 출력
        x = self.final_norm(x)
        return self.head(x)


class DiffusionBertCacModel(AbstractDiffusionModel):
    """
    BERT 인코더-디코더 구조와 Cross-Attention을 사용하여 조건부 노이즈를 예측하는 디퓨전 모델.
    - Encoder: 조건(c)을 입력받아 Key와 Value 역할을 하는 문맥 벡터(context vector)를 생성합니다.
    - Decoder: 노이즈 섞인 피처(x_t)와 시간(t) 정보를 Query로 사용하여, Encoder의 문맥 벡터와
               Cross-Attention을 수행함으로써 정교한 조건부 노이즈를 예측합니다.
    """
    def __init__(self, config):
        super().__init__(config)
        
        H = self.hidden_size

        # --- 1. 입력 프로젝션 ---
        self.proj_feat = nn.Linear(self.feature_input_dim, H)
        self.feat_ln = nn.LayerNorm(H)
        self.time_embed = SinusoidalPositionEmbeddings(H)

        # --- 2. 조건 인코더 (BERT Encoder) ---
        encoder_cfg = BertConfig(
            hidden_size=H,
            num_hidden_layers=self.config.MODEL_ARCH["n_layers"],
            num_attention_heads=self.config.MODEL_ARCH["n_heads"],
            intermediate_size=H * 4,
            max_position_embeddings=self.max_len,
            _attn_implementation="eager",  # attention implementation 설정
        )
        # --- 수정: 각 조건에 대한 별도의 프로젝션 레이어 생성 ---
        self.cond_proj_layers = nn.ModuleDict({
            key: nn.Linear(self.config.CONDITION_DIMS[key], H)
            for key in self.config.CONDITION_KEYS
        })
        self.cond_ln = nn.LayerNorm(H)
        self.cond_embeddings = BertEmbeddingsSimple(encoder_cfg)
        self.condition_encoder = BertEncoder(encoder_cfg)

        # --- 수정: 조건 결합 전략에 따른 레이어 초기화 ---
        self.merge_strategy = self.config.MODEL_ARCH.get("condition_merge_strategy", "sum")
        print(f"DiffusionBertCacModel using '{self.merge_strategy}' condition merge strategy.")
        if self.merge_strategy == 'concat':
            self.merge_proj = nn.Linear(H * len(self.config.CONDITION_KEYS), H)
        elif self.merge_strategy == 'gated':
            self.gate_proj = nn.Sequential(
                nn.Linear(H * len(self.config.CONDITION_KEYS), H),
                nn.Sigmoid()
            )
        elif self.merge_strategy == 'film':
            # ESM을 조건으로, Sequence를 메인으로 가정 (순서 중요)
            self.film_proj = nn.Linear(H, 2 * H) # gamma와 beta 생성
        # 'sum'은 별도 레이어 필요 없음


        # --- 3. 피처 디코더 (BERT Decoder with Cross-Attention) ---
        decoder_cfg = BertConfig(
            hidden_size=H,
            num_hidden_layers=self.config.MODEL_ARCH["n_layers"],
            num_attention_heads=self.config.MODEL_ARCH["n_heads"],
            intermediate_size=H * 4,
            max_position_embeddings=self.max_len,
            is_decoder=True,  # 디코더 모드 활성화
            add_cross_attention=True, # 크로스 어텐션 레이어 추가
            _attn_implementation="eager",  # attention implementation 설정
        )
        self.feature_embeddings = BertEmbeddingsSimple(decoder_cfg)
        self.feature_decoder = BertEncoder(decoder_cfg)

        # --- 4. 출력 헤드 ---
        self.head = AnglesPredictor(H, self.feature_input_dim)

    def forward(self, x_t, c, t, attn_mask=None, position_ids=None):
        # 1. 조건 인코딩 (Key, Value 생성)
        # 각 조건 텐서를 별도의 프로젝션 레이어에 통과시킨 후, 그 결과 임베딩들을 합산합니다.
        # 이를 통해 모델이 각 조건의 출처를 구분하여 학습할 수 있습니다.
        projected_conditions = []
        # config에 정의된 순서대로 조건을 처리하여 일관성 보장
        for key in self.config.CONDITION_KEYS:
            if key in c:
                projected_conditions.append(self.cond_proj_layers[key](c[key]))
        
        if not projected_conditions:
            raise ValueError("No valid conditions found in input dict 'c'. Check CONDITION_KEYS in config.")
            
        # --- 수정: 선택된 전략에 따라 조건 결합 ---
        if self.merge_strategy == 'sum':
            combined_cond_embeds = torch.stack(projected_conditions).sum(dim=0)
        elif self.merge_strategy == 'concat':
            concatenated = torch.cat(projected_conditions, dim=-1)
            combined_cond_embeds = self.merge_proj(concatenated)
        elif self.merge_strategy == 'gated':
            # config 순서상 첫번째를 주 정보, 두번째를 보조 정보로 가정
            main_info, assistant_info = projected_conditions[0], projected_conditions[1]
            gate_input = torch.cat(projected_conditions, dim=-1)
            gate_values = self.gate_proj(gate_input)
            combined_cond_embeds = gate_values * main_info + (1 - gate_values) * assistant_info
        elif self.merge_strategy == 'film':
            # config 순서상 첫번째를 메인, 두번째를 조건으로 가정 (예: seq, esm)
            main_stream, cond_stream = projected_conditions[0], projected_conditions[1]
            gamma_beta = self.film_proj(cond_stream)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            combined_cond_embeds = gamma * main_stream + beta
        else:
            raise ValueError(f"Unknown condition_merge_strategy: {self.merge_strategy}")

        cond_embeds = self.cond_ln(combined_cond_embeds)
        cond_embeds = self.cond_embeddings(cond_embeds)
        
        ext_attn_mask = None
        if attn_mask is not None:
            ext_attn_mask = attn_mask[:, None, None, :].to(x_t.dtype)
            ext_attn_mask = (1.0 - ext_attn_mask) * -10000.0

        encoder_hidden_states = self.condition_encoder(
            cond_embeds,
            attention_mask=ext_attn_mask
        )[0]

        # 2. 피처 및 시간 임베딩 (Query 생성)
        time_encoded = self.time_embed(t.view(-1)).unsqueeze(1)
        feat_embeds = self.feat_ln(self.proj_feat(x_t))
        feat_embeds = self.feature_embeddings(feat_embeds)
        query = feat_embeds + time_encoded

        # 3. 디코더 통과 (Cross-Attention 수행)
        decoder_output = self.feature_decoder(
            query,
            attention_mask=ext_attn_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=ext_attn_mask
        )[0]

        # 4. 노이즈 예측
        return self.head(decoder_output)


def get_model(model_config) -> AbstractDiffusionModel:
    """
    설정 파일(model_config)을 기반으로 적절한 모델 아키텍처를 찾아 인스턴스화하여 반환합니다.
    """
    model_type = model_config.TRAINING_PARAMS.get("model_architecture", "cac")
    if model_type == "dit":
        print("Initializing DiffusionBertDitModel...")
        return DiffusionBertDitModel(model_config)
    elif model_type == "cac":
        print("Initializing DiffusionBertCacModel...")
        return DiffusionBertCacModel(model_config)
    else:
        raise ValueError(f"Unknown model architecture specified in config: {model_type}")


# ==============================================================================
# Diffusion Trainer 클래스
# ==============================================================================
class DiffusionTrainer:
    """
    디퓨전 프로세스(노이즈 추가, 샘플링) 및 학습 손실 계산을 관리하는 클래스.
    """
    def __init__(self, model: AbstractDiffusionModel):
        self.model = model
        # 모델의 파라미터가 위치한 장치를 따라 device를 설정합니다.
        # 이를 통해 train.py와 generate_pdb.py에서 서로 다른 장치를 사용하더라도
        # DiffusionTrainer가 올바른 장치에서 텐서를 생성하도록 보장합니다.
        self.device = next(model.parameters()).device
        self.T = config.DIFFUSION_PARAMS["timesteps"]

        # DataParallel로 래핑된 경우 실제 모델 속성에 접근하기 위한 언랩
        base_model = model.module if isinstance(model, nn.DataParallel) else model

        # --------------------------------------------------------------------------
        # 노이즈 스케줄 (Cosine Schedule) 설정
        # --------------------------------------------------------------------------
        # 시간에 따라 노이즈를 얼마나 추가할지 결정하는 스케줄을 미리 계산.
        # Cosine 스케줄은 학습 초반과 후반에 노이즈를 천천히 변화시켜 학습 안정성을 높이는 경향이 있음.
        t_range = torch.arange(self.T, device=self.device, dtype=torch.float32)
        s = 0.008
        f0 = math.cos(((0.0 / self.T) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        f_t = torch.cos(((t_range / self.T) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        self.alpha_bars = (f_t / f0).clamp(min=1e-8, max=1.0) # alpha_bar_t: t 시점까지의 누적 alpha 곱
        self.alphas = torch.cat([self.alpha_bars[:1], self.alpha_bars[1:] / self.alpha_bars[:-1]])
        self.betas = (1.0 - self.alphas).clamp(min=1e-8, max=0.999) # beta_t: t 시점의 노이즈 스텝 크기

        # DataParallel 래핑과 무관하게 안전하게 속성 접근
        self.feature_input_dim = base_model.feature_input_dim
        self.max_len = getattr(base_model, "max_len", config.DATA_SPECS["max_len"])

        # --------------------------------------------------------------------------
        # 피처별 손실 가중치 텐서 생성
        # --------------------------------------------------------------------------
        # model_config에 정의된 순서와 가중치에 따라 손실 가중치 텐서를 동적으로 생성.
        # 예: bb_torsions(dim=3, w=1.0), bb_angles(dim=3, w=1.5) -> [1.0, 1.0, 1.0, 1.5, 1.5, 1.5]
        weights = []
        for key in config.FEATURE_KEYS:
            weight = config.LOSS_WEIGHTS.get(key, 1.0) # 설정 파일에 없으면 기본값 1.0
            dim = config.FEATURE_DIMS[key]
            weights.extend([weight] * dim)
        self.feature_weights = torch.tensor(weights, device=self.device)
        print(f"Using feature weights: {self.feature_weights.cpu().numpy()}")
        
    def q_sample(self, x0, t, noise=None):
        """
        Forward Process: 원본 데이터(x0)에 t 타임스텝만큼 노이즈를 추가합니다.
        수식: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1) # 각 샘플에 맞는 alpha_bar_t를 브로드캐스팅하기 위해 차원 변경
        return alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * noise, noise

    def training_loss(self, x0, c, seq_len_mask, feat_valid_mask, lengths, real_bond_lengths, cond_prob, current_epoch):
        """학습을 위한 전체 손실(피처 손실 + FAPE 손실)을 계산합니다."""
        
        # --- 1. 손실 가중치 계산 ---
        feat_loss_weight = config.RAMP_CONFIG.get("feat_loss_end_weight", 1.0)
        
        # (유지) 타임스텝 기반 FAPE 가중치: t값에 따라 가중치를 동적으로 조절합니다.
        fape_cfg = config.FAPE_LOSS_CONFIG
        fape_loss_weight = torch.tensor(0.0, device=self.device)
        if fape_cfg.get("enabled", False):
            # t는 (B,) 형태의 텐서입니다.
            B = x0.shape[0]
            t = torch.randint(0, self.T, (B,), device=self.device)
            
            min_w, max_w = fape_cfg.get("weight_schedule", [0.0, 0.0])
            # t가 클수록(노이즈 많음) 가중치가 낮아지고, t가 작을수록(복원 거의 완료) 가중치가 높아집니다.
            weights = min_w + (max_w - min_w) * ((self.T - 1 - t.float()) / (self.T - 1))
            fape_loss_weight = weights.mean() # 배치 전체의 평균 가중치를 사용
        else:
            B, L, _ = x0.shape
            t = torch.randint(0, self.T, (B,), device=self.device)

        # ramp_scale을 곱했던 기존 FAPE 가중치 계산은 삭제합니다.
        # fape_loss_weight = config.FAPE_LOSS_CONFIG.get("weight", 0.0) * ramp_scale
        
        B, L, _ = x0.shape
        # 2. 랜덤 타임스텝 t 선택 및 노이즈 주입 (Forward Process)
        x_t, noise = self.q_sample(x0, t)

        # (추가) GT 좌표(있으면 사용) 보존: cond 마스킹 전에 가져오기
        gt_coords = c.get('gt_backbone_coords', None)
        if gt_coords is not None:
            # 기대 형태: (B, max_len*3, 3) 또는 (max_len*3, 3)
            if gt_coords.dim() == 2:
                gt_coords = gt_coords.unsqueeze(0).expand(B, -1, -1)
            gt_coords = gt_coords.to(self.device)
        
        # 3. Classifier-Free Guidance를 위한 조건 마스킹
        rand_mask = (torch.rand(B, device=self.device) < cond_prob)
        c_masked = {}
        for key, val in c.items():
            if key == 'gt_backbone_coords':
                # GT 좌표는 마스킹하지 않음
                c_masked[key] = val
                continue
            c_masked[key] = val.clone()
            c_masked[key][rand_mask] = 0.0 # 배치 내 일부 샘플의 조건을 0으로 마스킹

        # 4. 모델을 통해 노이즈 예측
        eps_pred = self.model(x_t, c_masked, t, attn_mask=seq_len_mask)

        # --- 5. 피처별 손실 계산 ---
        feat_loss_num = torch.tensor(0.0, device=self.device)
        feat_loss_den = torch.tensor(0.0, device=self.device)
        per_feat_losses = {}

        # 피처 키와 해당 피처의 시작/끝 인덱스를 미리 계산
        feat_indices = {}
        current_idx = 0
        for key in config.FEATURE_KEYS:
            dim = config.FEATURE_DIMS[key]
            feat_indices[key] = (current_idx, current_idx + dim)
            current_idx += dim

        # 각 피처 차원(총 F_dim개)을 순회하며 손실을 계산하고 가중치를 적용
        for i in range(self.feature_input_dim):
            # 현재 인덱스(i)가 어떤 피처 키에 속하는지 찾기
            current_feat_key = None
            for key, (start, end) in feat_indices.items():
                if start <= i < end:
                    current_feat_key = key
                    break
            
            # 유효한 피처(패딩되거나 NaN이 아닌)에 대해서만 손실 계산
            valid_mask = feat_valid_mask[:, :, i] & seq_len_mask.bool()
            if not valid_mask.any(): continue
            
            # 피처가 각도인지 확인하고 적절한 손실 함수 선택
            is_angular = config.FEATURE_IS_ANGULAR.get(current_feat_key, False)
            
            pred_noise = eps_pred[:, :, i][valid_mask]
            true_noise = noise[:, :, i][valid_mask]

            if is_angular:
                l_i = radian_smooth_l1_loss(pred_noise, true_noise)
            else:
                l_i = F.smooth_l1_loss(pred_noise, true_noise)
            
            w_i = self.feature_weights[i] # 미리 계산된 피처별 가중치 적용
            feat_loss_num += (w_i * l_i)
            feat_loss_den += w_i
            per_feat_losses[f'feat_{i}_loss'] = l_i.item()
        
        feat_loss = feat_loss_num / (feat_loss_den + 1e-8) if feat_loss_den > 0 else torch.tensor(0.0, device=self.device)

        # --- 6. 3D 좌표 재구성 및 보조 손실 계산 ---
        # 이 블록은 FAPE 손실이 활성화된 경우에만 실행
        loss_fape = torch.tensor(0.0, device=self.device)
        raw_fape_loss = torch.tensor(0.0, device=self.device)
        fape_stats = {}

        # FAPE 손실이 활성화되어 있으면 좌표를 계산해야 함
        if fape_loss_weight > 0:
            alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
            x0_pred_rad = (x_t - (1 - alpha_bar_t).sqrt() * eps_pred) / (alpha_bar_t.sqrt() + 1e-8)
            # atan2 기반 래핑으로 경계 불연속 완화
            x0_pred_rad = torch.atan2(x0_pred_rad.sin(), x0_pred_rad.cos())
            
            # --- 동적 피처 인덱싱 ---
            # 설정 파일(config.FEATURE_KEYS)에 정의된 순서에 따라 각 피처의 시작 인덱스를 계산합니다.
            # 이를 통해 설정 파일이 변경되더라도 코드를 수정할 필요가 없습니다.
            feature_indices = {}
            current_idx = 0
            for key in config.FEATURE_KEYS:
                feature_indices[key] = slice(current_idx, current_idx + config.FEATURE_DIMS[key])
                current_idx += config.FEATURE_DIMS[key]

            def get_feature(tensor, key):
                if key not in feature_indices:
                    raise KeyError(f"Feature '{key}' not found in config.FEATURE_KEYS.")
                return tensor[:, :, feature_indices[key]]

            # 토션 각도 (phi, psi, omega) 추출
            torsions = get_feature(x0, 'bb_torsions')
            pred_torsions = get_feature(x0_pred_rad, 'bb_torsions')
            
            # 결합 각도 (N-CA-C, CA-C-N, C-N-CA) 추출
            bond_angles = get_feature(x0, 'bb_angles')
            pred_bond_angles = get_feature(x0_pred_rad, 'bb_angles')

            # --- target_coords 정의 ---
            if gt_coords is not None and gt_coords.shape[-1] == 3:
                target_coords = gt_coords.view(B, self.max_len * 3, 3)
            else:
                target_coords = nerf_build_batch(
                    phi=torsions[:, :, 0], psi=torsions[:, :, 1], omega=torsions[:, :, 2],
                    bond_angle_n_ca_c=bond_angles[:, :, 0], bond_angle_ca_c_n=bond_angles[:, :, 1], bond_angle_c_n_ca=bond_angles[:, :, 2],
                    bond_len_n_ca=real_bond_lengths[:,:,0], bond_len_ca_c=real_bond_lengths[:,:,1], bond_len_c_n=real_bond_lengths[:,:,2]
                )
            
            # --- pred_coords 정의 ---
            pred_coords = nerf_build_batch(
                phi=pred_torsions[:, :, 0], psi=pred_torsions[:, :, 1], omega=pred_torsions[:, :, 2],
                bond_angle_n_ca_c=pred_bond_angles[:, :, 0], bond_angle_ca_c_n=pred_bond_angles[:, :, 1], bond_angle_c_n_ca=pred_bond_angles[:, :, 2],
                bond_len_n_ca=real_bond_lengths[:,:,0], bond_len_ca_c=real_bond_lengths[:,:,1], bond_len_c_n=real_bond_lengths[:,:,2]
            )

            # --- FAPE 손실 계산 ---
            if fape_loss_weight > 0:
                #FAPE 내부 파라미터를 config 파일에서 직접 읽어와 상수로 고정합니다.
                use_ca_only = False
                local_k = None
                clamp_now = fape_cfg.get("clamp_end", 10.0)
                
                loss_f_out = fape_loss(
                    pred_coords=pred_coords.view(B, L * 3, 3),
                    target_coords=target_coords.view(B, L * 3, 3),
                    lengths=lengths,
                    clamp_dist=float(clamp_now),
                    use_ca_only=use_ca_only,
                    local_window_k=local_k,
                    return_stats=True
                )
                if isinstance(loss_f_out, tuple):
                    loss_f, fape_stats = loss_f_out
                else:
                    loss_f, fape_stats = loss_f_out, {}
                
                if not torch.isnan(loss_f) and not torch.isinf(loss_f) and loss_f > 0:
                    raw_fape_loss = loss_f
                    max_fape_val = config.FAPE_LOSS_CONFIG.get("clip_value", 50.0)
                    if torch.isfinite(raw_fape_loss):
                        raw_fape_loss.clamp_(max=max_fape_val)
                    loss_fape = raw_fape_loss

        # --- 7. 최종 손실 및 로그 ---
        total_loss = (feat_loss * feat_loss_weight) + (loss_fape * fape_loss_weight)
        
        loss_info = {
            'total_loss': total_loss.item(), 
            'feat_loss': feat_loss.item(),
            'feat_weight': feat_loss_weight,
            'fape_weight': fape_loss_weight,
        }
        if fape_loss_weight > 0:
            loss_info['raw_fape_loss'] = raw_fape_loss.item()
            loss_info['fape_loss'] = (loss_fape * fape_loss_weight).item()
            if 'clamp_ratio' in fape_stats:
                loss_info['fape_clamp_ratio'] = float(fape_stats['clamp_ratio'].item())
            loss_info['fape_clamp_dist'] = float(clamp_now)
            loss_info['fape_use_ca_only'] = int(use_ca_only)
            loss_info['fape_local_k'] = int(local_k) if local_k is not None else -1
        loss_info.update(per_feat_losses)
        
        return total_loss, loss_info

    @torch.no_grad()
    def sample(self, c: Dict[str, torch.Tensor], guidance_scale: float, shape: Tuple[int, int, int], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reverse Process: 노이즈로부터 원본 데이터를 복원(생성)하는 샘플링 과정.
        """
        B, _, _ = shape
        x_t = torch.randn(shape, device=self.device) # 순수 가우시안 노이즈에서 시작

        # --- 수정: 비조건부 샘플링을 위한 딕셔너리 생성 ---
        c_uncond = {key: torch.zeros_like(val) for key, val in c.items()}

        # T-1부터 0까지 타임스텝을 거꾸로 진행하며 노이즈를 점진적으로 제거
        for t_val in reversed(range(self.T)):
            t = torch.tensor([t_val] * B, device=self.device)

            # Classifier-Free Guidance: 조건부 예측과 비조건부 예측을 모두 수행
            eps_cond = self.model(x_t, c, t, attn_mask=mask)
            eps_uncond = self.model(x_t, c_uncond, t, attn_mask=mask)
            # 두 예측을 guidance_scale에 따라 조합하여 최종 노이즈 방향 결정
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # --- 추가: 샘플링 안정성을 위한 노이즈 클램핑 ---
            # 예측된 노이즈가 과도하게 커져서 샘플링 과정을 불안정하게 만드는 것을 방지합니다.
            eps = torch.clamp(eps, -10.0, 10.0)

            # DDPM 샘플링 공식에 따라 x_{t-1} 계산
            alpha_t = self.alphas[t_val]
            alpha_bar_t = self.alpha_bars[t_val]
            beta_t = self.betas[t_val]
            
            posterior_mean = (1 / alpha_t.sqrt()) * (x_t - beta_t / (1 - alpha_bar_t).sqrt() * eps)
            if t_val > 0:
                alpha_bar_t_prev = self.alpha_bars[t_val - 1]
                posterior_variance = beta_t * (1. - alpha_bar_t_prev) / (1. - alpha_bar_t)
                x_t = posterior_mean + posterior_variance.sqrt() * torch.randn_like(x_t)
            else:
                x_t = posterior_mean
        
        # 최종 생성된 각도 데이터를 -pi ~ pi 범위로 래핑하여 반환
        return wrap_pi(x_t)
