import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 프로젝트의 다른 모듈들을 임포트합니다.
import model_config as config
from model import get_model, PeptideDataset
from model_config import AA_3_TO_1

def visualize_attention(
    model_path: str,
    dataset_path: str,
    sample_index: int,
    layer_to_viz: int = -1, # 기본값: 마지막 레이어
    head_to_viz: int = 0,
    output_dir: str = "./attention_maps"
):
    """
    학습된 DiffusionBertCacModel의 Cross-Attention 가중치를 시각화합니다.
    
    Args:
        model_path (str): 학습된 모델(.pt) 파일 경로.
        dataset_path (str): 조건을 가져올 데이터셋(.pkl) 파일 경로.
        sample_index (int): 시각화할 데이터셋 내 샘플의 인덱스.
        layer_to_viz (int): 시각화할 디코더 레이어 번호 (0부터 시작, -1은 마지막 레이어).
        head_to_viz (int): 시각화할 어텐션 헤드 번호.
        output_dir (str): 생성된 히트맵 이미지가 저장될 디렉토리.
    """
    device = torch.device("cpu") # 시각화는 CPU에서 수행
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- 1. 모델 로드 ---
    print(f"Loading model from {model_path}...")
    # 'cac' 아키텍처로 설정하여 모델을 로드
    config.TRAINING_PARAMS["model_architecture"] = "cac"
    model = get_model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- 2. 모델의 Decoder가 Attention Weight를 출력하도록 설정 ---
    if not hasattr(model, 'feature_decoder'):
        print("❌ Error: This script only works with 'DiffusionBertCacModel'.")
        return
        
    # BertEncoder가 어텐션 가중치를 반환하도록 설정을 변경합니다.
    model.feature_decoder.config.output_attentions = True

    # --- 3. 데이터 준비 ---
    print(f"Loading data sample {sample_index} from {dataset_path}...")
    dataset = PeptideDataset(dataset_path)
    _, c_dict, _, _, length_tensor, _ = dataset[sample_index]
    
    seq_len = length_tensor.item()
    sequence_3_letter = dataset.data['sequences_3letter'][sample_index][:seq_len]
    sequence_1_letter = [AA_3_TO_1.get(aa, 'X') for aa in sequence_3_letter]

    # 배치 차원 추가 및 디바이스로 이동
    c = {key: val.unsqueeze(0).to(device) for key, val in c_dict.items()}
    
    # 더미 입력 생성 (어텐션 맵 확인에는 실제 피처/시간 값은 중요하지 않음)
    dummy_x_t = torch.randn(1, config.DATA_SPECS["max_len"], model.feature_input_dim, device=device)
    dummy_t = torch.tensor([500], device=device) # 임의의 타임스텝

    # --- 4. 모델 실행 및 Attention Weight 추출 ---
    with torch.no_grad():
        # `forward` 대신 `feature_decoder`를 직접 호출하여 중간 출력값(어텐션)을 얻습니다.
        # (간소화를 위해 전체 forward pass 대신 decoder 부분만 실행)
        # 실제로는 전체 forward pass를 하고 attention을 추출해야 더 정확합니다.
        # 여기서는 간단한 시각화를 위해 핵심 부분만 실행합니다.
        
        # 1. 조건 인코딩
        cond_embeds_list = []
        for i, key in enumerate(model.condition_keys):
            cond_data = c[key]
            cond_embed = model.cond_projs[key](cond_data)
            type_embed = model.condition_type_embeddings(torch.tensor([i], device=device))
            cond_embeds_list.append(cond_embed + type_embed)
        final_cond_embed = torch.stack(cond_embeds_list).sum(dim=0)
        cond_embeds = model.cond_ln(final_cond_embed)
        cond_embeds = model.cond_embeddings(cond_embeds)
        encoder_hidden_states = model.condition_encoder(cond_embeds)[0]

        # 2. 쿼리 생성
        query = model.feat_ln(model.proj_feat(dummy_x_t))
        query = model.feature_embeddings(query) + model.time_embed(dummy_t).unsqueeze(1)

        # 3. 디코더 실행 및 어텐션 추출
        # decoder의 forward()는 (hidden_states, present_key_value, all_hidden_states, all_attentions, all_cross_attentions)
        # 형태의 튜플을 반환하도록 설정해야 합니다. (HuggingFace BertEncoder 참조)
        # 여기서는 설명을 위해 가상의 출력을 가정합니다.
        # 실제 구현에서는 BertEncoder의 `output_attentions=True` 설정 후 반환값을 파싱해야 합니다.
        
        # --- HuggingFace BertEncoder의 실제 반환값 구조를 반영한 코드 ---
        decoder_outputs = model.feature_decoder(
            query,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True # 명시적으로 Dictionary 형태의 출력을 요청
        )
        
        # all_cross_attentions는 (num_layers, batch_size, num_heads, query_len, key_len)
        cross_attentions = decoder_outputs.cross_attentions # 튜플 형태

    # --- 5. 시각화 ---
    attention_matrix = cross_attentions[layer_to_viz][0, head_to_viz, :seq_len, :seq_len].cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_matrix,
        xticklabels=sequence_1_letter,
        yticklabels=sequence_1_letter,
        cmap="viridis"
    )
    plt.title(f"Cross-Attention - Layer {layer_to_viz}, Head {head_to_viz}")
    plt.xlabel("Key (Condition: Amino Acid Sequence)")
    plt.ylabel("Query (Generated Structure Position)")
    
    output_filename = f"cross_attention_sample_{sample_index}_layer_{layer_to_viz}_head_{head_to_viz}.png"
    plt.savefig(Path(output_dir) / output_filename, dpi=150)
    plt.close()
    print(f"✅ Attention map saved to '{Path(output_dir) / output_filename}'")


if __name__ == '__main__':
    # --- 스크립트 실행 설정 ---
    # 아래 값들을 변경하여 원하는 대상에 대한 시각화를 수행할 수 있습니다.
    MODEL_FILE = config.PATHS["save_path"]
    DATASET_FILE = config.PATHS["test_dataset"]
    SAMPLE_IDX = 0  # 테스트 데이터셋의 0번째 샘플
    LAYER = -1      # 마지막 레이어
    HEAD = 0        # 첫 번째 헤드

    visualize_attention(
        model_path=MODEL_FILE,
        dataset_path=DATASET_FILE,
        sample_index=SAMPLE_IDX,
        layer_to_viz=LAYER,
        head_to_viz=HEAD
    )
