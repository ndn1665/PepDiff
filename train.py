import os
# TensorFlow의 로그 레벨을 설정하여 INFO 및 WARNING 메시지를 비활성화합니다.
# '2'는 INFO와 WARNING을 모두 숨기는 레벨입니다.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import math
import multiprocessing
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt

# 분리된 모델 및 설정 파일 임포트
import model_config as config
from model import PeptideDataset, DiffusionTrainer, get_model
# 새로 추가된 분석 유틸리티 임포트 (경로 수정)
from utils.analysis_utils import run_validation_metrics

# ==============================================================================
# 학습 유틸리티
# ==============================================================================
class EarlyStopping:
    """
    검증 손실이 더 이상 개선되지 않을 때 학습을 조기 종료시키는 클래스.
    Overfitting을 방지하고 최적의 모델을 저장하는 데 사용됩니다.
    """
    def __init__(self, patience=30, min_delta=1e-5, save_path=None):
        """
        Args:
            patience (int): 검증 손실이 개선되지 않아도 참을 에폭 수.
            min_delta (float): 개선되었다고 판단할 최소 손실 감소량.
            save_path (str): 최적 모델의 state_dict가 저장될 경로.
        """
        self.patience, self.min_delta, self.save_path = patience, min_delta, save_path
        self.best_loss, self.counter = float('inf'), 0

    def step(self, val_loss, epoch, model, optimizer, scheduler):
        """매 에폭의 검증 단계가 끝난 후 호출됩니다."""
        # 현재 검증 손실이 기록된 최적 손실보다 유의미하게 낮은 경우
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0 # 카운터 초기화
            # 최적 모델 파라미터를 파일로 저장
            if self.save_path and model:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_loss': self.best_loss,
                }
                torch.save(checkpoint, self.save_path)
                print(f"   -> Saved checkpoint for epoch {epoch}. Best loss: {self.best_loss:.4f}")
        else:
            # 개선되지 않은 경우 카운터 증가
            self.counter += 1
        
        # 카운터가 설정된 patience를 초과하면 조기 종료 신호(True) 반환
        return self.counter >= self.patience

def build_warmup_cosine_scheduler(optimizer, total_epochs, warmup_ratio=0.05):
    """
    간단한 Warmup 후 Cosine Decay를 적용하는 학습률 스케줄러를 생성합니다.
    """
    warmup_epochs = max(1, int(total_epochs * warmup_ratio))
    
    def lr_lambda(epoch_idx):
        if epoch_idx < warmup_epochs:
            # Warmup: 0 -> 1
            return float(epoch_idx + 1) / float(warmup_epochs)
        else:
            # Cosine Decay: 1 -> 0
            progress = (epoch_idx - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def_wandb = False
try:
    import wandb
except ImportError:
    def_wandb = False


# ==============================================================================
# 메인 학습 함수
# ==============================================================================
def train_diffusion():
    """전체 학습 파이프라인을 관리하고 실행하는 메인 함수."""
    # --- 1. 기본 설정 불러오기 ---
    device = config.MISC["device"]
    B = config.TRAINING_PARAMS["batch_size"]
    print(f"Using device: {device}")

    # --- 1a. Wandb 초기화 (선택적) ---
    if config.MISC.get("wandb_project"):
        try:
            import wandb
            wandb.init(
                project=config.MISC["wandb_project"],
                entity=config.MISC.get("wandb_entity"),
                name=config.MISC.get("wandb_run_name"),
                config=config.to_dict() # 모든 설정을 wandb에 기록
            )
            def_wandb = True
        except ImportError:
            print("wandb not installed, skipping log. Please run 'pip install wandb'")
            def_wandb = False
    else:
        def_wandb = False

    # --- 2. 데이터셋 준비 ---
    # 데이터셋 파일 존재 여부 확인
    train_path = config.PATHS["train_dataset"]
    test_path = config.PATHS["test_dataset"]
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"🚨 Error: Dataset file not found. Checked paths: {train_path}, {test_path}")
        return None, None

    # PeptideDataset 클래스를 사용하여 데이터셋 객체 생성
    train_ds = PeptideDataset(train_path)
    val_ds = PeptideDataset(test_path)
    # DataLoader를 사용하여 배치 단위로 데이터를 로드할 수 있도록 설정
    train_loader = DataLoader(train_ds, batch_size=B, shuffle=True, num_workers=config.MISC["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=B, shuffle=False, num_workers=config.MISC["num_workers"], pin_memory=True, drop_last=True)

    # --- 3. 모델 및 EarlyStopping 초기화 ---
    model = get_model(config)
    model.to(device)  # 1. 모델을 먼저 GPU로 보냅니다.

    # EarlyStopping 객체를 먼저 생성합니다.
    early_stopper = EarlyStopping(
        patience=config.TRAINING_PARAMS["early_stop_patience"],
        save_path=config.PATHS["save_path"]
    )

    # --- 추가: 체크포인트 로드 ---
    start_epoch = 0
    save_path = config.PATHS["save_path"]
    optimizer_state_dict, scheduler_state_dict = None, None

    if os.path.exists(save_path):
        print(f"Resuming training from checkpoint: {save_path}")
        try:
            checkpoint = torch.load(save_path, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '')
                    new_state_dict[name] = v
                
                model.load_state_dict(new_state_dict)
                
                optimizer_state_dict = checkpoint.get('optimizer_state_dict')
                scheduler_state_dict = checkpoint.get('scheduler_state_dict')
                start_epoch = checkpoint['epoch']
                
                # 생성된 early_stopper 인스턴스의 best_loss 값을 업데이트합니다.
                early_stopper.best_loss = checkpoint.get('best_loss', float('inf'))

                print(f"   -> Resuming from epoch {start_epoch + 1}. Best validation loss was {early_stopper.best_loss:.4f}.")
            else:
                model.load_state_dict(checkpoint)
                print("   -> Loaded an old format checkpoint (only model state_dict). Starting from epoch 0.")

        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
    else:
        print("No checkpoint found, starting training from scratch.")

    # --- 3a. Multi-GPU 설정 ---
    if config.MISC.get("multi_gpu", False) and torch.cuda.device_count() > 1:
        print(f"🚀 Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    # 4. 모델이 GPU에 위치한 후 옵티마이저와 스케줄러를 생성합니다.
    optimizer = optim.AdamW(model.parameters(), lr=config.TRAINING_PARAMS["learning_rate"], weight_decay=0.01)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_epochs=config.TRAINING_PARAMS["epochs"],
        warmup_ratio=config.TRAINING_PARAMS["warmup_ratio"]
    )

    # 체크포인트에서 읽어온 상태가 있으면 로드합니다.
    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)
    if scheduler_state_dict:
        scheduler.load_state_dict(scheduler_state_dict)
    
    # 5. 모델이 최종 디바이스에 위치한 후 DiffusionTrainer를 초기화합니다.
    diffusion = DiffusionTrainer(model) 

    train_losses, val_losses = [], []

    # --- 4. 학습 루프 시작 ---
    print("🚀 Starting training...")
    for epoch in range(start_epoch, config.TRAINING_PARAMS["epochs"]):
        model.train() # 모델을 학습 모드로 설정 (Dropout 등 활성화)
        total_train_loss = 0.0

        # tqdm을 사용하여 학습 진행 상황을 시각적으로 표시
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_data in progress_bar:
            # --- 4a. 데이터 준비 (수정됨) ---
            # PeptideDataset의 __getitem__ 반환 형식에 맞게 언패킹 (조건이 dict 형태로 반환됨)
            x0_cpu, cond_dict_cpu, seq_mask_cpu, feat_mask_cpu, lengths_cpu, real_bond_lengths_cpu = batch_data
            
            x0 = x0_cpu.to(device, non_blocking=True)
            cond_dict = {k: v.to(device, non_blocking=True) for k, v in cond_dict_cpu.items()}
            seq_mask = seq_mask_cpu.to(device, non_blocking=True)
            feat_mask = feat_mask_cpu.to(device, non_blocking=True)
            lengths = lengths_cpu.to(device, non_blocking=True)
            real_bond_lengths = real_bond_lengths_cpu.to(device, non_blocking=True)
            
            # --- 4b. 손실 계산 및 역전파 (수정됨) ---
            # model.py의 training_loss 시그니처와 인자 순서를 정확하게 일치시킴
            total_loss, loss_info = diffusion.training_loss(
                x0=x0,
                c=cond_dict, # 딕셔너리 형태의 조건 전달
                seq_len_mask=seq_mask,
                feat_valid_mask=feat_mask,
                lengths=lengths,
                real_bond_lengths=real_bond_lengths,
                cond_prob=config.TRAINING_PARAMS["condition_dropout"],
                current_epoch=epoch + 1 # 현재 에폭 번호 전달
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            # 그래디언트 클리핑 추가 (학습 안정성 향상)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            # --- 5. 로깅 ---
            total_train_loss += loss_info['total_loss']
            
            # 로그에 표시할 FAPE 손실 문자열 포맷팅
            fape_log_str = f"{loss_info.get('raw_fape_loss', 0.0):.4f} (w: {loss_info.get('fape_loss', 0.0):.4f})"
            feat_weight = loss_info.get('feat_weight', 1.0)
            fape_weight = loss_info.get('fape_weight', 0.0)
            ca_only_flag = int(loss_info.get('fape_use_ca_only', 0))
            
            progress_bar.set_postfix({
                "loss": loss_info['total_loss'], 
                "feat_loss": loss_info['feat_loss'],
                "fape_loss(raw|w)": fape_log_str,
                "feat_w": feat_weight,
                "fape_w": fape_weight,
                "ca_only": ca_only_flag
            })
            if def_wandb:
                # 스텝별로 더 상세한 로그를 wandb에 기록
                log_data = {
                    "step_loss": loss_info['total_loss'],
                    "step_feat_loss": loss_info['feat_loss'],
                    "step_raw_fape_loss": loss_info.get('raw_fape_loss', 0.0),
                    "step_fape_loss_weighted": loss_info.get('fape_loss', 0.0),
                    "feat_loss_weight": feat_weight,
                    "fape_loss_weight": fape_weight,
                    "fape_use_ca_only": ca_only_flag,
                    "fape_local_k": loss_info.get('fape_local_k', -1),
                    "learning_rate": scheduler.get_last_lr()[0]
                }
                wandb.log(log_data)
            
        # 에폭 평균 손실 계산
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- Validation ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_data in val_loader:
                # PeptideDataset의 __getitem__ 반환 형식에 맞게 언패킹
                x0_cpu, cond_dict_cpu, seq_mask_cpu, feat_mask_cpu, lengths_cpu, real_bond_lengths_cpu = batch_data
                x0 = x0_cpu.to(device)
                cond_dict = {k: v.to(device) for k, v in cond_dict_cpu.items()}
                seq_mask = seq_mask_cpu.to(device)
                feat_mask = feat_mask_cpu.to(device)
                lengths = lengths_cpu.to(device)
                real_bond_lengths = real_bond_lengths_cpu.to(device)
                
                # 검증 시에는 ramp_scale과 cond_prob을 고정값으로 사용
                _, loss_info = diffusion.training_loss(
                    x0=x0,
                    c=cond_dict, # 딕셔너리 형태의 조건 전달
                    seq_len_mask=seq_mask,
                    feat_valid_mask=feat_mask,
                    lengths=lengths,
                    real_bond_lengths=real_bond_lengths,
                    cond_prob=1.0, # 항상 컨디션 사용
                    current_epoch=epoch + 1 # 검증 시에도 에폭 전달 (가중치는 내부에서 결정)
                )
                total_val_loss += loss_info['total_loss']

        scheduler.step()
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        current_lr = scheduler.get_last_lr()[0]
        val_losses.append(avg_val_loss)
        
        # --- 추가된 검증 메트릭 실행 ---
        avg_rmsd = -1.0 # 기본값
        # --- 수정: 검증 메트릭 실행 조건을 FAPE Loss 계산 시작 시점과 동기화 ---
        is_last_epoch = (epoch + 1) == config.TRAINING_PARAMS["epochs"]
        
        # (수정) 매 100 에포크마다 또는 마지막 에포크에 메트릭 실행
        should_run_metrics = ((epoch + 1) % 100 == 0)

        if should_run_metrics or is_last_epoch:
            print(f"\nRunning validation metrics (Backbone RMSD, Ramachandran) for epoch {epoch+1}...")
            avg_rmsd = run_validation_metrics(
                model, diffusion, val_loader, epoch + 1
            )
        
        # 에폭 결과 출력 (RMSD 추가)
        print(
            f"[Epoch {epoch+1}/{config.TRAINING_PARAMS['epochs']}] "
            f"LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val RMSD: {avg_rmsd:.4f}"
        )

        # --- Wandb 에폭 로그 ---
        if def_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
                "avg_val_rmsd": avg_rmsd,
                "learning_rate_epoch": current_lr
            })

        # --- 6. 조기 종료 확인 ---
        if early_stopper.step(avg_val_loss, epoch + 1, model, optimizer, scheduler):
            print(f"Early stopping at epoch {epoch+1}. Best Val Loss: {early_stopper.best_loss:.4f}")
            break

    # --- 7. 학습 종료 후 결과 저장 ---
    print("✅ Training finished.")
    # 손실 곡선 그래프를 이미지 파일로 저장
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Diffusion Training Loss Curve')
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig("diffusion_training_loss_curve.png", dpi=150)
        print("Saved loss curve to diffusion_training_loss_curve.png")
    except Exception as e:
        print(f"Could not save loss plot: {e}")

    return model, diffusion

# ==============================================================================
# 스크립트 실행 지점
# ==============================================================================
if __name__ == '__main__':
    # 멀티프로세싱 시작 방법 설정 (Windows/macOS 호환성 문제 방지)
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # 메인 학습 함수 호출
    trained_model, diffusion_trainer = train_diffusion()
