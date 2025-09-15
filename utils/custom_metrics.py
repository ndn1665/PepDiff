"""
Some custom metrics
"""
import functools
import multiprocessing
import logging

import numpy as np
from scipy import stats

import torch
import torch.nn.functional as F
import math

from torch.utils.data import Dataset


def kl_from_empirical(
    u: np.ndarray, v: np.ndarray, nbins: int = 100, pseudocount: bool = False
) -> float:
    """
    Compute the KL divergence between two empirical distributions u and v.

    Discretizes the u and v distributions using nbins bins
    """
    min_val = min(np.min(u), np.min(v))
    max_val = max(np.max(u), np.max(v))
    logging.debug(f"Creating {nbins} bins between {min_val} - {max_val}")

    bins = np.linspace(min_val, max_val, nbins + 1)
    if pseudocount:
        u = np.concatenate((u, bins))
        v = np.concatenate((v, bins))
    u_hist, _u_bin_edges = np.histogram(u, bins=bins, density=True)
    v_hist, _v_bin_edges = np.histogram(v, bins=bins, density=True)

    # Compute KL divergence
    # https://stackoverflow.com/questions/63369974/3-functions-for-computing-relative-entropy-in-scipy-whats-the-difference
    kl = stats.entropy(u_hist, v_hist)
    return kl

def _kl_helper(t: int, dset: Dataset) -> np.ndarray:
    """
    Compute the KL divergence for each feature at timestep t
    Returns an array of size (n_features,) corresponding to KL divergence
    dset should be a compatible Dataset like KLValidationDataset
    """
    assert hasattr(dset, "feature_names")
    assert hasattr(dset, "sample_noise")

    # 모든 유효한 위치의 값을 수집
    values_by_feature = [[] for _ in range(dset.feature_dim)]
    
    for i in range(len(dset)):
        # KLValidationDataset의 __getitem__ 호출
        sample = dset.__getitem__(i, use_t_val=t)
        
        # 마스크를 사용하여 유효한 데이터만 추출
        valid_data = sample["corrupted"][sample["attn_mask"]].cpu().numpy() # (valid_len, F)
        
        if valid_data.shape[0] > 0:
            for feature_idx in range(valid_data.shape[1]):
                values_by_feature[feature_idx].extend(valid_data[:, feature_idx])

    # 리스트를 numpy 배열로 변환
    values_list = [np.array(vals) for vals in values_by_feature]
    
    kl_values = []
    for i, values_for_feature in enumerate(values_list):
        if len(values_for_feature) < 2: # 히스토그램을 만들기에 데이터가 부족
            kl_values.append(np.nan)
            continue
            
        # 비교할 노이즈 샘플링
        noise_for_feature = dset.sample_noise(values_for_feature)
        
        # KL 발산 계산
        kl = kl_from_empirical(values_for_feature, noise_for_feature, nbins=100)
        kl_values.append(kl)
        
    return np.array(kl_values)


def kl_from_dset(dset: Dataset, single_thread: bool = False) -> np.ndarray:
    """
    For each timestep in the dataset, compute the KL divergence across each feature
    """
    assert hasattr(dset, "timesteps")
    pfunc = functools.partial(_kl_helper, dset=dset)
    if single_thread:
        kl_values = [pfunc(t) for t in range(dset.timesteps)]
    else:
        logging.info(
            f"Computing KL divergence for {dset.timesteps} timesteps using {multiprocessing.cpu_count()} workers"
        )
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        kl_values = pool.map(pfunc, range(dset.timesteps))
        pool.close()
        pool.join()
    return np.array(kl_values)


def wrapped_mean(x: np.ndarray, axis=None) -> float:
    """
    Wrap the mean function about [-pi, pi]
    """
    # https://rosettacode.org/wiki/Averages/Mean_angle
    sin_x = np.sin(x)
    cos_x = np.cos(x)

    retval = np.arctan2(np.nanmean(sin_x, axis=axis), np.nanmean(cos_x, axis=axis))
    return retval


def radian_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    두 각도(라디안) 사이의 최단 거리를 계산하는 L1 손실.
    주기성을 고려합니다.
    """
    # 1. 두 각도의 차이를 계산합니다.
    diff = pred - target
    
    # 2. 차이를 -pi ~ +pi 범위로 래핑(wrapping)합니다.
    # 예: +3.13 -> -3.15 (pi 근처), -3.13 -> +3.15 (-pi 근처)
    # (diff + pi) % (2*pi) - pi 트릭을 사용합니다.
    wrapped_diff = (diff + math.pi) % (2 * math.pi) - math.pi
    
    # 3. 래핑된 차이의 절댓값을 취합니다.
    return torch.abs(wrapped_diff)

def radian_smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Smooth L1 loss V2, but for radians.
    Wraps the prediction and target to be within [-pi, pi].
    See https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
    """
    assert pred.shape == target.shape
    d = pred - target
    # Wrap to [-pi, pi]
    d = torch.atan2(torch.sin(d), torch.cos(d))

    # Apply smooth l1 loss
    loss = torch.where(
        torch.abs(d) < beta,
        0.5 * (d**2) / beta,
        torch.abs(d) - 0.5 * beta,
    )
    return loss.mean()
