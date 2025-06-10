import traceback
import torch.nn.functional as F
import math
import torch


def si_snr_loss(estimate, target, epsilon=1e-8):
    # estimate: (batch_size, num_samples)
    # target: (batch_size, num_samples)
    # print(f"estimate shape: {estimate.shape}, target shape: {target.shape}")
    # Ensure shapes are compatible, if estimate is (batch, channels, samples) and target is (batch, samples)
    if estimate.ndim == 3 and estimate.shape[1] == 1: # Assuming single channel output from model
        estimate = estimate.squeeze(1)

    if estimate.shape != target.shape:
        # This might happen if the model outputs a different number of samples than the target.
        # You might need to truncate or pad one of them. For now, let's assume they should match.
        # Or, if one is (batch, 1, samples) and other is (batch, samples)
        min_len = min(estimate.shape[-1], target.shape[-1])
        estimate = estimate[..., :min_len]
        target = target[..., :min_len]
        # print(f"Adjusted shapes: estimate: {estimate.shape}, target: {target.shape}")


    # zero-mean
    target_zm = target - torch.mean(target, dim=-1, keepdim=True)
    estimate_zm = estimate - torch.mean(estimate, dim=-1, keepdim=True)

    s_target = torch.sum(target_zm * estimate_zm, dim=-1, keepdim=True) * target_zm / (torch.sum(target_zm**2, dim=-1, keepdim=True) + epsilon)
    e_noise = estimate_zm - s_target

    si_snr = torch.sum(s_target**2, dim=-1) / (torch.sum(e_noise**2, dim=-1) + epsilon)
    si_snr_db = -10 * torch.log10(si_snr + epsilon) # Negative because we want to minimize it
    return torch.mean(si_snr_db)

def cosine_similarity_loss(estimate, target, epsilon=1e-8):
    # estimate: (batch_size, num_samples)
    # target: (batch_size, num_samples)
    if estimate.ndim == 3 and estimate.shape[1] == 1: # Assuming single channel output from model
        estimate = estimate.squeeze(1)
    
    if estimate.shape != target.shape:
        min_len = min(estimate.shape[-1], target.shape[-1])
        estimate = estimate[..., :min_len]
        target = target[..., :min_len]

    cos_sim = F.cosine_similarity(estimate, target, dim=-1, eps=epsilon)
    return -torch.mean(cos_sim) # Negative because we want to minimize it