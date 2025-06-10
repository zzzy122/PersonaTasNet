import os
import math
import torch
import torch.nn as nn
import traceback
import torch.nn.functional as F
from tqdm import tqdm
from .adabound import AdaBound
from .audio import Audio
from .evaluation import validate,validate_tas
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder
from tasnet_model.conv_tasnet import ConvTasNet
from .uitls import si_snr_loss, cosine_similarity_loss
import matplotlib.pyplot as plt
import numpy as np


def train(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str):
    # load embedder
    embedder_pt = torch.load(args.embedder_path)
    embedder = SpeechEmbedder(hp).cuda()
    embedder.load_state_dict(embedder_pt)
    embedder.eval()

    audio = Audio(hp)
    # model = VoiceFilter(hp).cuda()


    M, N, L, T = 2, 256, 4, 48000

    K = 2*T//L-1
    B, H, P, X, R, C, norm_type, causal = 2, 3, 3, 3, 2, 1, "gLN", False 
    # test Conv-TasNet
    model = ConvTasNet(N, L, B, H, P, X, R, C, norm_type=norm_type).cuda()

    if hp.train.optimizer == 'adabound':
        optimizer = AdaBound(model.parameters(),
                             lr=hp.train.adabound.initial,
                             final_lr=hp.train.adabound.final)
    elif hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hp.train.adam)
    else:
        raise Exception("%s optimizer not supported" % hp.train.optimizer)

    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']

        # will use new given hparams.
        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint.")
    else:
        logger.info("Starting new training run")
    
    metrics = {
        'train': {'loss': [], 'si_snr': [], 'sdr': [], 'pesq': []},
        'val': {'loss': [], 'si_snr': [], 'sdr': [], 'pesq': []}
    }
    
    # 获取总epoch数
    epochs = hp.train.epochs
    
 
    # criterion = nn.MSELoss() # Old loss
    alpha = hp.train.loss_alpha if hasattr(hp.train, 'loss_alpha') else 0.5 # Weight for SI-SNR, default 0.5
    beta = hp.train.loss_beta if hasattr(hp.train, 'loss_beta') else 0.5    # Weight for Cosine Sim, default 0.5


    # 添加进度条
    for epoch in range(1, epochs + 1):
        epoch_train_loss = 0.0
        epoch_train_si_snr = 0.0
        epoch_train_sdr = 0.0
        epoch_train_pesq = 0.0
        train_count = 0

        model.train()
        with tqdm(total=len(trainloader), desc=f'Epoch {epoch}/{epochs}') as pbar:
            for i, (dvec_mels, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase) in enumerate(trainloader):

                mixed_wav = mixed_wav.cuda()
                target_wav = target_wav.cuda()

                dvec_list = list()
                for mel in dvec_mels:
                    mel = mel.cuda()
                    dvec = embedder(mel)
                    dvec_list.append(dvec)
                dvec = torch.stack(dvec_list, dim=0)
                dvec = dvec.detach()

                output = model(mixed_wav, dvec)


                loss_si_snr = si_snr_loss(output, target_wav)
                loss_cos_sim = cosine_similarity_loss(output, target_wav)
                loss = alpha * loss_si_snr + beta * loss_cos_sim


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

        
         # 计算指标
                with torch.no_grad():
                    # 计算SI-SNR
                    si_snr_val = compute_si_snr(output, target_wav).item()
                    # 计算SDR和PESQ（这里需要实现实际计算函数）
                    sdr_val = calculate_sdr(output, target_wav)  # 需要实现
                    pesq_val = calculate_pesq(output, target_wav, hp.audio.sample_rate)  # 需要实现
                
                # 累加指标
                epoch_train_loss += loss.item()
                epoch_train_si_snr += si_snr_val
                epoch_train_sdr += sdr_val
                epoch_train_pesq += pesq_val
                train_count += 1
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'SI-SNR': f'{si_snr_val:.2f}'
                })
                
                # ... 保存检查点和验证的代码调整到epoch结束后 ...
        
        # 计算并记录训练指标
        avg_train_loss = epoch_train_loss / train_count
        avg_train_si_snr = epoch_train_si_snr / train_count
        avg_train_sdr = epoch_train_sdr / train_count
        avg_train_pesq = epoch_train_pesq / train_count
        
        metrics['train']['loss'].append(avg_train_loss)
        metrics['train']['si_snr'].append(avg_train_si_snr)
        metrics['train']['sdr'].append(avg_train_sdr)
        metrics['train']['pesq'].append(avg_train_pesq)
        
        # 验证
        val_metrics = validate_tas(args, audio, model, embedder, testloader, writer, epoch * len(trainloader), hp)
        
        # 记录验证指标
        metrics['val']['loss'].append(val_metrics['loss'])
        metrics['val']['si_snr'].append(val_metrics['si_snr'])
        metrics['val']['sdr'].append(val_metrics['sdr'])
        metrics['val']['pesq'].append(val_metrics['pesq'])
        
        # 打印epoch结果
        logger.info(f"\nEpoch {epoch}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | SI-SNR: {avg_train_si_snr:.2f} | "
                    f"SDR: {avg_train_sdr:.2f} | PESQ: {avg_train_pesq:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | Val SI-SNR: {val_metrics['si_snr']:.2f} | "
                    f"Val SDR: {val_metrics['sdr']:.2f} | Val PESQ: {val_metrics['pesq']:.4f}")
        
        # 保存检查点（每个epoch保存一次）
        save_path = os.path.join(pt_dir, f'chkpt_{epoch}.pt')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'hp_str': hp_str,
        }, save_path)
        logger.info(f"Saved checkpoint to: {save_path}")
    
    # 训练完成后绘制指标图表
    plot_metrics(metrics, pt_dir)
    logger.info(f"Training completed. Metrics plots saved to {pt_dir}")
    
    return metrics

def validate_tas(args, audio, model, embedder, testloader, writer, step, hp):
    model.eval()
    embedder.eval()
    
    total_loss = 0.0
    total_si_snr = 0.0
    total_sdr = 0.0
    total_pesq = 0.0
    count = 0
    
    with torch.no_grad():
         for batch in testloader:
            dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase = batch[0]
             
            dvec_mel = dvec_mel.cuda()
            # target_mag = target_mag.unsqueeze(0).cuda()
            # mixed_mag = mixed_mag.unsqueeze(0).cuda()
            mixed_wav = torch.tensor(mixed_wav).cuda().unsqueeze(0)
            target_wav = torch.tensor(target_wav).cuda().unsqueeze(0)     

            dvec = embedder(dvec_mel)
            dvec = dvec.unsqueeze(0)
            output = model(mixed_wav, dvec)
            
            
            # 计算损失
            loss_si_snr = si_snr_loss(output, target_wav)
            loss_cos_sim = cosine_similarity_loss(output, target_wav)
            loss = loss_si_snr + 0.1 * loss_cos_sim
            
            # 计算指标
            si_snr_val = compute_si_snr(output, target_wav).item()
            sdr_val = calculate_sdr(output, target_wav)  # 需要实现
            pesq_val = calculate_pesq(output, target_wav, hp.audio.sample_rate)  # 需要实现
            
            # 累加指标
            total_loss += loss.item()
            total_si_snr += si_snr_val
            total_sdr += sdr_val
            total_pesq += pesq_val
            count += 1
    
    # 计算平均指标
    avg_loss = total_loss / count
    avg_si_snr = total_si_snr / count
    avg_sdr = total_sdr / count
    avg_pesq = total_pesq / count
    
    # 记录到TensorBoard
    writer.add_scalar('Val/loss', avg_loss, step)
    writer.add_scalar('Val/SI-SNR', avg_si_snr, step)
    writer.add_scalar('Val/SDR', avg_sdr, step)
    writer.add_scalar('Val/PESQ', avg_pesq, step)
    
    return {
        'loss': avg_loss,
        'si_snr': avg_si_snr,
        'sdr': avg_sdr,
        'pesq': avg_pesq
    }

def plot_metrics(metrics, output_dir):
    epochs = list(range(1, len(metrics['train']['loss']) + 1))
    
    # 创建指标图表
    plt.figure(figsize=(15, 12))
    
    # Loss图表
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics['train']['loss'], 'b-', label='Train')
    plt.plot(epochs, metrics['val']['loss'], 'r-', label='Validation')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # SI-SNR图表
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics['train']['si_snr'], 'b-', label='Train')
    plt.plot(epochs, metrics['val']['si_snr'], 'r-', label='Validation')
    plt.title('SI-SNR (dB)')
    plt.xlabel('Epochs')
    plt.ylabel('SI-SNR')
    plt.legend()
    plt.grid(True)
    
    # SDR图表
    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics['train']['sdr'], 'b-', label='Train')
    plt.plot(epochs, metrics['val']['sdr'], 'r-', label='Validation')
    plt.title('SDR (dB)')
    plt.xlabel('Epochs')
    plt.ylabel('SDR')
    plt.legend()
    plt.grid(True)
    
    # PESQ图表
    plt.subplot(2, 2, 4)
    plt.plot(epochs, metrics['train']['pesq'], 'b-', label='Train')
    plt.plot(epochs, metrics['val']['pesq'], 'r-', label='Validation')
    plt.title('PESQ')
    plt.xlabel('Epochs')
    plt.ylabel('PESQ')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()
    
    # 单独保存每个指标
    plt.figure()
    plt.plot(epochs, metrics['train']['loss'], 'b-', label='Train')
    plt.plot(epochs, metrics['val']['loss'], 'r-', label='Validation')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()
    
    plt.figure()
    plt.plot(epochs, metrics['train']['si_snr'], 'b-', label='Train')
    plt.plot(epochs, metrics['val']['si_snr'], 'r-', label='Validation')
    plt.title('SI-SNR (dB)')
    plt.xlabel('Epochs')
    plt.ylabel('SI-SNR')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'si_snr.png'))
    plt.close()
    
    plt.figure()
    plt.plot(epochs, metrics['train']['sdr'], 'b-', label='Train')
    plt.plot(epochs, metrics['val']['sdr'], 'r-', label='Validation')
    plt.title('SDR (dB)')
    plt.xlabel('Epochs')
    plt.ylabel('SDR')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'sdr.png'))
    plt.close()
    
    plt.figure()
    plt.plot(epochs, metrics['train']['pesq'], 'b-', label='Train')
    plt.plot(epochs, metrics['val']['pesq'], 'r-', label='Validation')
    plt.title('PESQ')
    plt.xlabel('Epochs')
    plt.ylabel('PESQ')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'pesq.png'))
    plt.close()

    
def compute_si_snr(source, estimate_source):
    """
    计算Scale-Invariant Signal-to-Noise Ratio (SI-SNR)指标
    参数:
        source: 干净目标音频 [batch, samples] 或 [batch, channels, samples]
        estimate_source: 增强后的音频 [batch, samples] 或 [batch, channels, samples]
    返回:
        SI-SNR值 (dB)
    """
    # 确保维度正确
    if len(source.shape) > 2:
        source = torch.squeeze(source, 1)
    if len(estimate_source.shape) > 2:
        estimate_source = torch.squeeze(estimate_source, 1)
    
    # 零均值化
    source = source - torch.mean(source, dim=-1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, dim=-1, keepdim=True)
    
    # 计算目标投影
    s_target = torch.sum(source * estimate_source, dim=-1, keepdim=True) * source
    s_target /= torch.norm(source, p=2, dim=-1, keepdim=True)**2 + 1e-8
    
    # 计算噪声分量
    e_noise = estimate_source - s_target
    
    # 计算SI-SNR
    si_snr = 10 * torch.log10(
        torch.norm(s_target, p=2, dim=-1)**2 / 
        (torch.norm(e_noise, p=2, dim=-1)**2 + 1e-8))
    
    return si_snr.mean()

        
# PESQ和SDR计算函数示例（需要安装pypesq和mir_eval）
def calculate_pesq(enhanced, target, sr):
    """
    计算PESQ指标
    参数:
        enhanced: 增强后的音频 (torch.Tensor)
        target: 原始目标音频 (torch.Tensor)
        sr: 采样率
    返回:
        pesq值
    """
    from pypesq import pesq
    try:
        # 转换为numpy数组
        enhanced = enhanced.detach().cpu().numpy().flatten()
        target = target.detach().cpu().numpy().flatten()
        
        # 确保长度相同
        min_len = min(len(enhanced), len(target))
        enhanced = enhanced[:min_len]
        target = target[:min_len]
        
        return pesq(target, enhanced, sr)
    except:
        return 0.0  # 计算失败时返回0

def calculate_sdr(enhanced, target):
    """
    更健壮的SDR计算实现
    """
    from mir_eval.separation import bss_eval_sources
    try:
        # 确保维度正确
        if len(enhanced.shape) == 3:
            enhanced = enhanced.squeeze(1)  # 移除通道维度
        if len(target.shape) == 3:
            target = target.squeeze(1)
        
        # 转换为numpy并确保长度匹配
        enhanced = enhanced.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        
        sdrs = []
        for i in range(enhanced.shape[0]):
            min_len = min(len(enhanced[i]), len(target[i]))
            e = enhanced[i][:min_len]
            t = target[i][:min_len]
            
            # 处理全零情况
            if np.all(e == 0) or np.all(t == 0):
                sdrs.append(0.0)
                continue
                
            # 计算SDR
            sdr, _, _, _ = bss_eval_sources(t.reshape(1, -1), e.reshape(1, -1))
            sdrs.append(np.mean(sdr))
        
        return np.mean(sdrs) if sdrs else 0.0
    except Exception as e:
        print(f"SDR calculation error: {e}")
        return 0.0

 