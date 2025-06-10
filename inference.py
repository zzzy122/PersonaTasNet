import os
import glob
import torch
import librosa
import argparse

from utils.audio import Audio
from utils.hparams import HParam
# from model.model import VoiceFilter
from model.embedder import SpeechEmbedder
import soundfile as sf
from tasnet_model.conv_tasnet import ConvTasNet

def main(args, hp):
    M, N, L, T = 2, 256, 4, 48000

    K = 2*T//L-1
    B, H, P, X, R, C, norm_type, causal = 2, 3, 3, 3, 2, 1, "gLN", False 
    # test Conv-TasNet
    model = ConvTasNet(N, L, B, H, P, X, R, C, norm_type=norm_type).cuda()
    chkpt_model = torch.load(args.checkpoint_path)['model']
    model.load_state_dict(chkpt_model)

    with torch.no_grad():

        # model = VoiceFilter(hp).cuda()                              # 加载模型
        model.eval()

        embedder = SpeechEmbedder(hp).cuda()
        chkpt_embed = torch.load(args.embedder_path)
        embedder.load_state_dict(chkpt_embed)
        embedder.eval()

        audio = Audio(hp)                                               # 加载
        dvec_wav, _ = librosa.load(args.reference_file, sr=16000)  
        
        # 新增：保存目标音频
        os.makedirs(args.out_dir, exist_ok=True)
        target_path = os.path.join(args.out_dir, 'target.wav')
        sf.write(target_path, dvec_wav, 16000)
        
        dvec_mel = audio.get_mel(dvec_wav)                              # dvec_mel.shape   -------->     (40, 301)
        dvec_mel = torch.from_numpy(dvec_mel).float().cuda()
        dvec = embedder(dvec_mel)
        dvec = dvec.unsqueeze(0)

        mixed_wav, _ = librosa.load(args.mixed_file, sr=16000)  
        
        # 新增：保存混合音频
        mixed_path = os.path.join(args.out_dir, 'mixed.wav')
        sf.write(mixed_path, mixed_wav, 16000)
        
        # mag, phase = audio.wav2spec(mixed_wav)
        # mag = torch.from_numpy(mag).float().cuda
        mixed_wav = torch.from_numpy(mixed_wav).float().cuda()


        mixed_wav = mixed_wav.unsqueeze(0)
        mask = model(mixed_wav, dvec)
        est_wav = mixed_wav * mask

        est_wav = est_wav[0].cpu().detach().numpy()
 

        # out_path = os.path.join(args.out_dir, 'result.wav')
        out_path = os.path.join(args.out_dir, 'separated.wav')
        
        # librosa.output.write_wav(out_path, est_wav, sr=16000)
        sf.write(out_path, est_wav, 16000)
        import numpy as np
        sf.write(out_path, np.ravel(est_wav), 16000)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', type=str, required=True,
    parser.add_argument('-c', '--config', type=str ,default=os.getcwd() + '/config/default_train.yaml',
                        help="yaml file for configuration")
    


    # parser.add_argument('-e', '--embedder_path', type=str, required=True,
    parser.add_argument('-e', '--embedder_path', type=str, default=os.getcwd() + '/data_ziyun/train-clean-100-1percent/embedder_path/embedder.pt',
                        help="path of embedder model pt file")
    
    # parser.add_argument('--checkpoint_path', type=str, default=None,
    # parser.add_argument('--checkpoint_path', type=str, default=os.getcwd() + '/data_ziyun/train-clean-100-1percent/model_path/chkpt_1.pt',
    # parser.add_argument('--checkpoint_path', type=str, default=os.getcwd() + '/data_ziyun/train-clean-100-1percent/model_path',
    parser.add_argument('--checkpoint_path', type=str, default=os.getcwd() + '/data_ziyun/train-clean-100-1percent/model_path/chkpt_50.pt',
                        help="path of checkpoint pt file")
    
    # parser.add_argument('-m', '--mixed_file', type=str, required=True,
    # parser.add_argument('-m', '--mixed_file', type=str, default=os.getcwd() + '/data_ziyun/output_ziyun/test',
    parser.add_argument('-m', '--mixed_file', type=str, default=os.getcwd() + '/data_ziyun/output_ziyun/train/000000-mixed.wav',
                        help='path of mixed wav file')
    # parser.add_argument('-r', '--reference_file', type=str, required=True,
    # parser.add_argument('-r', '--reference_file', type=str, default=os.getcwd() + '/data_ziyun/output_ziyun/test',
    parser.add_argument('-r', '--reference_file', type=str, default=os.getcwd() + '/data_ziyun/output_ziyun/train/000000-target.wav',
                        help='path of reference wav file')

    # parser.add_argument('-o', '--out_dir', type=str, required=True,
    parser.add_argument('-o', '--out_dir', type=str, default =os.getcwd() +'/data_ziyun/train-clean-100-1percent/out_dir',
                        help='directory of output')

    args = parser.parse_args()

    hp = HParam(args.config)

    main(args, hp)
