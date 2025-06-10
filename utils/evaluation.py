import torch
import torch.nn as nn
from mir_eval.separation import bss_eval_sources
import os,sys
sys.path.append(os.getcwd())
from utils.audio import Audio  as  ado
from utils.hparams import HParam
from .uitls import si_snr_loss, cosine_similarity_loss

# def validate(audio, model, embedder, testloader, writer, step):
def validate(args,audio, model, embedder, testloader, writer, step) :
    Audio = ado(HParam(args.config))


    model.eval()

    
    criterion = nn.MSELoss()
    val_loss = []
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
            est_wav = model(mixed_wav, dvec)
            # est_wav = est_mask * mixed_wav
            # test_loss = criterion(target_wav, est_wav).item()
            loss_si_snr = si_snr_loss(est_wav, target_wav)
            loss_cos_sim = cosine_similarity_loss(est_wav, target_wav)
            test_loss = loss_si_snr + 0.1*loss_cos_sim
            val_loss.append(test_loss)

            # mixed_mag = mixed_mag[0].cpu().detach().numpy()
            # target_mag = target_mag[0].cpu().detach().numpy()

            # est_mag = mixed_mag[0] 
            # est_wav = audio.spec2wav(est_mag, mixed_phase)
            # est_wav = Audio.spec2wav(est_mag, mixed_phase)
            # est_wav = est_wav.squeeze().cpu().detach().numpy()
            # est_mask = est_mask[0].cpu().detach().numpy()
            # target_wav = target_wav.squeeze().cpu().detach().numpy()

            # sdr = bss_eval_sources(target_wav, est_wav, False)[0][0]
            sdr = test_loss

 
            # writer.log_evaluation(test_loss, sdr,
            #                       mixed_wav.squeeze(), target_wav, est_wav,
            #                       mixed_mag.T, target_mag.T, est_mag.T, est_mask.T,
            #                       step)
             
    avg_loss = sum(val_loss) / len(val_loss)
    print(f"Average validation loss: {avg_loss:.4f}")
             
    model.train()




# def validate(audio, model, embedder, testloader, writer, step):
def validate_tas(args,audio, model, embedder, testloader, writer, step) :
    Audio = ado(HParam(args.config))


    model.eval()

    
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch in testloader:
            dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase = batch[0]


            # mixed_wav = mixed_wav.cuda()
            # target_wav = target_wav.cuda()    
            mixed_wav = torch.tensor(mixed_wav).cuda().unsqueeze(0)
            target_wav = torch.tensor(target_wav).cuda().unsqueeze(0)       
            output = model(mixed_wav)       # .to('cuda:0')  device
            # loss = criterion(output[:,0,:], target_wav)
            est_wav = output[:,0,:]
            test_loss = criterion(est_wav, target_wav).item()
            sdr = test_loss

            mixed_wav = mixed_wav[0].cpu().detach().numpy()
            target_wav= target_wav[0].cpu().detach().numpy()
            est_wav= est_wav.cpu()[0].detach().numpy()

            ######################## 凑数
            mixed_mag = mixed_mag.unsqueeze(0).cuda()
            mixed_mag = mixed_mag[0].cpu().detach().numpy()
            target_mag = mixed_mag
            est_mag = mixed_mag         #凑数
            est_mask = mixed_mag         # 凑数

            writer.log_evaluation(test_loss, sdr,
                                  mixed_wav, target_wav, est_wav,
                                  mixed_mag.T, target_mag.T, est_mag.T, est_mask.T,
                                #   mixed_mag.T, target_mag.T, target_mag.T, target_mag.T,
                                  step)



            # ########################## ---------------------------------上面是改动

            # dvec_mel = dvec_mel.cuda()
            # target_mag = target_mag.unsqueeze(0).cuda()
            # mixed_mag = mixed_mag.unsqueeze(0).cuda()

            # dvec = embedder(dvec_mel)
            # dvec = dvec.unsqueeze(0)
            # est_mask = model(mixed_mag, dvec)
            # est_mag = est_mask * mixed_mag
            # test_loss = criterion(target_mag, est_mag).item()

            # mixed_mag = mixed_mag[0].cpu().detach().numpy()
            # target_mag = target_mag[0].cpu().detach().numpy()
            # est_mag = est_mag[0].cpu().detach().numpy()
            # # est_wav = audio.spec2wav(est_mag, mixed_phase)
            # est_wav = Audio.spec2wav(est_mag, mixed_phase)

            # est_mask = est_mask[0].cpu().detach().numpy()

            # sdr = bss_eval_sources(target_wav, est_wav, False)[0][0]
            # writer.log_evaluation(test_loss, sdr,
            #                       mixed_wav, target_wav, est_wav,
            #                       mixed_mag.T, target_mag.T, est_mag.T, est_mask.T,
            #                       step)
            break

    model.train()
