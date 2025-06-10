import os
import glob
import torch
import librosa
from torch.utils.data import Dataset, DataLoader

from utils.audio import Audio


def create_dataloader(hp, args, train):
    def train_collate_fn(batch):
        # batch[0][0].shape        torch.Size([40, 1504])
        # batch[0][1].shape         torch.Size([301, 601])
        # batch[0][2].shape         torch.Size([301, 601])
        dvec_list = list()
        target_mag_list = list()
        mixed_mag_list = list()

        for dvec_mel, target_mag, mixed_mag in batch:
            dvec_list.append(dvec_mel)
            target_mag_list.append(target_mag)
            mixed_mag_list.append(mixed_mag)
        target_mag_list = torch.stack(target_mag_list, dim=0)
        mixed_mag_list = torch.stack(mixed_mag_list, dim=0)

        return dvec_list, target_mag_list, mixed_mag_list

    def test_collate_fn(batch):
        return batch

    if train:
        return DataLoader(dataset=VFDataset(hp, args, True),
                          batch_size=hp.train.batch_size,
                          shuffle=True,
                          num_workers=hp.train.num_workers,
                          collate_fn=train_collate_fn,
                          pin_memory=False,
                          drop_last=True,
                          sampler=None)
    else:                                                                           # 此处更改batchsize
        return DataLoader(dataset=VFDataset(hp, args, False),
                          collate_fn=test_collate_fn,
                          batch_size=1, shuffle=False, num_workers=0)
                        #   batch_size=1, shuffle=False, num_workers=0)



class VFDataset(Dataset):
    def __init__(self, hp, args, train):
        def find_all(file_format):
            return sorted(glob.glob(os.path.join(self.data_dir, file_format)))
        self.hp = hp
        self.args = args
        self.train = train
        self.data_dir = hp.data.train_dir if train else hp.data.test_dir

        self.dvec_list = find_all(hp.form.dvec) 
        # '/media/ros/1T001/yongtong/nlp/demo_27_chengfei/demo_02_Speech_separation/voicefilter/data_ziyun/output_ziyun/train/000000-dvec.txt'

        self.target_wav_list = find_all(hp.form.target.wav)
        # '/media/ros/1T001/yongtong/nlp/demo_27_chengfei/demo_02_Speech_separation/voicefilter/data_ziyun/output_ziyun/train/000000-target.wav'

        self.mixed_wav_list = find_all(hp.form.mixed.wav)
        # '/media/ros/1T001/yongtong/nlp/demo_27_chengfei/demo_02_Speech_separation/voicefilter/data_ziyun/output_ziyun/train/000000-mixed.wav'

        self.target_mag_list = find_all(hp.form.target.mag)
        # '/media/ros/1T001/yongtong/nlp/demo_27_chengfei/demo_02_Speech_separation/voicefilter/data_ziyun/output_ziyun/train/000000-target.pt'

        self.mixed_mag_list = find_all(hp.form.mixed.mag)
        # '/media/ros/1T001/yongtong/nlp/demo_27_chengfei/demo_02_Speech_separation/voicefilter/data_ziyun/output_ziyun/train/000000-mixed.pt'
        
        ############################################################################################## MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM

        # len(self.dvec_list)           92604           self.dvec_list[-1][-15:-9]              '098763'
        # len(self.target_wav_list)     92619           self.target_wav_list[-1][-17:-11]       '098778'
        # len(self.mixed_wav_list)      92606           self.mixed_wav_list[-1][-16:-10]        '098764'
        # len(self.target_mag_list)     92605           self.target_mag_list[-1][-16:-10]       '098763'
        # len(self.mixed_mag_list)      92605           self.mixed_mag_list[-1][-15:-9]         '098763'
        # len(self.dvec_list)           92604           self.dvec_list[-1][-15:-9]              '098763'

        # [s[-15:-9] for s in self.dvec_list]
        listnum_1 = [s[-15:-9] for s in self.dvec_list]
        listnum_2 = [s[-17:-11] for s in self.target_wav_list]
        listnum_3 = [s[-16:-10] for s in self.mixed_wav_list]
        listnum_4 = [s[-16:-10] for s in self.target_mag_list]
        listnum_5 = [s[-15:-9] for s in self.mixed_mag_list]

        # set(listnum_1).intersection(set(listnum_5))
        # print([x for x in A if x in B]) # [4, 5]
        # [x for x in listnum_1 if x in listnum_2  and x in listnum_3  and x in listnum_4  and x in listnum_5]
        # list(set(listnum_1).intersection(set(listnum_2),set(listnum_3),set(listnum_4),set(listnum_5)))            # 92604

        # set(listnum_1).intersection(set(listnum_2),set(listnum_3),set(listnum_4),set(listnum_5))
        # sorted(list(set(listnum_1).intersection(set(listnum_2),set(listnum_3),set(listnum_4),set(listnum_5))))
        # ['000000', '000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011',
        # listnum_all = [x for x in listnum_1 if x in listnum_2  and x in listnum_3  and x in listnum_4  and x in listnum_5]
        listnum_all = sorted(list(set(listnum_1).intersection(set(listnum_2),set(listnum_3),set(listnum_4),set(listnum_5)))) # listnum_all[-1]  '098763'
        
        self.dvec_list_ = []            # self.dvec_list[-1][-15:-9]           self.dvec_list[-1][:-15]+self.dvec_list[-1][-9:]
        self.dvec_list_ = [ self.dvec_list[-1][:-15]+ dvec0 + self.dvec_list[-1][-9:] for dvec0 in  listnum_all ]
        self.dvec_list = self.dvec_list_

        self.target_wav_list_ = []               # self.target_wav_list[-1][-17:-11] 
        self.target_wav_list_ = [ self.target_wav_list[-1][:-17]+ dvec0 + self.target_wav_list[-1][-11:] for dvec0 in  listnum_all ]
        self.target_wav_list = self.target_wav_list_ 

        self.mixed_wav_list_ = []               # self.mixed_wav_list[-1][-16:-10]        '098764'
        self.mixed_wav_list_ = [ self.mixed_wav_list[-1][:-16]+ dvec0 + self.mixed_wav_list[-1][-10:] for dvec0 in  listnum_all ]
        self.mixed_wav_list = self.mixed_wav_list_ 

        self.target_mag_list_ = []               # self.target_mag_list[-1][-16:-10] 
        self.target_mag_list_ = [ self.target_mag_list[-1][:-16]+ dvec0 + self.target_mag_list[-1][-10:] for dvec0 in  listnum_all ]
        self.target_mag_list = self.target_mag_list_ 


        self.mixed_mag_list_ = []               # self.mixed_mag_list[-1][-15:-9]
        self.mixed_mag_list_ = [ self.mixed_mag_list[-1][:-15]+ dvec0 + self.mixed_mag_list[-1][-9:] for dvec0 in  listnum_all ]
        self.mixed_mag_list = self.mixed_mag_list_ 
        ############################################################################################## WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW





        assert len(self.dvec_list) == len(self.target_wav_list) == len(self.mixed_wav_list) == \
            len(self.target_mag_list) == len(self.mixed_mag_list), "number of training files must match"
        assert len(self.dvec_list) != 0, \
            "no training file found"

        self.audio = Audio(hp)

    def __len__(self):
        return len(self.dvec_list)

    def __getitem__(self, idx):
        with open(self.dvec_list[idx], 'r') as f:
            dvec_path = f.readline().strip()

        dvec_wav, _ = librosa.load(dvec_path, sr=self.hp.audio.sample_rate)
        dvec_mel = self.audio.get_mel(dvec_wav)
        dvec_mel = torch.from_numpy(dvec_mel).float()

        if self.train: # need to be fast                                #   训练时只返回一部分数据  不包含音频
            target_mag = torch.load(self.target_mag_list[idx])
            mixed_mag = torch.load(self.mixed_mag_list[idx])
            return dvec_mel, target_mag, mixed_mag
            # return dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase

        else:                                                           #   测试返回全数据  包含音频
            # target_wav, _ = librosa.load(self.target_wav_list[idx], self.hp.audio.sample_rate)
            # mixed_wav, _ = librosa.load(self.mixed_wav_list[idx], self.hp.audio.sample_rate)
            target_wav, _ = librosa.load(self.target_wav_list[idx], sr = self.hp.audio.sample_rate)
            mixed_wav, _ = librosa.load(self.mixed_wav_list[idx], sr = self.hp.audio.sample_rate)

            target_mag, _ = self.wav2magphase(self.target_wav_list[idx])
            mixed_mag, mixed_phase = self.wav2magphase(self.mixed_wav_list[idx])
            target_mag = torch.from_numpy(target_mag)
            mixed_mag = torch.from_numpy(mixed_mag)
            # mixed_phase = torch.from_numpy(mixed_phase)
            return dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase

    def wav2magphase(self, path):
        # wav, _ = librosa.load(path, self.hp.audio.sample_rate)
        wav, _ = librosa.load(path, sr = self.hp.audio.sample_rate)

        mag, phase = self.audio.wav2spec(wav)
        return mag, phase



####################################################################### -------------------------------------------------------------------------

def create_dataloader_tas(hp, args, train):
    def train_collate_fn(batch):
        # return dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase
            # return dvec_mel, target_mag, mixed_mag
        # len(batch)     8             len(batch[0])    6          

        # batch[0][0].shape        torch.Size([40, 1504])
        # batch[0][1].shape         torch.Size([301, 601])
        # batch[0][2].shape         torch.Size([301, 601])
        dvec_list = list()
        target_mag_list = list()
        mixed_mag_list = list()

        target_wav_list = list()
        mixed_wav_list = list()
        mixed_phase_list = list() 

        # for dvec_mel, target_mag, mixed_mag in batch:
        for dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase in batch:

            dvec_list.append(dvec_mel)
            target_mag_list.append(target_mag)
            mixed_mag_list.append(mixed_mag)

            target_wav_list.append(target_wav)
            mixed_wav_list.append(mixed_wav)
            mixed_phase_list.append(mixed_phase) 


        target_mag_list = torch.stack(target_mag_list, dim=0)
        mixed_mag_list = torch.stack(mixed_mag_list, dim=0)

        target_wav_list = torch.stack(target_wav_list, dim=0)
        mixed_wav_list = torch.stack(mixed_wav_list, dim=0)
        mixed_phase_list = torch.stack(mixed_phase_list, dim=0)

        # return dvec_list, target_mag_list, mixed_mag_list
        return dvec_list, target_wav_list, mixed_wav_list, target_mag_list, mixed_mag_list, mixed_phase_list

    def test_collate_fn(batch):
        return batch

    if train:
        # return DataLoader(dataset=VFDataset(hp, args, True),
        #                   batch_size=hp.train.batch_size,
        #                   shuffle=True,
        #                   num_workers=hp.train.num_workers,
        #                   collate_fn=train_collate_fn,
        #                   pin_memory=True,
        #                   drop_last=True,
        #                   sampler=None)
        # return DataLoader(dataset=VFDataset(hp, args, False),           ##   测试返回全数据  包含音频
        return DataLoader(dataset=VFDataset_tas(hp, args, True),           ##   测试返回全数据  包含音频
                    batch_size=hp.train.batch_size,
                    shuffle=True,
                    num_workers=hp.train.num_workers,
                    collate_fn=train_collate_fn,
                    pin_memory=True,
                    drop_last=True,
                    sampler=None)
    else:                                                                           # 此处更改batchsize
        return DataLoader(dataset=VFDataset_tas(hp, args, False),
                          collate_fn=test_collate_fn,
                          batch_size=1, shuffle=False, num_workers=0)
                        #   batch_size=hp.train.batch_size, shuffle=False, num_workers=0)







# class VFDataset(Dataset):
class VFDataset_tas(Dataset):

    def __init__(self, hp, args, train):
        def find_all(file_format):
            return sorted(glob.glob(os.path.join(self.data_dir, file_format)))
        self.hp = hp
        self.args = args
        self.train = train
        self.data_dir = hp.data.train_dir if train else hp.data.test_dir

        self.dvec_list = find_all(hp.form.dvec) 
        # '/media/ros/1T001/yongtong/nlp/demo_27_chengfei/demo_02_Speech_separation/voicefilter/data_ziyun/output_ziyun/train/000000-dvec.txt'

        self.target_wav_list = find_all(hp.form.target.wav)
        # '/media/ros/1T001/yongtong/nlp/demo_27_chengfei/demo_02_Speech_separation/voicefilter/data_ziyun/output_ziyun/train/000000-target.wav'

        self.mixed_wav_list = find_all(hp.form.mixed.wav)
        # '/media/ros/1T001/yongtong/nlp/demo_27_chengfei/demo_02_Speech_separation/voicefilter/data_ziyun/output_ziyun/train/000000-mixed.wav'

        self.target_mag_list = find_all(hp.form.target.mag)
        # '/media/ros/1T001/yongtong/nlp/demo_27_chengfei/demo_02_Speech_separation/voicefilter/data_ziyun/output_ziyun/train/000000-target.pt'

        self.mixed_mag_list = find_all(hp.form.mixed.mag)
        # '/media/ros/1T001/yongtong/nlp/demo_27_chengfei/demo_02_Speech_separation/voicefilter/data_ziyun/output_ziyun/train/000000-mixed.pt'
        
        ############################################################################################## MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM

        # len(self.dvec_list)           92604           self.dvec_list[-1][-15:-9]              '098763'
        # len(self.target_wav_list)     92619           self.target_wav_list[-1][-17:-11]       '098778'
        # len(self.mixed_wav_list)      92606           self.mixed_wav_list[-1][-16:-10]        '098764'
        # len(self.target_mag_list)     92605           self.target_mag_list[-1][-16:-10]       '098763'
        # len(self.mixed_mag_list)      92605           self.mixed_mag_list[-1][-15:-9]         '098763'
        # len(self.dvec_list)           92604           self.dvec_list[-1][-15:-9]              '098763'

        # [s[-15:-9] for s in self.dvec_list]
        listnum_1 = [s[-15:-9] for s in self.dvec_list]
        listnum_2 = [s[-17:-11] for s in self.target_wav_list]
        listnum_3 = [s[-16:-10] for s in self.mixed_wav_list]
        listnum_4 = [s[-16:-10] for s in self.target_mag_list]
        listnum_5 = [s[-15:-9] for s in self.mixed_mag_list]

        # set(listnum_1).intersection(set(listnum_5))
        # print([x for x in A if x in B]) # [4, 5]
        # [x for x in listnum_1 if x in listnum_2  and x in listnum_3  and x in listnum_4  and x in listnum_5]
        # list(set(listnum_1).intersection(set(listnum_2),set(listnum_3),set(listnum_4),set(listnum_5)))            # 92604

        # set(listnum_1).intersection(set(listnum_2),set(listnum_3),set(listnum_4),set(listnum_5))
        # sorted(list(set(listnum_1).intersection(set(listnum_2),set(listnum_3),set(listnum_4),set(listnum_5))))
        # ['000000', '000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011',
        # listnum_all = [x for x in listnum_1 if x in listnum_2  and x in listnum_3  and x in listnum_4  and x in listnum_5]
        listnum_all = sorted(list(set(listnum_1).intersection(set(listnum_2),set(listnum_3),set(listnum_4),set(listnum_5)))) # listnum_all[-1]  '098763'
        
        self.dvec_list_ = []            # self.dvec_list[-1][-15:-9]           self.dvec_list[-1][:-15]+self.dvec_list[-1][-9:]
        self.dvec_list_ = [ self.dvec_list[-1][:-15]+ dvec0 + self.dvec_list[-1][-9:] for dvec0 in  listnum_all ]
        self.dvec_list = self.dvec_list_

        self.target_wav_list_ = []               # self.target_wav_list[-1][-17:-11] 
        self.target_wav_list_ = [ self.target_wav_list[-1][:-17]+ dvec0 + self.target_wav_list[-1][-11:] for dvec0 in  listnum_all ]
        self.target_wav_list = self.target_wav_list_ 

        self.mixed_wav_list_ = []               # self.mixed_wav_list[-1][-16:-10]        '098764'
        self.mixed_wav_list_ = [ self.mixed_wav_list[-1][:-16]+ dvec0 + self.mixed_wav_list[-1][-10:] for dvec0 in  listnum_all ]
        self.mixed_wav_list = self.mixed_wav_list_ 

        self.target_mag_list_ = []               # self.target_mag_list[-1][-16:-10] 
        self.target_mag_list_ = [ self.target_mag_list[-1][:-16]+ dvec0 + self.target_mag_list[-1][-10:] for dvec0 in  listnum_all ]
        self.target_mag_list = self.target_mag_list_ 


        self.mixed_mag_list_ = []               # self.mixed_mag_list[-1][-15:-9]
        self.mixed_mag_list_ = [ self.mixed_mag_list[-1][:-15]+ dvec0 + self.mixed_mag_list[-1][-9:] for dvec0 in  listnum_all ]
        self.mixed_mag_list = self.mixed_mag_list_ 
        ############################################################################################## WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW





        assert len(self.dvec_list) == len(self.target_wav_list) == len(self.mixed_wav_list) == \
            len(self.target_mag_list) == len(self.mixed_mag_list), "number of training files must match"
        assert len(self.dvec_list) != 0, \
            "no training file found"

        self.audio = Audio(hp)

    def __len__(self):
        return len(self.dvec_list)

    def __getitem__(self, idx):
        with open(self.dvec_list[idx], 'r') as f:
            dvec_path = f.readline().strip()

        dvec_wav, _ = librosa.load(dvec_path, sr=self.hp.audio.sample_rate)
        dvec_mel = self.audio.get_mel(dvec_wav)
        dvec_mel = torch.from_numpy(dvec_mel).float()

        if self.train: # need to be fast                                #   训练时只返回一部分数据  不包含音频
            mixed_mag, mixed_phase = self.wav2magphase(self.mixed_wav_list[idx])
            mixed_phase = torch.tensor(mixed_phase)

            target_mag = torch.load(self.target_mag_list[idx])
            mixed_mag = torch.load(self.mixed_mag_list[idx])

            target_wav, _ = librosa.load(self.target_wav_list[idx], sr = self.hp.audio.sample_rate)
            target_wav = torch.tensor(target_wav)

            mixed_wav, _ = librosa.load(self.mixed_wav_list[idx], sr = self.hp.audio.sample_rate)
            mixed_wav = torch.tensor(mixed_wav)







            # return dvec_mel, target_mag, mixed_mag
            return dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase

        else:                                                           #   测试返回全数据  包含音频
            # target_wav, _ = librosa.load(self.target_wav_list[idx], self.hp.audio.sample_rate)
            # mixed_wav, _ = librosa.load(self.mixed_wav_list[idx], self.hp.audio.sample_rate)
            target_wav, _ = librosa.load(self.target_wav_list[idx], sr = self.hp.audio.sample_rate)
            mixed_wav, _ = librosa.load(self.mixed_wav_list[idx], sr = self.hp.audio.sample_rate)

            target_mag, _ = self.wav2magphase(self.target_wav_list[idx])
            mixed_mag, mixed_phase = self.wav2magphase(self.mixed_wav_list[idx])
            target_mag = torch.from_numpy(target_mag)
            mixed_mag = torch.from_numpy(mixed_mag)
            # mixed_phase = torch.from_numpy(mixed_phase)
            return dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase

    def wav2magphase(self, path):
        # wav, _ = librosa.load(path, self.hp.audio.sample_rate)
        wav, _ = librosa.load(path, sr = self.hp.audio.sample_rate)

        mag, phase = self.audio.wav2spec(wav)
        return mag, phase
