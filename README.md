# 个性化语音增强系统：基于说话人嵌入与时域卷积网络

## 项目简介

本项目实现了一套基于**说话人嵌入（d-vector）和时域卷积网络（Conv-TasNet）**的个性化语音增强系统。该系统能够在混合语音中精准提取目标说话人的语音，广泛适用于智能设备、远程通信、助听等场景。

**核心创新：**
- 直接将说话人d-vector嵌入与时域分离主干（Conv-TasNet）结合，避免频域方法的信息损失。
- 支持多目标损失函数（如SI-SNR+余弦相似度），提升信号保真与说话人一致性。
- 端到端训练，兼容大规模数据和GPU加速。

---

## 目录结构

```
.
├── config/           # 配置文件（模型、训练、数据等参数）
├── datasets/         # 数据加载与预处理
├── model/            # 说话人嵌入与主分离模型
├── tasnet_model/     # Conv-TasNet核心结构与工具
├── utils/            # 音频处理、损失函数、训练辅助等
├── generator.py      # 训练/测试数据生成脚本
├── trainer.py        # 训练主程序
├── inference.py      # 推理/评估主程序
├── requirements.txt  # 依赖包列表
└── README.md         # 项目说明（本文件）
```

---

## 环境依赖

- Python 3.8+
- PyTorch 1.7+
- librosa
- soundfile
- numpy
- tqdm
- 其他依赖见 requirements.txt

安装依赖：
```bash
pip install -r requirements.txt
```

---

## 数据准备

1. **下载语音数据集**  
   推荐使用 [LibriSpeech](http://www.openslr.org/12/) 或 [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)。

2. **音频重采样与归一化**  
   ```bash
   sh utils/normalize-resample.sh
   ```

3. **生成训练/测试样本**  
   ```bash
   python generator.py -c config/default.yaml -d [数据集目录] -o [输出目录] -p [进程数]
   ```

---

## 训练流程

1. **训练说话人嵌入模型（d-vector）**  
   可使用预训练模型，或用VoxCeleb2等数据自行训练。

2. **训练分离模型**  
   ```bash
   python trainer.py -c config/default.yaml -e [embedder模型路径]
   ```

   - 支持多目标损失（SI-SNR + 余弦相似度），可在 config 中调整权重。
   - 支持断点续训、TensorBoard可视化。

---

## 推理与评估

```bash
python inference.py -c config/default.yaml -e [embedder模型路径] --checkpoint_path [分离模型权重] -m [混合音频] -r [参考音频] -o [输出目录]
```

- 支持批量推理和单条音频分离。
- 评估指标包括 SI-SNR、SDR、PESQ 等。

---

## 主要模块说明

- **model/embedder.py**：说话人d-vector嵌入提取（LSTM结构）
- **model/model.py**：主分离模型（VoiceFilter/Conv-TasNet），实现d-vector条件建模
- **tasnet_model/conv_tasnet.py**：时域分离主干（编码-分离-解码）
- **utils/uitls.py**：SI-SNR、余弦相似度等损失函数实现
- **datasets/dataloader.py**：数据加载与批处理

---

## 损失函数与优化

- **多目标损失**：SI-SNR损失 + 余弦相似度损失（可加权组合）
- **优化器**：Adam 或 AdaBound
- **训练目标**：同时提升信号保真度和说话人特征一致性

---

## 参考与致谢

- [VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking](https://arxiv.org/abs/1810.04826)
- [Conv-TasNet: Surpassing Ideal Time–Frequency Masking for Speech Separation](https://arxiv.org/abs/1809.07454)
- 本项目部分代码参考自上述论文及其开源实现。

---

## 联系方式


---

