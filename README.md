# Speaker-Conditional Conv-TasNet for Personalized Speech Enhancement

## Overview

This project implements a **personalized speech enhancement system** based on **speaker embeddings (d-vector) and temporal convolutional networks (Conv-TasNet)**. The system can accurately extract target speaker's voice from mixed audio, making it suitable for smart devices, remote communication, hearing aids, and other applications.

**Key Innovations:**
- Direct integration of speaker d-vectors with time-domain separation backbone (Conv-TasNet), avoiding information loss from frequency-domain methods
- Multi-objective loss function support (SI-SNR + cosine similarity) for improved signal fidelity and speaker consistency
- End-to-end training with support for large-scale data and GPU acceleration

---

## Architecture Overview

The system consists of three main components:

1. **Encoder**: Converts time-domain waveform to feature representation using 1D convolution
2. **Speaker-Conditional Separator**: Fuses d-vector with mixed audio features and applies temporal convolution network for separation
3. **Decoder**: Reconstructs separated waveforms using overlap-add method

```
Mixed Audio [M,T] → Encoder → Feature Fusion ← D-vector [M,256]
                                    ↓
                             Temporal ConvNet
                                    ↓
                           Mask Generation → Decoder → Separated Audio [M,C,T]
```

---

## Project Structure

```
.
├── config/                 # Configuration files (model, training, data parameters)
│   ├── default.yaml       # Base configuration template
│   └── default_train.yaml # Training configuration with data paths
├── datasets/               # Data loading and preprocessing
│   └── dataloader.py      # Main data loader implementation
├── model/                  # Speaker embedding and separation models
│   ├── embedder.py        # Speaker d-vector extraction (LSTM-based)
│   └── model.py           # Main separation model definitions
├── tasnet_model/          # Conv-TasNet core architecture
│   ├── conv_tasnet.py     # Main Conv-TasNet implementation
│   └── tasnet_utils.py    # Utility functions for TasNet
├── utils/                 # Audio processing, loss functions, training utilities
│   ├── train.py          # Main training script
│   ├── audio.py          # Audio processing utilities
│   └── evaluation.py     # Evaluation metrics
├── generator.py           # Training/testing data generation script
├── trainer.py            # Training main program
├── inference.py          # Inference/evaluation main program
├── embedder.pt           # Pre-trained speaker embedding model
└── README.md             # Project documentation
```

---

## Requirements

- Python 3.8+
- PyTorch 1.7+
- librosa
- soundfile
- numpy
- tqdm
- yaml
- scipy

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Data Preparation

### 1. Download Speech Dataset
Recommended datasets: [LibriSpeech](http://www.openslr.org/12/) or [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)

### 2. Audio Normalization and Resampling
```bash
sh utils/normalize-resample.sh
```

### 3. Generate Training/Testing Samples
```bash
python generator.py -c config/default_train.yaml -d [dataset_directory] -o [output_directory] -p [num_processes]
```

The data generation process creates:
- `mixed.wav`: Mixed audio (2 speakers)
- `target.wav`: Target speaker audio
- `dvec.txt`: Reference audio path for d-vector extraction
- `*.pt`: Pre-computed spectral features (optional)

---

## Training

### 1. Speaker Embedding Model
The project uses a pre-trained speaker embedding model (`embedder.pt`). You can also train your own using VoxCeleb2 or similar datasets.

### 2. Train Separation Model
```bash
python trainer.py -c config/default_train.yaml -e [embedder_model_path]
```

**Training Features:**
- Multi-objective loss (SI-SNR + cosine similarity) with adjustable weights
- Checkpoint saving and resuming
- TensorBoard visualization support
- Mixed precision training (optional)

**Key Training Parameters:**
- `N=256`: Number of encoder filters
- `L=40`: Filter length
- `B=128`: Bottleneck channels
- `H=512`: Convolution block channels
- `R=3`: Number of repeats
- `X=8`: Blocks per repeat

---

## Inference and Evaluation

### Single Audio Inference
```bash
python inference.py -c config/default_train.yaml -e [embedder_model_path] --checkpoint_path [separation_model_weights] -m [mixed_audio] -r [reference_audio] -o [output_directory]
```

### Batch Inference
```bash
python inference.py -c config/default_train.yaml -e [embedder_model_path] --checkpoint_path [separation_model_weights] --test_dir [test_data_directory] -o [output_directory]
```

**Evaluation Metrics:**
- SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
- SDR (Signal-to-Distortion Ratio)
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)

---

## Key Modules

### Core Models
- **`model/embedder.py`**: Speaker d-vector embedding extraction (LSTM-based architecture)
- **`tasnet_model/conv_tasnet.py`**: Time-domain separation backbone (encoder-separator-decoder)
- **`datasets/dataloader.py`**: Data loading with d-vector integration

### Loss Functions
- **SI-SNR Loss**: Scale-invariant signal-to-noise ratio for separation quality
- **Cosine Similarity Loss**: Speaker embedding consistency between reference and separated audio
- **Combined Loss**: Weighted combination of above losses

### Training Infrastructure
- **`utils/train.py`**: Main training loop with mixed precision support
- **`trainer.py`**: Training coordinator with configuration management
- **`utils/evaluation.py`**: Comprehensive evaluation metrics

---

## Model Architecture Details

### Speaker-Conditional Fusion
The d-vector is integrated into the separation process through feature-level fusion:

```python
# D-vector expansion and fusion
dvec = dvec.unsqueeze(2).expand(M, -1, K)  # [M, 256] → [M, 256, K]
mixture_w = mixture_w + dvec  # Element-wise addition
```

### Temporal Convolution Network
- **Encoder**: 1D convolution with 50% overlap (kernel=40, stride=20)
- **TCN Blocks**: Dilated convolutions with exponentially increasing dilation factors
- **Decoder**: Linear transformation with overlap-add reconstruction

---

## Performance Optimization

### Training Tips
1. **Data Augmentation**: Use different speaker combinations and noise levels
2. **Learning Rate Scheduling**: Cosine annealing or step decay
3. **Mixed Precision**: Enable for faster training with minimal quality loss
4. **Gradient Clipping**: Prevent gradient explosion in deep networks

### Inference Optimization
1. **Batch Processing**: Process multiple files simultaneously
2. **GPU Utilization**: Use CUDA for faster inference
3. **Memory Management**: Process long audio files in chunks

---

## Configuration

The system uses YAML configuration files for easy parameter management:

```yaml
# Model parameters
model:
  N: 256          # Encoder filters
  L: 40           # Filter length
  B: 128          # Bottleneck channels
  H: 512          # Conv block channels
  P: 3            # Kernel size
  X: 8            # Blocks per repeat
  R: 3            # Number of repeats
  C: 2            # Number of speakers

# Training parameters
training:
  batch_size: 4
  learning_rate: 1e-3
  num_epochs: 100
  loss_weights:
    sisnr: 1.0
    cosine: 0.1
```

---

## References

- [VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking](https://arxiv.org/abs/1810.04826)
- [Conv-TasNet: Surpassing Ideal Time–Frequency Masking for Speech Separation](https://arxiv.org/abs/1809.07454)
- [Speaker-independent Speech Separation with Deep Attractor Network](https://arxiv.org/abs/1707.03634)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

---

## Acknowledgments

- This project is based on the Conv-TasNet and VoiceFilter architectures
- Speaker embedding model adapted from various deep speaker recognition works
- Thanks to the LibriSpeech and VoxCeleb teams for providing high-quality datasets


---

