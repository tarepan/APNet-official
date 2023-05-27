<div align="center">

# APNet-official <!-- omit in toc -->
[![ColabBadge]][notebook]
[![PaperBadge]][paper]  

</div>

Clone of official ***APNet***, all-frame-level mag/phase neural vocoder.

<!-- Auto-generated by "Markdown All in One" extension -->
- [Demo](#demo)
- [Usage](#usage)
  - [Install](#install)
  - [Train](#train)
  - [Inference](#inference)
- [Results](#results)
- [References](#references)

## Demo
[Official Demo page].  

## Usage
### Install
```bash
# pip install "torch==1.8.1+cu111" -q
pip install numpy==1.21.6 soundfile==0.10.3 matplotlib==3.1.3
```
<!-- pip install git+https://github.com/tarepan/APNet-official -->

### Train
#### Dataset
Audio sampling rate should be 16kHz.

Dataset: Write the list paths of training set and validation set to `input_training_wav_list` and `input_validation_wav_list` in `config.json`, respectively.

#### Run
```
tensorboard --logdir=cp_APNet/logs
CUDA_VISIBLE_DEVICES=0 python train.py
```

<!-- Jump to ☞ [![ColabBadge]][notebook], then Run. That's all!  

For arguments, check [./apnet/config.py](https://github.com/tarepan/APNet-official/blob/main/apnet/config.py).  
For dataset, check [`speechcorpusy`](https://github.com/tarepan/speechcorpusy).   -->

### Inference
#### Dataset
Audio sampling rate should be 16kHz.

- wave-to-mel-to-wave resynthesis:
  - `"test_mel_load": 0` in `config.json`
  - write the test set waveform path to `test_input_wavs_dir` in `config.json`
- mel-to-wave vocoding:
  - `"test_mel_load": 1` in `config.json`
  - write the test set mel spectrogram (size is `80*frames`) path to `test_input_mels_dir` in `config.json`

#### Run
Write the checkpoint path to `checkpoint_file_load` in `config.json`.

```
CUDA_VISIBLE_DEVICES=<0|CPU> python inference.py
```

<!-- Both CLI and Python supported.  
For detail, jump to ☞ [![ColabBadge]][notebook] and check it.   -->

## Results
### Sample <!-- omit in toc -->
[Demo](#demo)

### Performance <!-- omit in toc -->
- training
  - x.x [iter/sec] @ NVIDIA X0 on Google Colaboratory (AMP+)
  - take about y days for whole training
- inference
  - z.z [sec/sample] @ xx

## References
### Original paper <!-- omit in toc -->
[![PaperBadge]][paper]  
```bibtex
@article{ai2023apnet,
  title={A{PN}et: An All-Frame-Level Neural Vocoder Incorporating Direct Prediction of Amplitude and Phase Spectra},
  author={Ai, Yang and Ling, Zhen-Hua},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2023}
}
```

### Acknowlegements <!-- omit in toc -->
- [HiFi-GAN](https://github.com/jik876/hifi-gan): referred for implementation


[ColabBadge]:https://colab.research.google.com/assets/colab-badge.svg

[paper]:https://arxiv.org/abs/2305.07952
[PaperBadge]:https://img.shields.io/badge/paper-arxiv.2305.07952-B31B1B.svg
[notebook]:https://colab.research.google.com/github/tarepan/APNet-official/blob/main/apnet.ipynb
[Official Demo page]:https://yangai520.github.io/APNet/