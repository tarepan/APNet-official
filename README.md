# APNet: An All-Frame-Level Neural Vocoder Incorporating Direct Prediction of Amplitude and Phase Spectra
### Yang Ai, Zhen-Hua Ling

In our [paper](https://arxiv.org/xxx), 
we proposed APNet: An all-frame-level neural vocoder reconstructing speech waveforms from acoustic features by predicting amplitude and phase spectra directly.<br/>
We provide our implementation and pretrained models as open source in this repository.

**Abstract :**
This paper presents a novel neural vocoder named APNet which reconstructs speech waveforms from acoustic features by predicting amplitude and phase spectra directly. The APNet vocoder is composed of an amplitude spectrum predictor (ASP) and a phase spectrum predictor (PSP). The ASP is a residual convolution network which predicts frame-level log amplitude spectra from acoustic features. The PSP also adopts a residual convolution network using acoustic features as input, then passes the output of this network through two parallel linear convolution layers respectively, and finally integrates into a phase calculation formula to estimate frame-level phase spectra. Finally, the outputs of ASP and PSP are combined to reconstruct speech waveforms by inverse short-time Fourier transform (ISTFT). All operations of the ASP and PSP are performed at the frame level. We train the ASP and PSP jointly and define multi-level loss functions based on amplitude mean square error, phase anti-wrapping error, short-time spectral inconsistency error and time domain reconstruction error. Experimental results show that our proposed APNet vocoder achieves about 8x faster inference speed than HiFi-GAN v1 on a CPU due to the all-frame-level operations while its synthesized speech quality is comparable to HiFi-GAN v1. The synthesized speech quality of the APNet vocoder is also better than several equally efficient models. Ablation experiments also confirm that the proposed parallel phase estimation architecture is essential to phase modeling and the proposed loss functions are helpful for improving the synthesized speech quality.

Visit our [demo website](http://staff.ustc.edu.cn/~yangai/APNet/demo.html) for audio samples.

## Requirements
```
torch==1.4.0
numpy==1.17.4
librosa==0.7.2
tensorboard==2.0
soundfile==0.10.3.post1
matplotlib==3.1.3
```
