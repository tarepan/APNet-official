import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm
from extorch import Conv1dEx

from utils import init_weights

LRELU_SLOPE = 0.1


class ResBlock(torch.nn.Module):
    """ResBlock Q=3, Res[LReLU-DConv-LReLU-Conv]x3."""
    def __init__(self, channels: int, kernel_size: int, dilation: tuple[int, int, int], causal: bool = False):
        super().__init__()

        # Validation
        assert kernel_size % 2 == 1, f"Support only odd-number kernel, but set to {kernel_size}."

        # DilatedConv
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1dEx(channels, channels, kernel_size, dilation=dilation[0], padding="same", causal=causal)),
            weight_norm(Conv1dEx(channels, channels, kernel_size, dilation=dilation[1], padding="same", causal=causal)),
            weight_norm(Conv1dEx(channels, channels, kernel_size, dilation=dilation[2], padding="same", causal=causal)),
        ])
        self.convs1.apply(init_weights)
        # Conv
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1dEx(channels, channels, kernel_size,                       padding="same", causal=causal)),
            weight_norm(Conv1dEx(channels, channels, kernel_size,                       padding="same", causal=causal)),
            weight_norm(Conv1dEx(channels, channels, kernel_size,                       padding="same", causal=causal)),
        ])
        self.convs2.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            # Res[LReLU-DConv-LReLU-Conv]
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class Generator(torch.nn.Module):
    def __init__(self, h):
        super().__init__()

        # Validation
        assert h.PSP_input_conv_kernel_size    % 2 == 1, f"Support only odd-number kernel, but set to {h.PSP_input_conv_kernel_size}."
        assert h.ASP_input_conv_kernel_size    % 2 == 1, f"Support only odd-number kernel, but set to {h.ASP_input_conv_kernel_size}."
        assert h.ASP_output_conv_kernel_size   % 2 == 1, f"Support only odd-number kernel, but set to {h.ASP_output_conv_kernel_size}."
        assert h.PSP_output_R_conv_kernel_size % 2 == 1, f"Support only odd-number kernel, but set to {h.PSP_output_R_conv_kernel_size}."
        assert h.PSP_output_I_conv_kernel_size % 2 == 1, f"Support only odd-number kernel, but set to {h.PSP_output_I_conv_kernel_size}."

        self.h = h
        self.asp_num_kernels = len(h.ASP_resblock_kernel_sizes) # `P` of ASP
        self.psp_num_kernels = len(h.PSP_resblock_kernel_sizes) # `P` of PSP
        freq = h.n_fft // 2 + 1

        # PreNets
        self.ASP_input_conv = weight_norm(Conv1dEx(h.num_mels, h.ASP_channel, h.ASP_input_conv_kernel_size, padding="same", causal=h.causal))
        self.PSP_input_conv = weight_norm(Conv1dEx(h.num_mels, h.PSP_channel, h.PSP_input_conv_kernel_size, padding="same", causal=h.causal))

        # MainNets - `P`-kernel multi receptive field network for ASP and PSP
        self.ASP_ResNet = nn.ModuleList()
        for k, d in zip(h.ASP_resblock_kernel_sizes, h.ASP_resblock_dilation_sizes):
            self.ASP_ResNet.append(ResBlock(h.ASP_channel, k, d, h.causal))
        self.PSP_ResNet = nn.ModuleList()
        for k, d in zip(h.PSP_resblock_kernel_sizes, h.PSP_resblock_dilation_sizes):
            self.PSP_ResNet.append(ResBlock(h.PSP_channel, k, d, h.causal))

        # PostNet
        self.ASP_output_conv   = weight_norm(Conv1dEx(h.ASP_channel, freq, h.ASP_output_conv_kernel_size,   padding="same", causal=h.causal))
        self.PSP_output_R_conv = weight_norm(Conv1dEx(h.PSP_channel, freq, h.PSP_output_R_conv_kernel_size, padding="same", causal=h.causal))
        self.PSP_output_I_conv = weight_norm(Conv1dEx(h.PSP_channel, freq, h.PSP_output_I_conv_kernel_size, padding="same", causal=h.causal))
        self.ASP_output_conv.apply(init_weights)
        self.PSP_output_R_conv.apply(init_weights)
        self.PSP_output_I_conv.apply(init_weights)

        if h.use_fc:
            raise RuntimeError("Currently, `use_fc` is not supported in Generator.")

    def forward(self, mel):
        """
        Args:
            mel - Mel-Frequency 
        Returns:
            logamp - Linear-Frequency Log-Amplitude spectrogram
            pha
            real
            imag
            audio
        """
        # ASP
        ## PreNet
        logamp = self.ASP_input_conv(mel)
        ## MainNet - MRF
        logamps = self.ASP_ResNet[0](logamp)
        for j in range(1, self.asp_num_kernels):
            logamps += self.ASP_ResNet[j](logamp)
        logamp = logamps / self.asp_num_kernels
        logamp = F.leaky_relu(logamp)
        ## PostNet
        logamp = self.ASP_output_conv(logamp)

        # PSP
        ## PreNet
        pha = self.PSP_input_conv(mel)
        ## MainNet - MRF
        phas = self.PSP_ResNet[0](pha)
        for j in range(1, self.psp_num_kernels):
            phas += self.PSP_ResNet[j](pha)
        pha = phas / self.psp_num_kernels
        pha = F.leaky_relu(pha)   
        ## PostNet - parallel phase estimation
        R = self.PSP_output_R_conv(pha)
        I = self.PSP_output_I_conv(pha)
        pha = torch.atan2(I, R)

        # Complex spectrogram
        spec = torch.exp(logamp) * (torch.exp(1j * pha))
        real = torch.exp(logamp) * torch.cos(pha)
        imag = torch.exp(logamp) * torch.sin(pha)

        # iSTFT
        audio = torch.istft(spec, self.h.n_fft, hop_length=self.h.hop_size, win_length=self.h.win_size, window=torch.hann_window(self.h.win_size).to(mel.device), center=True)

        return logamp, pha, real, imag, audio.unsqueeze(1)


class UnifiedGenerator(torch.nn.Module):
    def __init__(self, h):
        super().__init__()

        # Validation
        assert h.ASP_input_conv_kernel_size    % 2 == 1, f"Support only odd-number kernel, but set to {h.ASP_input_conv_kernel_size}."
        assert h.ASP_output_conv_kernel_size   % 2 == 1, f"Support only odd-number kernel, but set to {h.ASP_output_conv_kernel_size}."
        assert h.PSP_output_R_conv_kernel_size % 2 == 1, f"Support only odd-number kernel, but set to {h.PSP_output_R_conv_kernel_size}."
        assert h.PSP_output_I_conv_kernel_size % 2 == 1, f"Support only odd-number kernel, but set to {h.PSP_output_I_conv_kernel_size}."

        # Params
        self.h = h
        self.num_kernels = len(h.ASP_resblock_kernel_sizes) # `P` of ASP
        freq = h.n_fft // 2 + 1
        feat_h = h.ASP_channel

        # MainNet - `P`-kernel multi receptive field network
        self.mainnet = nn.ModuleList()
        for k, d in zip(h.ASP_resblock_kernel_sizes, h.ASP_resblock_dilation_sizes):
            self.mainnet.append(ResBlock(feat_h, k, d, h.causal))

        # PreNet/PostNets
        self.prenet    = weight_norm(Conv1dEx(h.num_mels, feat_h, h.ASP_input_conv_kernel_size,    padding="same", causal=h.causal))
        self.postnet_a = weight_norm(Conv1dEx(feat_h,     freq,   h.ASP_output_conv_kernel_size,   padding="same", causal=h.causal))
        self.postnet_r = weight_norm(Conv1dEx(feat_h,     freq,   h.PSP_output_R_conv_kernel_size, padding="same", causal=h.causal))
        self.postnet_i = weight_norm(Conv1dEx(feat_h,     freq,   h.PSP_output_I_conv_kernel_size, padding="same", causal=h.causal))
        self.postnet_a.apply(init_weights)
        self.postnet_r.apply(init_weights)
        self.postnet_i.apply(init_weights)

        if h.use_fc:
            self.fcistft = \
                         weight_norm(Conv1dEx(freq*2, h.hop_size, 1))

    def forward(self, mel):
        """
        Args:
            mel - Mel-Frequency 
        Returns:
            logamp - Linear-Frequency Log-Amplitude spectrogram
            phase
            real
            imag
            audio
        """
        # ASP
        ## PreNet
        h = self.prenet(mel)
        ## MainNet - MRF
        hs      = self.mainnet[0](h)
        for j in range(1, self.num_kernels):
            hs += self.mainnet[j](h)
        h = hs / self.num_kernels
        h = F.leaky_relu(h)
        ## PostNet - LogAmp/PhaseRe/PhaseIm
        logamp  = self.postnet_a(h)
        phase_r = self.postnet_r(h)
        phase_i = self.postnet_i(h)
        phase = torch.atan2(phase_i, phase_r)

        real = torch.exp(logamp) * torch.cos(phase)
        imag = torch.exp(logamp) * torch.sin(phase)

        if not self.h.use_fc:
            # iSTFT :: (B, Feat=2*freq, Frame=frm) -> (B, T=)
            spec = torch.exp(logamp) * (torch.exp(1j * phase))
            audio = torch.istft(spec, self.h.n_fft, hop_length=self.h.hop_size, win_length=self.h.win_size, window=torch.hann_window(self.h.win_size).to(mel.device), center=True)
        else:
            # TODO: padding
            # FC :: (B, Feat=2*freq, Frame=frm) -> (B, Feat=hop, Frame=frm) -> (B, T=hop*frm)
            audio = self.fcistft(torch.cat([logamp, phase], dim=-2)).transpose(-2, -1).flatten(start_dim=1)
            audio = audio[:, 80:]

        return logamp, phase, real, imag, audio.unsqueeze(1)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        pad_l = int((5 - 1)/2)
        self.convs = nn.ModuleList([
            norm_f(Conv2d(   1,   32, (kernel_size, 1), (stride, 1), padding=(pad_l, 0))),
            norm_f(Conv2d(  32,  128, (kernel_size, 1), (stride, 1), padding=(pad_l, 0))),
            norm_f(Conv2d( 128,  512, (kernel_size, 1), (stride, 1), padding=(pad_l, 0))),
            norm_f(Conv2d( 512, 1024, (kernel_size, 1), (stride, 1), padding=(pad_l, 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1),           1, padding=(    2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    """5-periods MDP, p=2/3/5/7/11"""
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(   1,  128, 15,               padding="same")),
            norm_f(Conv1d( 128,  128, 41, 2, groups= 4, padding=20)),
            norm_f(Conv1d( 128,  256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d( 256,  512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d( 512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41,    groups=16, padding="same")),
            norm_f(Conv1d(1024, 1024,  5,               padding="same")),
        ])
        self.conv_post = \
            norm_f(Conv1d(1024,    1,  3,               padding="same"))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    """3-scale MSD, x1/x2/x4"""
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2

def phase_loss(phase_r, phase_g, n_fft, frames):

    MSELoss = torch.nn.MSELoss()

    GD_matrix = torch.triu(torch.ones(n_fft//2+1,n_fft//2+1),diagonal=1)-torch.triu(torch.ones(n_fft//2+1,n_fft//2+1),diagonal=2)-torch.eye(n_fft//2+1)
    GD_matrix = GD_matrix.to(phase_g.device)

    GD_r = torch.matmul(phase_r.permute(0,2,1), GD_matrix)
    GD_g = torch.matmul(phase_g.permute(0,2,1), GD_matrix)

    PTD_matrix = torch.triu(torch.ones(frames,frames),diagonal=1)-torch.triu(torch.ones(frames,frames),diagonal=2)-torch.eye(frames)
    PTD_matrix = PTD_matrix.to(phase_g.device)

    PTD_r = torch.matmul(phase_r, PTD_matrix)
    PTD_g = torch.matmul(phase_g, PTD_matrix)

    IP_loss = torch.mean(-torch.cos(phase_r-phase_g))
    GD_loss = torch.mean(-torch.cos(GD_r-GD_g))
    PTD_loss = torch.mean(-torch.cos(PTD_r-PTD_g))


    return IP_loss, GD_loss, PTD_loss


def amplitude_loss(log_amplitude_r, log_amplitude_g):

    MSELoss = torch.nn.MSELoss()

    amplitude_loss = MSELoss(log_amplitude_r, log_amplitude_g)

    return amplitude_loss

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

def STFT_consistency_loss(rea_r, rea_g, imag_r, imag_g):

    C_loss=torch.mean(torch.mean((rea_r-rea_g)**2+(imag_r-imag_g)**2,(1,2)))
    
    return C_loss

