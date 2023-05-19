import random

import torch
from torch import Tensor
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
import librosa


def load_wav(full_path: str, sample_rate: int):
    """Load resampled mono waveform in range [-1, 1]."""
    return librosa.load(full_path, sr=sample_rate, mono=True)[0]


def _spectral_normalize_torch(magnitudes: Tensor):
    return torch.log(torch.clamp(magnitudes, min=1e-5))


def mel_spectrogram(
        y: Tensor, # :: (*, T)           - Waveform
        n_fft,
        num_mels,
        sampling_rate,
        hop_size,
        win_size,
        fmin,
        fmax,
    ) -> Tensor:   # :: (*, Freq, Frame) - Mel-frequency Log-Amplitude spectrogram
    """waveform to Mel-frequency Log-Amplitude spectrogram."""

    # MelBasis and Window
    mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
    mel_basis = torch.from_numpy(mel).float().to(y.device)
    hann_window = torch.hann_window(win_size).to(y.device)

    # Linear-frequency Linear-Amplitude spectrogram
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=True)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    # Mel-frequency Linear-Amplitude spectrogram
    melspec = torch.matmul(mel_basis, spec)

    # Mel-frequency Log-Amplitude spectrogram
    melspec = _spectral_normalize_torch(melspec)

    return melspec


def amp_pha_specturm(
        y,        #             :: (B, T)           - Waveforms
        n_fft,
        hop_size,
        win_size,
    ) -> tuple[
        Tensor, # log_amplitude :: (B, Freq, Frame) - Linear-frequency Log-Amplitude spectrogram
        Tensor, # phase         :: (B, Freq, Frame) - (maybe) Phase spectrogram
        Tensor, # real          :: (B, Freq, Frame) - STFT real      value
        Tensor, # imag          :: (B, Freq, Frame) - STFT imaginary value
        ]:

    hann_window=torch.hann_window(win_size).to(y.device)

    # :: (B, Freq, Frame, RealImag=2) - Linear-frequency Linear-Amplitude spectrogram
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=True)

    # real & imaginary value :: (B, Freq, Frame)
    real = spec[:, :, :, 0]
    imag = spec[:, :, :, 1]

    # :: (B, Freq, Frame) - Linear-frequency Log-Amplitude spectrogram
    log_amplitude = torch.log(torch.abs(torch.sqrt(torch.pow(real,2)+torch.pow(imag,2)))+1e-5)

    # :: (B, Freq, Frame)
    phase=torch.atan2(imag, real) #[batch_size, n_fft//2+1, frames]

    return log_amplitude, phase, real, imag


def get_dataset_filelist(input_training_wav_list,input_validation_wav_list):

    with open(input_training_wav_list, 'r') as fi:
        training_files = [x for x in fi.read().split('\n') if len(x) > 0]

    with open(input_validation_wav_list, 'r') as fi:
        validation_files = [x for x in fi.read().split('\n') if len(x) > 0]

    return training_files, validation_files


class Dataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                ):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft, self.hop_size, self.win_size = n_fft, hop_size, win_size
        self.num_mels, self.fmin, self.fmax = num_mels, fmin, fmax
        self.cached_wav, self.n_cache_reuse, self._cache_ref_count = None, n_cache_reuse, 0

    def __getitem__(self, index):
        """
        Returns:
            mel           :: (Freq, Frame) - Mel-frequency Log-Amplitude spectrogram
            log_amplitude :: (Freq, Frame)
            phase         :: (Freq, Frame)
            real          :: (Freq, Frame)
            imag          :: (Freq, Frame)
            audio         :: (T,)          - Waveform, in range [-1, 1]
        """
        # Load audio
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio = load_wav(filename, self.sampling_rate)
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        # audio :: (1, T) - Waveform in range [-1, 1]
        audio = torch.FloatTensor(audio).unsqueeze(0)
        if self.split:
            if audio.size(1) >= self.segment_size:
                max_audio_start = audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start: audio_start + self.segment_size] #[1,T]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        # mel :: (1, Freq, Frame) - Mel-frequency Log-Amplitude spectrogram
        mel = mel_spectrogram(audio, self.n_fft, self.num_mels, self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax)

        # :: (1, Freq, Frame)
        log_amplitude, phase, real, imag = amp_pha_specturm(audio, self.n_fft, self.hop_size, self.win_size)


        return (mel.squeeze(), log_amplitude.squeeze(), phase.squeeze(), real.squeeze(), imag.squeeze(), audio.squeeze(0))

    def __len__(self):
        return len(self.audio_files)
