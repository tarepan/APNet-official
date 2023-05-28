import random

import torch
from torch import Tensor
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
import librosa
from torchaudio.transforms import MelSpectrogram

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

    menizer = MelSpectrogram(sampling_rate, n_fft, win_size, hop_size, fmin, fmax, n_mels=num_mels, power=1, norm="slaney", mel_scale="slaney").to(y.device)
    melspec = menizer(y)
    logmelspec = torch.log(torch.clamp(melspec, min=1e-5))

    return logmelspec


def amp_pha_specturm(
        y,        #             :: (B, T)           - Waveforms
        n_fft,
        hop_size,
        win_size,
    ) -> tuple[
        Tensor, # log_amplitude :: (B, Freq, Frame) - Linear-frequency Log-Amplitude spectrogram
        Tensor, # phase         :: (B, Freq, Frame) - Phase spectrogram, in range [-pi, -pi]
        Tensor, # real          :: (B, Freq, Frame) - STFT real      value
        Tensor, # imag          :: (B, Freq, Frame) - STFT imaginary value
        ]:

    # :: (B, Freq, Frame, RealImag=2) - Linear-frequency Linear-Amplitude spectrogram
    hann_window=torch.hann_window(win_size).to(y.device)
    spec = torch.view_as_real(torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=True, return_complex=True))

    # real & imaginary value :: (B, Freq, Frame)
    real = spec[:, :, :, 0]
    imag = spec[:, :, :, 1]

    # :: (B, Freq, Frame) - Linear-frequency Log-Amplitude spectrogram
    log_amplitude = torch.log(torch.clamp(torch.sqrt(torch.pow(real,2)+torch.pow(imag,2)), min=1e-5))

    # Phase :: (B, Freq, Frame) - Angle on complex plane, in range [-pi, -pi]
    phase = torch.atan2(imag, real)

    return log_amplitude, phase, real, imag


def get_dataset_filelist(input_training_wav_list,input_validation_wav_list):

    with open(input_training_wav_list, 'r') as fi:
        training_files = [x for x in fi.read().split('\n') if len(x) > 0]

    with open(input_validation_wav_list, 'r') as fi:
        validation_files = [x for x in fi.read().split('\n') if len(x) > 0]

    return training_files, validation_files


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 training_files: list[str], # List of 16kHz audio file path
                 segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax,
                 split:   bool, # Whether to clip/pad audio into fixed segment size
                 shuffle: bool, # Wether to shuffle datum id (not per-epoch dataset shuffle config)
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

    def __getitem__(self, index):
        """
        Returns:
            mel           :: (Freq, Frame) - Mel-frequency    Log-Amplitude spectrogram
            log_amplitude :: (Freq, Frame) - Linear-frequency Log-Amplitude spectrogram
            phase         :: (Freq, Frame) - Phase spectrogram, in range [-pi, -pi]
            real          :: (Freq, Frame) - STFT real      value
            imag          :: (Freq, Frame) - STFT imaginary value
            audio         :: (T,)          - Waveform, in range [-1, 1]
        """

        # Load audio
        filename = self.audio_files[index]
        audio = librosa.load(filename, sr=self.sampling_rate, mono=True)[0]

        # audio :: (1, T) - Waveform in range [-1, 1]
        audio = torch.FloatTensor(audio).unsqueeze(0)
        ## Clipping | Padding
        if self.split:
            if audio.size(1) >= self.segment_size:
                max_audio_start = audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start: audio_start + self.segment_size] #[1,T]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        # TODO: preprocessing
        # mel :: (1, Freq, Frame) - Mel-frequency Log-Amplitude spectrogram
        mel = mel_spectrogram(audio, self.n_fft, self.num_mels, self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax)

        # STFT features :: all (1, Freq, Frame)
        log_amplitude, phase, real, imag = amp_pha_specturm(audio, self.n_fft, self.hop_size, self.win_size)

        return (mel.squeeze(), log_amplitude.squeeze(), phase.squeeze(), real.squeeze(), imag.squeeze(), audio.squeeze(0))

    def __len__(self):
        return len(self.audio_files)
