from dataclasses import dataclass
import glob
import os
import shutil

import matplotlib
import torch
from torch.nn.utils import weight_norm
import matplotlib.pylab as plt


matplotlib.use("Agg")


@dataclass
class GlobalConf:
    input_training_wav_list:   str   # -
    input_validation_wav_list: str   # -
    n_train:                   int   # -
    n_test:                    int   # -
    data_root:                 str   # -
    test_input_wavs_dir:       str   # -
    test_input_mels_dir:       str   # -
    test_mel_load:             int   # -
    test_output_dir:           str   # -

    batch_size:                int   # ✔
    learning_rate:             float # ✔
    adam_b1:                   float # ✔
    adam_b2:                   float # ✔
    lr_decay:                  float # ✔
    seed:                      int   # -
    training_epochs:           int   # ✔

    stdout_interval:           int   # -
    checkpoint_interval:       int   # -
    summary_interval:          int   # -
    validation_interval:       int   # -

    checkpoint_path:           str   # -
    checkpoint_file_load:      str   # -

    ASP_channel:                   int                                                                     # ✔
    ASP_resblock_kernel_sizes:     tuple[int, int, int]                                                    # ✔
    ASP_resblock_dilation_sizes:   tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] # ✔
    ASP_input_conv_kernel_size:    int
    ASP_output_conv_kernel_size:   int
    PSP_channel:                   int                                                                     # ✔
    PSP_resblock_kernel_sizes:     tuple[int, int, int]                                                    # ✔
    PSP_resblock_dilation_sizes:   tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] # ✔
    PSP_input_conv_kernel_size:    int
    PSP_output_R_conv_kernel_size: int
    PSP_output_I_conv_kernel_size: int

    unified_mainnet:               bool # Whether to use ASP/PSP unified generator
    causal:                        bool # Whether to use causal conv
    use_fc:                        bool # Whether to use FC
    wo_d:                          bool # Whether to train without Discriminator
    loss_on_wave:                  bool # Whether to use losses based on waveform
    segment_size:  int # ✔
    n_fft:         int # ✔
    hop_size:      int # ✔
    win_size:      int # ✔
    sampling_rate: int # ✔
    num_mels:      int # ✔
    fmin:          int
    fmax:          int
    num_workers:   int


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config: str, config_name: str, path: str):
    """
    Args:
        config - Path to the config file
        config_name - New config file name
        path - New config file's parent directory path
    """
    t_path = os.path.join(path, config_name)
    # If specify same config, pass
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, t_path)


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)




def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

