import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from speechcorpusy import load_preset

from dataset import Dataset, mel_spectrogram, amp_pha_specturm, get_dataset_filelist
from models import Generator, UnifiedGenerator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss, amplitude_loss, phase_loss, STFT_consistency_loss
from utils import AttrDict, GlobalConf, build_env, plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint


torch.backends.cudnn.benchmark = True


def train(h: GlobalConf):

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:0')

    # Model
    generator =               Generator(h).to(device) if not h.unified_mainnet else UnifiedGenerator(h).to(device)
    mpd       = MultiPeriodDiscriminator().to(device)
    msd       =  MultiScaleDiscriminator().to(device)
    print(generator)

    # State
    os.makedirs(h.checkpoint_path, exist_ok=True)
    print("checkpoints directory : ", h.checkpoint_path)
    if os.path.isdir(h.checkpoint_path):
        cp_g  = scan_checkpoint(h.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(h.checkpoint_path, 'do_')
    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g  = load_checkpoint(cp_g,  device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    # Optim/Sched
    optim_g = torch.optim.AdamW(generator.parameters(),                              h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    # Data
    ## Path list
    if h.input_training_wav_list[:8] != "CORPUSY:":
        training_filelist, validation_filelist = get_dataset_filelist(h.input_training_wav_list, h.input_validation_wav_list)
    else:
        # e.g. `CORPUSY:LJ`
        corpus = load_preset(h.input_training_wav_list[8:], root=h.data_root, download=False)
        corpus.get_contents()
        all_uttr_paths = list(map(lambda id: corpus.get_item_path(id).with_suffix(".16k.wav"), corpus.get_identities()))
        training_filelist, validation_filelist = all_uttr_paths[:h.n_train], all_uttr_paths[h.n_train:-1*h.n_test]
    ## Dataset/Loader
    trainset = Dataset(training_filelist,   h.segment_size, h.n_fft, h.num_mels, h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, split=True,  shuffle=True)
    validset = Dataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels, h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, split=False, shuffle=False)
    # TODO: shuffle in training
    train_loader      = DataLoader(trainset, num_workers=h.num_workers, shuffle=False, batch_size=h.batch_size, pin_memory=True, drop_last=True)
    validation_loader = DataLoader(validset, num_workers=1,             shuffle=False, batch_size=1,            pin_memory=True, drop_last=True)

    # Logger
    sw = SummaryWriter(os.path.join(h.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()

    for epoch in range(max(0, last_epoch), h.training_epochs):
        #### Epoch ##############################################################################################################################

        start = time.time()
        print(f"Epoch: {epoch+1}")

        for batch in train_loader:
            #### Step #######################################################################################################################
            start_b = time.time()

            # Data
            x, logamp_gt, phase_gt, real_gt, imag_gt, wave_gt = batch
            x         =         x.to(device, non_blocking=True) # :: (B, Freq, Frame) -              Mel-frequency    Log-Amplitude spectrogram
            logamp_gt = logamp_gt.to(device, non_blocking=True) # :: (B, Freq, Frame) - Ground-truth Linear-frequency Log-Amplitude spectrogram
            phase_gt  =  phase_gt.to(device, non_blocking=True) # :: (B, Freq, Frame) - Ground-truth Phase spectrogram, in range [-pi, -pi]
            real_gt   =   real_gt.to(device, non_blocking=True) # :: (B, Freq, Frame) - Ground-truth STFT real      part
            imag_gt   =   imag_gt.to(device, non_blocking=True) # :: (B, Freq, Frame) - Ground-truth STFT imaginary part
            wave_gt   =   wave_gt.to(device, non_blocking=True) # :: (B, T)           - Ground-truth waveform, in range [-1, 1]

            # Reshape
            wave_gt = wave_gt.unsqueeze(1) # :: (B, 1, T)

            # Common_Forward - Properties yield from the network directly
            logamp_pred_direct, phase_pred_direct, real_pred_direct, imag_pred_direct, wave_pred = generator(x)

            # Discriminator
            if not h.wo_d:
                optim_d.zero_grad()
                # D_Forward
                y_df_hat_r, y_df_hat_g, _, _ = mpd(wave_gt, wave_pred.detach())
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(wave_gt, wave_pred.detach())
                # D_Loss
                loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
                loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
                L_D = loss_disc_s + loss_disc_f
                # D_Backward/Optim
                L_D.backward()
                optim_d.step()

            # Generator
            optim_g.zero_grad()

            # Properties extracted from predicted waveform (!= predicted STFT, from which predicted waveform is generated by iSTFT)
            logamp_pred_wave, phase_pred_wave, real_pred_wave, imag_pred_wave = amp_pha_specturm(wave_pred.squeeze(1), h.n_fft, h.hop_size, h.win_size)

            # Loss mode - loss on direct network output STFT OR on final waveform
            logamp_pred = logamp_pred_direct if h.loss_on_wave else logamp_pred_wave
            phase_pred  = phase_pred_direct  if h.loss_on_wave else phase_pred_wave

            # G_Loss
            ## log-Amplitude spectra
            L_A = amplitude_loss(logamp_gt, logamp_pred)
            ## Phase spectra
            L_IP, L_GD, L_PTD = phase_loss(phase_gt, phase_pred, h.n_fft, phase_gt.size()[-1])
            L_P = L_IP + L_GD + L_PTD
            ## STFT spectra
            if not h.loss_on_wave:
                L_C = STFT_consistency_loss(real_pred_direct, real_pred_wave, imag_pred_direct, imag_pred_wave)
                L_R = F.l1_loss(real_gt, real_pred_direct)
                L_I = F.l1_loss(imag_gt, imag_pred_direct)
            w_ri = 2.25
            L_S = (L_C + w_ri * (L_R + L_I)) if not h.loss_on_wave else 0
            ## Waveform
            if not h.wo_d:
                _, y_df_g, fmap_f_r, fmap_f_g = mpd(wave_gt, wave_pred)
                _, y_ds_g, fmap_s_r, fmap_s_g = msd(wave_gt, wave_pred)
                ### Feature matching loss
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                L_FM = loss_fm_s + loss_fm_f
                ### Adversarial loss
                loss_gen_f, _ = generator_loss(y_df_g)
                loss_gen_s, _ = generator_loss(y_ds_g)
                L_GAN_G = loss_gen_s + loss_gen_f
            ### Mel loss
            y_g_mel = mel_spectrogram(wave_pred.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
            L_Mel = F.l1_loss(x, y_g_mel)
            ###
            w_mel = 45
            L_W = L_GAN_G + L_FM + w_mel * L_Mel if not h.wo_d else w_mel * L_Mel

            ## Total
            w_a, w_p, w_s = 45, 100, 20
            L_G = w_a * L_A + w_p * L_P + w_s * L_S + L_W
            # G_Backward/Optim
            L_G.backward()
            optim_g.step()

            # Logging
            if steps % h.stdout_interval == 0:
                with torch.no_grad():
                    A_error = amplitude_loss(logamp_gt, logamp_pred).item()
                    IP_error, GD_error, PTD_error = phase_loss(phase_gt, phase_pred, h.n_fft, phase_gt.size()[-1])
                    IP_error, GD_error, PTD_error = IP_error.item(), GD_error.item(), PTD_error.item()
                    C_error = STFT_consistency_loss(real_pred_direct, real_pred_wave, imag_pred_direct, imag_pred_wave).item() if not h.loss_on_wave else 0
                    R_error = F.l1_loss(real_gt, real_pred_direct).item()                                                      if not h.loss_on_wave else 0
                    I_error = F.l1_loss(imag_gt, imag_pred_direct).item()                                                      if not h.loss_on_wave else 0
                    Mel_error = F.l1_loss(x, y_g_mel).item()

                print('Steps : {:d}, Gen Loss Total : {:4.3f}, Amplitude Loss : {:4.3f}, Instantaneous Phase Loss : {:4.3f}, Group Delay Loss : {:4.3f}, Phase Time Difference Loss : {:4.3f}, STFT Consistency Loss : {:4.3f}, Real Part Loss : {:4.3f}, Imaginary Part Loss : {:4.3f}, Mel Spectrogram Loss : {:4.3f}, s/b : {:4.3f}'.
                      format(steps, L_G, A_error, IP_error, GD_error, PTD_error, C_error, R_error, I_error, Mel_error, time.time() - start_b))

            # checkpointing
            if steps % h.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = "{}/g_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path, {'generator': generator.state_dict()})
                checkpoint_path = "{}/do_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path, 
                                {'mpd': mpd.state_dict(),
                                 'msd': msd.state_dict(),
                                 'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                 'epoch': epoch})

            # Tensorboard summary logging
            if steps % h.summary_interval == 0:
                sw.add_scalar("Training/Generator_Total_Loss", L_G, steps)
                sw.add_scalar("Training/Mel_Spectrogram_Loss", Mel_error, steps)

            # Validation
            if steps % h.validation_interval == 0:  # and steps != 0:
                generator.eval()
                torch.cuda.empty_cache()

                # Losses
                val_A_err_tot   = 0
                val_IP_err_tot  = 0
                val_GD_err_tot  = 0
                val_PTD_err_tot = 0
                val_C_err_tot   = 0
                val_R_err_tot   = 0
                val_I_err_tot   = 0
                val_Mel_err_tot = 0

                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):

                        # Data
                        x, logamp_gt, phase_gt, real_gt, imag_gt, wave_gt = batch
                        x         =         x.to(device, non_blocking=True) # :: (B=1, Freq, Frame) -              Mel-frequency    Log-Amplitude spectrogram
                        logamp_gt = logamp_gt.to(device, non_blocking=True) # :: (B=1, Freq, Frame) - Goundr-truth Linear-frequency Log-Amplitude spectrogram
                        phase_gt  =  phase_gt.to(device, non_blocking=True) # :: (B=1, Freq, Frame) - Goundr-truth Phase spectrogram, in range [-pi, -pi]
                        real_gt   =   real_gt.to(device, non_blocking=True) # :: (B=1, Freq, Frame) - Goundr-truth STFT real      value
                        imag_gt   =   imag_gt.to(device, non_blocking=True) # :: (B=1, Freq, Frame) - Goundr-truth STFT imaginary value
                        wave_gt   =   wave_gt.to(device, non_blocking=True) # :: (B=1, T)           - Goundr-truth waveform, in range [-1, 1]

                        # Forward
                        logamp_pred_direct, phase_pred_direct, real_pred_direct, imag_pred_direct, wave_pred = generator(x)

                        # Transform
                        y_g_mel = mel_spectrogram(wave_pred.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,h.hop_size, h.win_size,h.fmin, h.fmax)
                        logamp_pred_wave, phase_pred_wave, real_pred_wave, imag_pred_wave = amp_pha_specturm(wave_pred.squeeze(1), h.n_fft, h.hop_size, h.win_size)

                        # Loss mode - loss on direct network output STFT OR on final waveform
                        logamp_pred = logamp_pred_direct if h.loss_on_wave else logamp_pred_wave
                        phase_pred  = phase_pred_direct  if h.loss_on_wave else phase_pred_wave

                        # Loss
                        val_A_err_tot   += amplitude_loss(logamp_gt, logamp_pred).item()
                        val_IP_err, val_GD_err, val_PTD_err = phase_loss(phase_gt, phase_pred, h.n_fft, phase_gt.size()[-1])
                        val_IP_err_tot  += val_IP_err.item()
                        val_GD_err_tot  += val_GD_err.item()
                        val_PTD_err_tot += val_PTD_err.item()
                        val_C_err_tot   += STFT_consistency_loss(real_pred_direct, real_pred_wave, imag_pred_direct, imag_pred_wave).item() if not h.loss_on_wave else 0
                        val_R_err_tot   += F.l1_loss(real_gt, real_pred_direct).item()                                                      if not h.loss_on_wave else 0
                        val_I_err_tot   += F.l1_loss(imag_gt, imag_pred_direct).item()                                                      if not h.loss_on_wave else 0
                        val_Mel_err_tot += F.l1_loss(x, y_g_mel).item()

                        # Log Audio & Mel image
                        if j <= 4:
                            # Transform
                            y_g_spec = mel_spectrogram(wave_pred.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
                            # Ground-Truth (only global step 0)
                            if steps == 0:
                                sw.add_audio(f'gt/y_{j}',                              wave_gt[0].cpu().numpy(),  steps, h.sampling_rate)
                                sw.add_figure(f'gt/y_spec_{j}',      plot_spectrogram(       x[0].cpu().numpy()), steps)
                            sw.add_audio(f'generated/y_g_{j}',                       wave_pred[0].cpu().numpy(),  steps, h.sampling_rate)
                            sw.add_figure(f'generated/y_g_spec_{j}', plot_spectrogram(y_g_spec[0].cpu().numpy()), steps)

                    # Average
                    val_A_err   =   val_A_err_tot / (j+1)
                    val_IP_err  =  val_IP_err_tot / (j+1)
                    val_GD_err  =  val_GD_err_tot / (j+1)
                    val_PTD_err = val_PTD_err_tot / (j+1)
                    val_C_err   =   val_C_err_tot / (j+1)
                    val_R_err   =   val_R_err_tot / (j+1)
                    val_I_err   =   val_I_err_tot / (j+1)
                    val_Mel_err = val_Mel_err_tot / (j+1)

                    # Log losses
                    sw.add_scalar("Validation/Amplitude_Loss",             val_A_err,   steps)
                    sw.add_scalar("Validation/Instantaneous_Phase_Loss",   val_IP_err,  steps)
                    sw.add_scalar("Validation/Group_Delay_Loss",           val_GD_err,  steps)
                    sw.add_scalar("Validation/Phase_Time_Difference_Loss", val_PTD_err, steps)
                    sw.add_scalar("Validation/STFT_Consistency_Loss",      val_C_err,   steps)
                    sw.add_scalar("Validation/Real_Part_Loss",             val_R_err,   steps)
                    sw.add_scalar("Validation/Imaginary_Part_Loss",        val_I_err,   steps)
                    sw.add_scalar("Validation/Mel_Spectrogram_loss",       val_Mel_err, steps)

                generator.train()

            steps += 1
            #### /Step ######################################################################################################################

        scheduler_g.step()
        scheduler_d.step()

        print(f'Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n')
        #### /Epoch #############################################################################################################################


def main():
    print('Initializing Training Process..')

    # Load config
    config_file = 'config.json'
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h: GlobalConf = AttrDict(json_config)

    # Save config
    build_env(config_file, 'config.json', h.checkpoint_path)

    # Seeds
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)

    # Run
    train(h)


if __name__ == '__main__':
    main()
