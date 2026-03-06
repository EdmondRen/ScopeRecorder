# %%
import math, sys, time
from importlib import reload
from tqdm import tqdm
import datetime

import joblib
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

import helper_visa as vs
import helper_fft
import user_io.IO_rigol as uio

address = "TCPIP::192.168.1.171::INSTR"
device = vs.connect(address = address, timeout=3_000, idn=False) # set 10 second timeout

#  ----------------------------------------------------------------------
# SETTINGS

CHANNELS = [1]
LENGTH = 524288 # 2**19
PREAMP_GAIN = 1

FFT_N_AVERAGE = 20 # MUST be less than 1000
SAVE_FFT = True
SAVE_TRACE = False

SAVE_DIR = r'C:\Users\Tom\Downloads\2026-2-noise-hunting\tuesday\\'
SAVE_FORMAT = "%Y%m%d_%H%M%S"

# ----------------------------------------------------------------------










RUN_TIME = 10000 if not SAVE_FFT else FFT_N_AVERAGE
TRACE_BUFFER = np.zeros([LENGTH, FFT_N_AVERAGE])

FFT_LENGTH = LENGTH//2+1
FFT_BINS = 1920

FFT_BUFFER = np.zeros([FFT_LENGTH, FFT_N_AVERAGE])
FFT_BUFFER_DISP_LEN = max(min(300, FFT_N_AVERAGE*3), 40)
FFT_BUFFER_DISP =  np.zeros([FFT_BINS, FFT_BUFFER_DISP_LEN])

bin_id, counts, f_centers, switch_frequency_ind = helper_fft.make_logbin_map(LENGTH, 1, n_log_bins=FFT_BINS, fmin=0, fmax=1/2)


SERIES_NUMBER = datetime.datetime.now().strftime(SAVE_FORMAT)
get_filename = lambda : SAVE_DIR + SERIES_NUMBER + ".joblib"
print("Series number", SERIES_NUMBER)


plt.ion() # Turn on interactive mode
fig, axs = plt.subplots(3,1, figsize = (8,7), gridspec_kw={"height_ratios": [0.5, 2, 1]},
                        constrained_layout=True)
color_background = (0.06,0.05,0.05)
fig.patch.set_facecolor(color_background)
for ax in axs:
    ax.set_facecolor(color_background)

# Trace data
x_data, y_data = np.arange(LENGTH), np.zeros(LENGTH)
line, = axs[0].plot(x_data, y_data, 'C0', alpha=0.9, linewidth=1) # Create an empty line object

# FFT data
x_data2, y_data2 = np.arange(FFT_LENGTH)+1, np.arange(FFT_LENGTH)+1
linefft, = axs[1].plot(x_data2, y_data2, color="orangered", alpha=1, linewidth=1) # Create an empty line object
linefft2, = axs[1].plot(x_data2, y_data2, color="orangered", alpha=0.3) # Create an empty line object
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].grid(which="both", alpha=0.2)
axs[1].set_ylabel(r"Noise [V/$\sqrt{Hz}$]")
axs[2].set_xlabel("Frequency [Hz]")


# Waterfall data
im = axs[2].imshow(
    FFT_BUFFER_DISP.T,
    origin="lower",
    aspect="auto",
    interpolation="nearest",
    vmin=-0.1, vmax=0.1, 
    cmap="magma"
)

for i in range(RUN_TIME):
    print(f"{i}/{RUN_TIME}", end="\r")
    try:
        rawdata, time_series = uio.read_waveform(device, read_channel=CHANNELS, acquire_length=LENGTH, calibrate=True)   
    except KeyboardInterrupt:
        sys.exit(0)
    except:
        rawdata, time_series = uio.read_waveform(device, read_channel=CHANNELS, acquire_length=LENGTH, calibrate=True)
        
    rawdata[CHANNELS[0]] /= PREAMP_GAIN
    if i<FFT_N_AVERAGE:
        TRACE_BUFFER[:,i] = rawdata[CHANNELS[0]]
        
    dt = time_series[1]-time_series[0]
    fft = np.fft.rfft(rawdata[CHANNELS[0]])
    fft = (fft.real*fft.real + fft.imag*fft.imag)/len(fft)*dt
    fft_freq = np.fft.rfftfreq(len(rawdata[CHANNELS[0]]), time_series[1]-time_series[0])
    fft_log_power = helper_fft.rebin_to_log(fft, bin_id, counts, f_centers, switch_frequency_ind)
    
    if any(fft==np.inf):
        continue
    
    FFT_BUFFER[:, :-1] = FFT_BUFFER[:, 1:]   # shift left
    FFT_BUFFER[:, -1] = fft
    FFT_BUFFER_DISP[:, :-1] = FFT_BUFFER_DISP[:, 1:]   # shift left
    FFT_BUFFER_DISP[:, -1] = np.log10(np.sqrt(fft_log_power))*20
    
    if i==0:
        line.set_xdata(time_series-time_series[0])
        linefft.set_xdata(fft_freq)
        linefft2.set_xdata(fft_freq)
        axs[0].set_xlim([0, -time_series[0]+time_series[-1]])
        axs[1].set_xlim([fft_freq[1], fft_freq[-1]])
        FFT_BUFFER_DISP[:, :-1] = np.min(FFT_BUFFER_DISP[:, -1])
        
        # Set ticks for spectrogram
        freq_ticks = axs[1].get_xticks()          # numeric tick positions (Hz)
        minor_freq_ticks = axs[1].xaxis.get_minorticklocs()
        freq_labels = [t.get_text() for t in axs[1].get_xticklabels()]
        freq_labels[0] = freq_labels[1] = freq_labels[-1] = ""
        
        # Map each tick frequency to nearest row
        row_ticks = np.searchsorted(f_centers/dt, freq_ticks)
        row_ticks = np.clip(row_ticks, 0, len(f_centers)-1)
        row_minor = np.searchsorted(f_centers, minor_freq_ticks)
        row_minor = np.clip(row_minor, 0, len(f_centers) - 1)
        axs[2].set_xticks(row_ticks)
        axs[2].set_xticks(row_minor, minor=True)
        axs[2].set_xticklabels(freq_labels)
        cbar = fig.colorbar(im, ax=axs, location="right", pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        

    # Update 1
    line.set_ydata(rawdata[CHANNELS[0]]) # Update the line object's y data
    axs[0].set_ylim([min(rawdata[CHANNELS[0]]),max(rawdata[CHANNELS[0]])]) # Update the line object's y data\
    
    noise_range_dB = [np.floor(np.min(FFT_BUFFER_DISP[10:,:])/10)*10,np.ceil(np.max(FFT_BUFFER_DISP[10:,:])/10)*10]
    noise_range_dB[0] = max(-190,noise_range_dB[0])
    
    # Update 2
    n_avg = FFT_N_AVERAGE if i>FFT_N_AVERAGE else i
    fft_avg = np.sqrt(np.mean(FFT_BUFFER[:,-n_avg:], axis=1))
    linefft.set_ydata(fft_avg)
    if i>0:
        linefft2.set_ydata(np.sqrt(FFT_BUFFER[:,-2]))
            
    # axs[1].set_ylim([np.log10(min(fft_avg[1:])),max(fft_avg[1:])]) # Update the line object's y data\
    axs[1].set_ylim(10**(noise_range_dB[0]/20), 10**(noise_range_dB[1]/20))

    
    # Update 3: spectrogram
    im.set_data(FFT_BUFFER_DISP.T)
    im.set_clim(*noise_range_dB)
    cbar.update_normal(im)      # refresh colorbar
    
    # Colorbar spanning all subplots
    cbar.ax.set_title("   [dBV]", pad=6)
    
    fig.suptitle(f"{SERIES_NUMBER}, avg={n_avg}")
        
    # Redraw and flush events
    fig.canvas.draw()
    fig.canvas.flush_events() 
    time.sleep(0.1) # Pause for a short interval    

result_psd_avg = {"freq": fft_freq, "psd": fft_avg}

# Save
plt.savefig(f"{SAVE_DIR}{SERIES_NUMBER}_plots.pdf")

if SAVE_FFT:
    joblib.dump(result_psd_avg, f"{SAVE_DIR}{SERIES_NUMBER}_psd.joblib")
    print("FFT saved:", f"{SAVE_DIR}{SERIES_NUMBER}_psd.joblib")
    
if SAVE_TRACE:
    joblib.dump(TRACE_BUFFER, f"{SAVE_DIR}{SERIES_NUMBER}_trace.joblib")
    print("Traces saved:", f"{SAVE_DIR}{SERIES_NUMBER}_trace.joblib")
    

