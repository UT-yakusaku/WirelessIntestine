import os
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 
import h5py
from scipy.signal import stft
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Input, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm
import os
from scipy import signal
import struct

def loadData(path):
    data = []
    time = []
    time_diff_ns = 0
    nrow = 0
    time_top = 0
    rowNo_top = 0
    rowNo_loop_cnt = 0
    with open(path, 'rb') as f:
        f.seek(0, os.SEEK_SET)
        if f.read(3).decode("ascii") != "DAT":
            raise Exception
            return
        f.seek(4, os.SEEK_SET)
        nrow = struct.unpack('<L', f.read(4))[0]
        time_diff_ns = struct.unpack('<L', f.read(4))[0]
        time_diff_ms = time_diff_ns // 1000
        f.seek(32, os.SEEK_SET)
        time_top = time_diff_ms
        buf = struct.unpack('>9h', f.read(18))
        rowNo = struct.unpack('<3H', f.read(6))
        print(buf)
        rowNo_top = rowNo[0]
        for j in range(9):
            time.append(time_top + 10000 * j)
            data.append(buf[j])
        while True:
            try:
                buf = struct.unpack('>9h', f.read(18))
                rowNo = struct.unpack('<3H', f.read(6))
            except:
                break
            if rowNo[0] + rowNo_loop_cnt * 65536 < rowNo_top:
                rowNo_loop_cnt += 1
            time_top += (rowNo[0] + rowNo_loop_cnt * 65536 - rowNo_top) * 90000
            rowNo_top = rowNo[0] + rowNo_loop_cnt * 65536
            for j in range(9):
                time.append(time_top + 10000 * j)
                data.append(buf[j])
    return data, time

def convertVolt(data, amp_resistance):
    adc_multiplier = 0.125
    amp_gain = 1.0 + (100000.0 / amp_resistance)
    volt_coeff = adc_multiplier / amp_gain
    return data * volt_coeff

def bandpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2
    wp = fp / fn
    ws = fs / fn
    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(N, Wn, "band")
    y = signal.filtfilt(b, a, x)
    return y   

# Constants
SAMPLE_RATE = 100  # Samples per second
DURATION = 2000 # Duration in seconds
FREQ = 0.6  # Target frequency in Hz
NOISE_COEFF = (0.8, 1.2)
CHUNK_SIZE = 1024
FFT_SIZE = 1024
MODEL_FILE = "autoencoder_params.keras"
HDF5_OUTPUT = "denoised_output.h5"

# Define formatting constants (in points and mm)
MM_TO_PT = 2.83465
MM_TO_INCH = 25.4
AXIS_LINEWIDTH = 0 * MM_TO_PT # replace with value
ERRORBAR_LINEWIDTH = 0 * MM_TO_PT # replace with value
TICK_LENGTH = 0 * MM_TO_PT # replace with value
TICK_WIDTH = 0 * MM_TO_PT # replace with value
SCALEBAR_WIDTH = 0 * MM_TO_PT # replace with value
RASTER_WIDTH = 0 * MM_TO_PT # replace with value
RASTER_LENGTH = 0 * MM_TO_PT # replace with value
TRACE_WIDTH = 0 * MM_TO_PT # replace with value
DOT_SIZE = 0 * MM_TO_PT # replace with value

# Font sizes
AXIS_TITLE_SIZE = 8
TICK_LABEL_SIZE = 7
INSET_LABEL_SIZE = 6
FONT_FAMILY = 'Arial'

# Define custom colors
COLORS = {
    'black': (0, 0, 0),
}


# Function to generate a single synthetic AM signal
def generate_am_signal(duration, fs, mod_freq, noise_coeff):
    t = []
    am_signal = []
    for i in range(10):
        t_ = np.linspace(0, duration // 10, int(fs * duration // 10))

        # Modulating wave
        mod_freq = np.random.uniform(0.001, 0.02)
        mod_wave = (np.sin(2 * np.pi * mod_freq * t_) + 1) / 2  # Normalize between 0 and 1
        mod_wave = mod_wave * mod_wave
        
        # Carrier wave (randomized frequency)
        carrier_freq = np.random.uniform(0.5, 0.7)  # Random carrier frequency between 5 Hz and 10 Hz
        carrier_wave = np.sin(2 * np.pi * carrier_freq * t_)

        # Amplitude-modulated signal
        am_signal_ = mod_wave * carrier_wave

        am_signal_random = []
        state = 0 # 0 for cut, 1 for duplicate
        for am in am_signal_:
            if np.random.uniform(0, 1) < 0.3:
                if state == 0:
                    state = 1
                    continue
                else:
                    state = 0
                    am_signal_random.append(am)
            am_signal_random.append(am)

        if len(am_signal_) > len(am_signal_random):
            am_signal_random.append(am_signal_random[-1])
        am_signal_ = np.array(am_signal_random) / 100
        t.append(t_)
        am_signal.append(am_signal_ * 2 ** np.random.uniform(-10, 0.2))

    t = np.hstack(t)
    am_signal = np.hstack(am_signal)
    # Add pink noise
    coeff = 1 if np.random.random() > 0.15 else 5
    pink_noise_data, pink_noise_05 = pink_noise(len(t))
    pink_noise_data /= 100
    pink_noise_05 /= 100
    noisy_signal = am_signal / coeff + 0.8 * coeff * pink_noise_05
    noise = 0.8 * coeff * pink_noise_05 - 0.1 * coeff * pink_noise_data
    am_signal = am_signal / coeff + 0.1 * coeff * pink_noise_data
    return noisy_signal * 200, am_signal * 200, noise * 200


# Function to generate pink noise
def pink_noise(N):
    uneven = N % 2
    X = np.random.randn(N // 2 + 1 + uneven) + 1j * np.random.randn(N // 2 + 1 + uneven)
    S = np.sqrt(1.0 / (np.arange(len(X)) + 0.68))  # 1/f spectrum
    S_noise = np.sqrt(1.0 / (np.arange(len(X)) + 0.53))  # 1/f spectrum
    y = (np.fft.irfft(X * S)).real
    y_noise = (np.fft.irfft(X * S_noise)).real
    if uneven:
        y = y[:-1]
        y_noise = y_noise[:-1]
    return y / np.std(y), y_noise / np.std(y_noise)

# Create the autoencoder model
def create_autoencoder(input_shape):
    input_layer = Input(shape=input_shape)
    x = Reshape((input_shape[0], input_shape[1], 1))(input_layer)  # Reshape for 2D input
    
    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Bottleneck
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(1, (3, 3), activation='linear', padding='same')(x)

    output_layer = Reshape((input_shape[0], input_shape[1]))(x)

    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder

# Short-Time Fourier Transform (STFT) with frequency limitation
def apply_stft(data, n_fft=FFT_SIZE, freq_range=(0.3, 2.0)):
    f, t, Zxx = stft(data, fs=SAMPLE_RATE, nperseg=n_fft)
    
    # Find indices corresponding to the desired frequency range
    freq_min_idx = np.searchsorted(f, freq_range[0])
    
    # Slice the frequency and STFT arrays
    f = f[freq_min_idx:freq_min_idx + 64]
    Zxx = Zxx[freq_min_idx:freq_min_idx + 64, :]
    
    return f, t, Zxx

# Train or load autoencoder
def train_or_load_autoencoder(noisy_data, clean_data, hdf5_file):
    input_shape = (64, 64)
    if os.path.exists(hdf5_file):
        print("Loading pre-trained model...")
        autoencoder = load_model(hdf5_file)
    else:
        print("Training autoencoder...")
        autoencoder = create_autoencoder(input_shape)
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        autoencoder.fit(noisy_data, clean_data, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=1)
        autoencoder.save(hdf5_file)
    return autoencoder

def flatten_matrix(output_data):
    Zxx_denoised = []
    for i in range(output_data.shape[0]):
        Zxx_denoised.append(output_data[i])
    Zxx_denoised = np.hstack(Zxx_denoised)
    return Zxx_denoised

# Process real data chunks
def process_real_data(data, autoencoder):
    denoised_chunks = []
    f = []
    t_chunks = []
    f, t, Zxx = apply_stft(data)

    Zxx_magnitude = np.abs(Zxx)
    input_data = np.array([Zxx_magnitude[:, i:i+64] for i in range(0, Zxx_magnitude.shape[1] - 63, 64)])
    input_data = input_data.reshape(-1, 64, 64, 1)  # Add channel dimension for Conv2D
    output_data = autoencoder.predict(input_data)
    Zxx_denoised = []
    for i in range(output_data.shape[0]):
        Zxx_denoised.append(output_data[i])
    Zxx_denoised = np.hstack(Zxx_denoised)

    return f, t, Zxx_magnitude, Zxx_denoised

# Save results to HDF5
def save_results_to_hdf5(frequencies, times, denoised_data, file_name):
    with h5py.File(file_name, 'w') as h5_file:
        h5_file.create_dataset('frequencies', data=frequencies)
        h5_file.create_dataset('times', data=times)
        h5_file.create_dataset('denoised_data', data=denoised_data)

def plot_06(f, t, Zxx_orig, Zxx_denoised):
    fig, axs = plt.subplots(2, 1, figsize=(100 / MM_TO_INCH, 80 / MM_TO_INCH))
    freq_min_idx = np.searchsorted(f, 0.4)
    freq_max_idx = np.searchsorted(f, 0.8)
    Zxx_orig_sum = np.sum(Zxx_orig[freq_min_idx:freq_max_idx], axis=0)
    Zxx_denoised_sum = np.sum(Zxx_denoised[freq_min_idx:freq_max_idx], axis=0)
    axs[0].plot(t / 60, Zxx_orig_sum, linewidth=TRACE_WIDTH)
    axs[0].set_title("Original Data", fontsize=AXIS_TITLE_SIZE, family=FONT_FAMILY)
    t = t[:Zxx_denoised.shape[1]]
    axs[1].plot(t / 60, Zxx_denoised_sum, linewidth=TRACE_WIDTH)
    axs[1].set_title("Denoised Data", fontsize=AXIS_TITLE_SIZE, family=FONT_FAMILY)

    for ax in axs:
        ax.set_ylabel("Amplitude [mV]", fontsize=TICK_LABEL_SIZE, family=FONT_FAMILY)
        ax.set_xlabel("Time [min]", fontsize=TICK_LABEL_SIZE, family=FONT_FAMILY)
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE, width=TICK_WIDTH, length=TICK_LENGTH)
    plt.tight_layout()
    plt.show()

# Visualization
def plot_scalograms(f, t, Zxx_orig, Zxx_denoised, Zxx_noise):
    fig, axs = plt.subplots(3, 1, figsize=(26 / MM_TO_INCH, 80 / MM_TO_INCH))

    length = Zxx_denoised.shape[1]
    t = t[:length]
    cs0 = axs[0].pcolormesh(t / 60, f, np.abs(Zxx_orig.squeeze()[:, :(length)]), shading='gouraud', cmap='jet', norm=LogNorm(vmin=3*1e-3, vmax=8*1e-1))
    axs[0].set_title("Original Data", fontsize=AXIS_TITLE_SIZE, family=FONT_FAMILY)

    cs1 = axs[1].pcolormesh(t / 60, f, np.abs(Zxx_denoised.squeeze()[:, :length]), shading='gouraud', cmap='jet', norm=LogNorm(vmin=3*1e-3, vmax=8*1e-1))
    axs[1].set_title("Denoised Data", fontsize=AXIS_TITLE_SIZE, family=FONT_FAMILY)

    cs2 = axs[2].pcolormesh(t / 60, f, np.abs(Zxx_noise.squeeze()[:, :length]), shading='gouraud', cmap='jet', norm=LogNorm(vmin=3*1e-3, vmax=8*1e-1))
    axs[2].set_title("Noise", fontsize=AXIS_TITLE_SIZE, family=FONT_FAMILY)

    for ax in axs:
        ax.set_ylabel("Frequency [Hz]", fontsize=TICK_LABEL_SIZE, family=FONT_FAMILY)
        ax.set_xlabel("Time [min]", fontsize=TICK_LABEL_SIZE, family=FONT_FAMILY)
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE, width=TICK_WIDTH, length=TICK_LENGTH)
        ax.set_ylim([0.3, 1.8])
        #ax.set_xlim([330, 550])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig('figure_output_scalogram_model.tiff', format='tiff', dpi=300)
    plt.show()


def motion_levels(video):

    vid = np.loadtxt(video, delimiter=',')   # Analyze the first 30 seconds, skipping 5 frames each time
    nbins = 800
    vid = np.array(vid) / 1000000
    duration = 5

    vid_ = []
    for i in range(len(vid)//duration):
        vid_.append(np.sum(vid[i * duration: i * duration + duration]) / duration)
    vid = np.array(vid_)

    hist, bin_edges = np.histogram(vid, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit a Gaussian Mixture Model (GMM) with 2 components
    gmm = GaussianMixture(n_components=2, random_state=42)
    data_reshaped = vid.reshape(-1, 1)
    gmm.fit(data_reshaped)

    # Predict cluster membership
    labels = gmm.predict(data_reshaped)
    means = gmm.means_.flatten()

    # Determine which cluster corresponds to the Gaussian core
    core_cluster = np.argmin(means)  # Cluster with the smaller mean
    outlier_cluster = 1 - core_cluster

    # Find the threshold (minimum value in the outlier cluster)
    threshold = vid[labels == outlier_cluster].min()

    print("The percentage of values over threshold:", len(vid[vid > threshold]) / len(vid) * 100)

    # Plot the time series with shading for values over the threshold
    time = np.arange(len(vid))  # Create a time array
    motion = [1 if _ > threshold else 0 for _ in vid]

    return time * 6 * duration / 60, motion

def calculate_weighted_means(orig, motion, orig_interval=10.24, motion_interval=30):
    # Calculate the number of elements in orig and motion
    num_orig = len(orig)
    num_motion = len(motion)

    # Initialize an array to store the weighted means
    weighted_means = np.zeros(num_motion)

    for i in range(num_motion):
        start_motion = i * motion_interval
        end_motion = (i + 1) * motion_interval

        # Find the overlapping intervals in `orig`
        start_orig_idx = int(start_motion // orig_interval)
        end_orig_idx = int(end_motion // orig_interval)

        # Collect weighted values
        weighted_sum = 0
        total_weight = 0

        for j in range(start_orig_idx, end_orig_idx + 1):
            # Calculate overlap duration
            start_orig = j * orig_interval
            end_orig = (j + 1) * orig_interval
            overlap_start = max(start_motion, start_orig)
            overlap_end = min(end_motion, end_orig)

            if overlap_start < overlap_end:  # Check if there's an overlap
                weight = overlap_end - overlap_start
                weighted_sum += orig[j] * weight
                total_weight += weight

        # Calculate the weighted mean for the motion window
        weighted_means[i] = weighted_sum / total_weight if total_weight > 0 else 0

    weighted_means = weighted_means.tolist()

    # Calculate mean for each state
    state_0_means = []
    state_1_means = []
    for i in range(len(weighted_means)):
        if motion[i] == 0: state_0_means.append(weighted_means[i])
        else: state_1_means.append(weighted_means[i])
    mean_state_0 = np.mean(state_0_means) if len(state_0_means) > 0 else 0
    mean_state_1 = np.mean(state_1_means) if len(state_1_means) > 0 else 0

    return weighted_means, mean_state_0, mean_state_1

# Main workflow
if __name__ == "__main__":

    noisy_data = []
    clean_data = []
    noise_data = []

    for i in range(100):
        # Step 1: Generate artificial training data
        noisy_signal, clean_signal, noise = generate_am_signal(DURATION, SAMPLE_RATE, FREQ, NOISE_COEFF)
        
        # Prepare data for training
        f, _, noisy_stft = apply_stft(noisy_signal)
        _, _, clean_stft = apply_stft(clean_signal)
        _, _, noise_stft = apply_stft(noise)

        noisy_data.append([np.abs(noisy_stft[:, i:i+64]) for i in range(0, noisy_stft.shape[1] - 63, 64)])
        clean_data.append([np.abs(clean_stft[:, i:i+64]) for i in range(0, clean_stft.shape[1] - 63, 64)])
        noise_data.append([np.abs(noise_stft[:, i:i+64]) for i in range(0, noise_stft.shape[1] - 63, 64)])

    noisy_data = np.vstack(noisy_data)
    clean_data = np.vstack(clean_data)
    noise_data = np.vstack(noise_data)

    window_size = 1
    plot_scalograms(f, np.arange(noisy_data.shape[0] * noisy_data.shape[2]), flatten_matrix(noisy_data[window_size:window_size+1, :, :]), flatten_matrix(clean_data[window_size:window_size+1, :, :]), flatten_matrix(noise_data[window_size:window_size+1, :, :]))

    noisy_data = noisy_data.reshape(-1, 64, 64, 1)  # Add channel dimension for Conv2D
    clean_data = clean_data.reshape(-1, 64, 64, 1)  # Add channel dimension for Conv2D

    # Train or load the autoencoder
    autoencoder = train_or_load_autoencoder(noisy_data, clean_data, MODEL_FILE)

    _, _, original_data, denoised_data = process_real_data(noisy_data, autoencoder)
    plot_scalograms(f, np.arange(noisy_data.shape[0] * noisy_data.shape[2]), denoised_data[:, :window_size*64], denoised_data[:, :window_size*64], flatten_matrix(noise_data[:window_size, :, :]))

    path_ctrl = [("/path/to/data", "/path/to/motion_levels")]

    # Step 2: Process real data
    # Assuming `real_data` is a numpy array
    orig_inactive_means = []
    orig_active_means = []
    denoised_inactive_means = []
    denoised_active_means = []

    path, video = path_ctrl[0]
    d, t = loadData(path)
    d = [convertVolt(_, 430.0) for _ in d]
    real_data = np.array(d) * 100  # Replace with actual real data
    f, t, original_data, denoised_data = process_real_data(real_data, autoencoder)

    time, motion = motion_levels(video)

    plot_06(f, t, original_data, denoised_data)

    motion_a = [0.5 if _ == 1 else np.nan for _ in motion]
    motion_i = [0.5 if _ == 0 else np.nan for _ in motion]
    fig, ax = plt.subplots(figsize=(120 / MM_TO_INCH, 45 / MM_TO_INCH))
    ax.plot(np.array(time), motion_a, label="Motion level", color=COLORS['red'], linewidth=TICK_WIDTH)

    # Save and show figure
    plt.tight_layout()
    plt.savefig('figure_scalogram_motion.tiff', format='tiff', dpi=300)

    plt.show()