import os
from scipy.stats import ttest_rel
from scipy.signal import find_peaks
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
import scipy.stats as stats

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
FREQ = 0.6  # Target frequency in Hz
NOISE_COEFF = (0.8, 1.2)
CHUNK_SIZE = 1024
FFT_SIZE = 1024
MODEL_FILE = "autoencoder_params.keras"

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
        mod_freq = np.random.uniform(0.01, 0.02)
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
        am_signal.append(am_signal_ * np.random.uniform(0.0, 1.2))

    t = np.hstack(t)
    am_signal = np.hstack(am_signal)
    # Add pink noise
    coeff = np.random.uniform(1, 3)
    pink_noise_data = pink_noise(len(t)) / 100
    noisy_signal = am_signal / coeff + 1.5 * coeff * pink_noise_data
    am_signal = am_signal / coeff + 0.1 * coeff * pink_noise_data
    return noisy_signal * 100, am_signal * 100


# Function to generate pink noise
def pink_noise(N):
    uneven = N % 2
    X = np.random.randn(N // 2 + 1 + uneven) + 1j * np.random.randn(N // 2 + 1 + uneven)
    S = np.sqrt(1.0 / (np.arange(len(X)) + 1.0))  # 1/f spectrum
    y = (np.fft.irfft(X * S)).real
    if uneven:
        y = y[:-1]
    return y / np.std(y)

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
def apply_stft(data, n_fft=FFT_SIZE, freq_range=(0.3, 2.0), sample_rate = SAMPLE_RATE):
    f, t, Zxx = stft(data, fs=sample_rate, nperseg=n_fft)
    
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
def process_real_data(data, autoencoder, sample_rate):
    denoised_chunks = []
    f = []
    t_chunks = []
    f, t, Zxx = apply_stft(data, sample_rate=sample_rate)

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

def sum06(f, t, Zxx_orig, Zxx_denoised):
    freq_min_idx = np.searchsorted(f, 0.4)
    freq_max_idx = np.searchsorted(f, 0.8)
    Zxx_orig_sum = np.sum(Zxx_orig[freq_min_idx:freq_max_idx], axis=0)
    Zxx_denoised_sum = np.sum(Zxx_denoised[freq_min_idx:freq_max_idx], axis=0)[10:]
    return Zxx_orig_sum, Zxx_denoised_sum

def hist06(f, t, Zxx_orig, Zxx_denoised):
    freq_min_idx = np.searchsorted(f, 0.4)
    freq_max_idx = np.searchsorted(f, 0.8)
    Zxx_orig_sum = np.sum(Zxx_orig[freq_min_idx:freq_max_idx], axis=0)
    Zxx_denoised_sum = np.sum(Zxx_denoised[freq_min_idx:freq_max_idx], axis=0)[10:]

    # Parameters
    sampling_rate = 12  # 12 samples per minute
    window_size = 5  # in seconds
    window_samples = int(window_size / (60 / sampling_rate))  # Convert to samples
    threshold = np.percentile(Zxx_denoised_sum, 5) + 2 * np.std(Zxx_denoised_sum)  # Example threshold to separate events
    print(threshold)

    # Smooth the data (moving average)
    smoothed_data = Zxx_denoised_sum

    # Detect events
    event_indices, _ = find_peaks(smoothed_data, height=threshold, distance=sampling_rate)
    event_times = event_indices / sampling_rate  # Convert indices to minutes
    intervals = np.diff(event_times)  # Calculate intervals between events

    return intervals

# Visualization
def plot_scalograms(f, t, Zxx_orig, Zxx_denoised):
    fig, axs = plt.subplots(2, 1, figsize=(100 / MM_TO_INCH, 80 / MM_TO_INCH))

    cs0 = axs[0].pcolormesh(t / 60, f, np.abs(Zxx_orig), shading='gouraud', cmap='viridis', norm=LogNorm(vmin=3*1e-3, vmax=3*1e-1))
    axs[0].set_title("Original Data", fontsize=AXIS_TITLE_SIZE, family=FONT_FAMILY)

    t = t[:Zxx_denoised.shape[1]]
    cs1 = axs[1].pcolormesh(t / 60, f, np.abs(Zxx_denoised.squeeze()), shading='gouraud', cmap='viridis', norm=LogNorm(vmin=3*1e-3, vmax=3*1e-1))
    axs[1].set_title("Denoised Data", fontsize=AXIS_TITLE_SIZE, family=FONT_FAMILY)

    for ax in axs:
        ax.set_ylabel("Frequency [Hz]", fontsize=TICK_LABEL_SIZE, family=FONT_FAMILY)
        ax.set_xlabel("Time [min]", fontsize=TICK_LABEL_SIZE, family=FONT_FAMILY)
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE, width=TICK_WIDTH, length=TICK_LENGTH)
        ax.set_ylim([0.3, 1.8])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    cbar = fig.colorbar(cs0, shrink = 0.95)
    cbar.ax.set_frame_on(False)
    #[i.set_linewidth(0) for i in cbar.ax.spines.itervalues()]
    cbar = fig.colorbar(cs1, shrink = 0.95)
    cbar.ax.set_frame_on(False)
    plt.tight_layout()
    plt.savefig('figure_output_scalogram.tiff', format='tiff', dpi=300)
    plt.show()


def motion_levels(video, start = 0, end = -1):

    # start will be indicated in minutes

    vid = np.loadtxt(video, delimiter=',')   # Analyze the first 30 seconds, skipping 5 frames each time
    nbins = 800
    vid = np.array(vid) / 1000000

    duration = 5

    vid_ = []
    for i in range(len(vid)//duration):
        vid_.append(np.sum(vid[i * duration: i * duration + duration]) / duration)
    vid = np.array(vid_)

    vid_lim = vid[start * 10 // duration: end * 10 // duration]

    hist, bin_edges = np.histogram(vid_lim, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit a Gaussian Mixture Model (GMM) with 2 components
    gmm = GaussianMixture(n_components=2, random_state=42)
    data_reshaped = vid_lim.reshape(-1, 1)
    gmm.fit(data_reshaped)

    # Predict cluster membership
    labels = gmm.predict(data_reshaped)
    means = gmm.means_.flatten()

    # Determine which cluster corresponds to the Gaussian core
    core_cluster = np.argmin(means)  # Cluster with the smaller mean
    outlier_cluster = 1 - core_cluster

    # Find the threshold (minimum value in the outlier cluster)
    threshold = vid_lim[labels == outlier_cluster].min()

    print("The percentage of values over threshold:", len(vid_lim[vid_lim > threshold]) / len(vid_lim) * 100)

    # Plot the time series with shading for values over the threshold
    time = np.arange(len(vid)) * 6 * duration  # Create a time array
    motion = [1 if _ > threshold else 0 for _ in vid]

    for i in range(len(time)):
        if time[i] / 60 < start or time[i] / 60 > end:
            motion[i] = 2

    time = time[:end * 2 + 1]
    motion = motion[:end * 2 + 1]

    return time, motion

def calculate_weighted_means(orig, motion, orig_interval=5.12, motion_interval=30):
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
        elif motion[i] == 1: state_1_means.append(weighted_means[i])
    mean_state_0 = np.mean(state_0_means) if len(state_0_means) > 0 else 0
    mean_state_1 = np.mean(state_1_means) if len(state_1_means) > 0 else 0

    return state_0_means, state_1_means


# Main workflow
if __name__ == "__main__":

    noisy_data = []
    clean_data = []

    for i in range(1):
        # Step 1: Generate artificial training data
        noisy_signal, clean_signal = generate_am_signal(DURATION, SAMPLE_RATE, FREQ, NOISE_COEFF)
        
        # Prepare data for training
        f, _, noisy_stft = apply_stft(noisy_signal)
        _, _, clean_stft = apply_stft(clean_signal)

        noisy_data.append([np.abs(noisy_stft[:, i:i+64]) for i in range(0, noisy_stft.shape[1] - 63, 64)])
        clean_data.append([np.abs(clean_stft[:, i:i+64]) for i in range(0, clean_stft.shape[1] - 63, 64)])

    noisy_data = np.vstack(noisy_data)
    clean_data = np.vstack(clean_data)

    noisy_data = noisy_data.reshape(-1, 64, 64, 1)  # Add channel dimension for Conv2D
    clean_data = clean_data.reshape(-1, 64, 64, 1)  # Add channel dimension for Conv2D

    # Train or load the autoencoder
    autoencoder = train_or_load_autoencoder(noisy_data, clean_data, MODEL_FILE)


    path_ctrl = [
                  ("/path/to/data", "/path/to/motion_levels", "recorded time", "start(min)", "end(min)")
                ]
    

    # Step 2: Process real data
    # Assuming `real_data` is a numpy array
    orig_inactive_percent = []
    orig_active_percent = []
    denoised_inactive_percent = []
    denoised_active_percent = []

    for path, video, dur, start, end in path_ctrl:
        d, t = loadData(path)
        d = [convertVolt(_, 430.0) for _ in d]

        real_data = np.array(d) * 100  # Replace with actual real data
        f, t, original_data, denoised_data = process_real_data(real_data, autoencoder, len(d) / dur)

        time, motion = motion_levels(video, start, end)
        orig, denoised = sum06(f, t, original_data, denoised_data)

        orig_inactive_means, orig_active_means = calculate_weighted_means(orig, motion)
        denoised_inactive_means, denoised_active_means = calculate_weighted_means(denoised, motion)

        orig_inactive_means = np.array(orig_inactive_means)
        orig_active_means = np.array(orig_active_means)
        denoised_inactive_means = np.array(denoised_inactive_means)
        denoised_active_means = np.array(denoised_active_means)

        # calc percent of orig

        means_array = np.array(orig_inactive_means)

        nbins = 400
        # Fit a Gaussian Mixture Model (GMM) with 2 components
        gmm = GaussianMixture(n_components=2, random_state=42)
        data_reshaped = means_array.reshape(-1, 1)
        gmm.fit(data_reshaped)

        # Predict cluster membership
        labels = gmm.predict(data_reshaped)
        means = gmm.means_.flatten()

        # Determine which cluster corresponds to the Gaussian core
        core_cluster = np.argmin(means)  # Cluster with the smaller mean
        outlier_cluster = 1 - core_cluster

        # Find the threshold (minimum value in the outlier cluster)
        threshold = means_array[labels == outlier_cluster].min()

        print("The percentage of values over threshold:", len(means_array[means_array > threshold]) / len(means_array) * 100)
        print("threshold:", threshold)

        orig_active_percent.append(len(orig_active_means[orig_active_means > threshold]) / len(orig_active_means) * 100)
        orig_inactive_percent.append(len(orig_inactive_means[orig_inactive_means > threshold]) / len(orig_inactive_means) * 100)

        # calc percent of denoised

        means_array = np.array(denoised_active_means)

        nbins = 80
        # Fit a Gaussian Mixture Model (GMM) with 2 components
        gmm = GaussianMixture(n_components=2, random_state=42)
        data_reshaped = means_array.reshape(-1, 1)
        gmm.fit(data_reshaped)

        # Predict cluster membership
        labels = gmm.predict(data_reshaped)
        means = gmm.means_.flatten()

        # Determine which cluster corresponds to the Gaussian core
        core_cluster = np.argmin(means)  # Cluster with the smaller mean
        outlier_cluster = 1 - core_cluster

        # Find the threshold (minimum value in the outlier cluster)
        threshold = means_array[labels == outlier_cluster].min()

        print("The percentage of values over threshold:", len(means_array[means_array > threshold]) / len(means_array) * 100)
        print("threshold:", threshold)

        denoised_active_percent.append(len(denoised_active_means[denoised_active_means > threshold]) / len(denoised_active_means) * 100)
        denoised_inactive_percent.append(len(denoised_inactive_means[denoised_inactive_means > threshold]) / len(denoised_inactive_means) * 100)

    # Original version

    group1 = orig_inactive_percent
    group2 = orig_active_percent

    print(f"mean inactive: {np.mean(orig_inactive_percent)}, mean active: {np.mean(orig_active_percent)}")

    # Perform paired t-test
    t_stat, p_value = ttest_rel(group1, group2)
    print(f"t-statistic: {t_stat}, p-value: {p_value}")


    # Create a figure
    fig, ax = plt.subplots(figsize=(30 / MM_TO_INCH, 45 / MM_TO_INCH))

    # Boxplot with group-specific colors
    boxplot = ax.boxplot([group1, group2], widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor='none', edgecolor=COLORS['blue'], linewidth=TRACE_WIDTH),
                        whiskerprops=dict(color=COLORS['blue'], linewidth=TRACE_WIDTH),
                        capprops=dict(color=COLORS['blue'], linewidth=TRACE_WIDTH),
                        medianprops=dict(color=COLORS['blue'], linewidth=TRACE_WIDTH))

    # Update second boxplot color to red
    for box, median in zip(boxplot['boxes'][1:], boxplot['medians'][1:]):
        box.set_edgecolor(COLORS['red'])
        median.set_color(COLORS['red'])

    for whisker, cap in zip(boxplot['whiskers'][2:4], boxplot['caps'][2:4]):
        whisker.set_color(COLORS['red'])
        whisker.set_linewidth(TRACE_WIDTH)
        cap.set_color(COLORS['red'])
        cap.set_linewidth(TRACE_WIDTH)

    # Plot individual data points and connect paired data
    x_positions = [1, 2]
    for i in range(len(group1)):
        ax.plot(x_positions, [group1[i], group2[i]], color='gray', linewidth=TRACE_WIDTH, alpha=0.5)
        ax.scatter(1, group1[i], color=COLORS['blue'], s=DOT_SIZE**2, zorder=3)
        ax.scatter(2, group2[i], color=COLORS['red'], s=DOT_SIZE**2, zorder=3)


    # Customize axis appearance
    ax.spines['top'].set_linewidth(AXIS_LINEWIDTH)
    ax.spines['right'].set_linewidth(AXIS_LINEWIDTH)
    ax.spines['bottom'].set_linewidth(AXIS_LINEWIDTH)
    ax.spines['left'].set_linewidth(AXIS_LINEWIDTH)


    # Customize ticks
    ax.tick_params(axis='both', which='both', length=TICK_LENGTH, width=TICK_WIDTH, labelsize=TICK_LABEL_SIZE)
    ax.set_xticks(x_positions)
    ax.set_yticks([10, 30, 50])
    ax.set_ylim([5, 58])
    ax.set_xticklabels(['Immobile', 'Behaving'],  rotation = 30, fontsize=AXIS_TITLE_SIZE, fontname=FONT_FAMILY)

    # Set title
    ax.set_title('Original', fontsize=AXIS_TITLE_SIZE, fontname=FONT_FAMILY)

    # Save and show figure
    plt.tight_layout()
    plt.savefig('paired_t_orig_boxplot.tiff', format='tiff', dpi=300)


    # Denoised version

    group1 = denoised_inactive_percent
    group2 = denoised_active_percent

    print(f"mean inactive: {np.mean(denoised_inactive_percent)}, mean active: {np.mean(denoised_active_percent)}")

    # Perform paired t-test
    t_stat, p_value = ttest_rel(group1, group2)
    print(f"t-statistic: {t_stat}, p-value: {p_value}")


    # Create a figure
    fig, ax = plt.subplots(figsize=(30 / MM_TO_INCH, 45 / MM_TO_INCH))

    # Boxplot with group-specific colors
    boxplot = ax.boxplot([group1, group2], widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor='none', edgecolor=COLORS['blue'], linewidth=TRACE_WIDTH),
                        whiskerprops=dict(color=COLORS['blue'], linewidth=TRACE_WIDTH),
                        capprops=dict(color=COLORS['blue'], linewidth=TRACE_WIDTH),
                        medianprops=dict(color=COLORS['blue'], linewidth=TRACE_WIDTH))

    # Update second boxplot color to red
    for box, median in zip(boxplot['boxes'][1:], boxplot['medians'][1:]):
        box.set_edgecolor(COLORS['red'])
        median.set_color(COLORS['red'])

    for whisker, cap in zip(boxplot['whiskers'][2:4], boxplot['caps'][2:4]):
        whisker.set_color(COLORS['red'])
        whisker.set_linewidth(TRACE_WIDTH)
        cap.set_color(COLORS['red'])
        cap.set_linewidth(TRACE_WIDTH)

    # Plot individual data points and connect paired data
    x_positions = [1, 2]
    for i in range(len(group1)):
        ax.plot(x_positions, [group1[i], group2[i]], color='gray', linewidth=TRACE_WIDTH, alpha=0.5)
        ax.scatter(1, group1[i], color=COLORS['blue'], s=DOT_SIZE**2, zorder=3)
        ax.scatter(2, group2[i], color=COLORS['red'], s=DOT_SIZE**2, zorder=3)


    # Customize axis appearance
    ax.spines['top'].set_linewidth(AXIS_LINEWIDTH)
    ax.spines['right'].set_linewidth(AXIS_LINEWIDTH)
    ax.spines['bottom'].set_linewidth(AXIS_LINEWIDTH)
    ax.spines['left'].set_linewidth(AXIS_LINEWIDTH)


    # Customize ticks
    ax.tick_params(axis='both', which='both', length=TICK_LENGTH, width=TICK_WIDTH, labelsize=TICK_LABEL_SIZE)
    ax.set_xticks(x_positions)
    ax.set_yticks([10, 30, 50])
    ax.set_ylim([5, 58])
    ax.set_xticklabels(['Immobile', 'Behaving'],  rotation = 30, fontsize=AXIS_TITLE_SIZE, fontname=FONT_FAMILY)

    # Set title
    ax.set_title('Denoised', fontsize=AXIS_TITLE_SIZE, fontname=FONT_FAMILY)

    # Save and show figure
    plt.tight_layout()
    plt.savefig('paired_t_denoised_boxplot.tiff', format='tiff', dpi=300)
    plt.show()