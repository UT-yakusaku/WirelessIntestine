import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

# Function to compute the absolute difference between two frames
def motion_level(prev_frame, curr_frame, threshold=25):
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute the absolute difference between frames
    diff = cv2.absdiff(prev_gray, curr_gray)
    
    # Sum the differences (this will give us a 'motion level')
    motion_value = np.sum(diff)  # Total difference
    return motion_value

# Function to process the video for motion detection
def process_video(video_path, max_duration=30, frame_skip=5, threshold=25, csv_file='motion_levels.csv'):
    # Check if the CSV file already exists to load data
    if os.path.exists(csv_file):
        print(f"Loading motion levels from {csv_file}...")
        motion_levels = np.loadtxt(csv_file, delimiter=',')
        return motion_levels

    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Limit the duration based on the provided max_duration (in seconds)
    max_frames = min(int(max_duration), frame_count)
    
    motion_levels = []  # To store motion frames
    
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    # Progress bar using tqdm
    with tqdm(total=max_frames // frame_skip, desc="Processing Video", unit="frame") as pbar:
        for i in range(1, max_frames, frame_skip):
            # Skip frames
            for _ in range(frame_skip - 1):
                ret, _ = cap.read()
                if not ret:
                    break
            
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            motion_value = motion_level(prev_frame, curr_frame)
            motion_levels.append(motion_value)
            
            # Update the previous frame for the next iteration
            prev_frame = curr_frame
            
            # Update progress bar
            pbar.update(1)

    cap.release()
    print(f"Saving motion levels to {csv_file}...")
    np.savetxt(csv_file, motion_levels, delimiter=',')
    
    # Plot the motion over time
    return motion_levels


import os
import struct
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.stats import ttest_rel
from scipy.fft import fft, fftfreq


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


def main():

    nbins = 800

    # Define formatting constants (in points and mm)
    MM_TO_PT = 2.83465
    MM_TO_INCH = 25.4
    AXIS_LINEWIDTH = 0.2 * MM_TO_PT
    ERRORBAR_LINEWIDTH = 0.2 * MM_TO_PT
    TICK_LENGTH = 0.8 * MM_TO_PT
    TICK_WIDTH = 0.2 * MM_TO_PT
    SCALEBAR_WIDTH = 0.3 * MM_TO_PT
    RASTER_WIDTH = 0.2 * MM_TO_PT
    RASTER_LENGTH = 0.8 * MM_TO_PT
    TRACE_WIDTH = 0.12 * MM_TO_PT
    DOT_SIZE = 0.8 * MM_TO_PT

    # Font sizes
    AXIS_TITLE_SIZE = 8
    TICK_LABEL_SIZE = 7
    INSET_LABEL_SIZE = 6
    FONT_FAMILY = 'Arial'

    # Define custom colors
    COLORS = {
        'blue': (0/255, 128/255, 192/255),
        'red': (255/255, 70/255, 50/255),
        'pink': (255/255, 150/255, 200/255),
        'green': (20/255, 180/255, 20/255),
        'yellow': (230/255, 160/255, 20/255),
        'gray': (128/255, 128/255, 128/255),
        'purple': (200/255, 50/255, 255/255),
        'cyan': (20/255, 200/255, 200/255),
        'brown': (128/255, 0/255, 0/255),
        'navy': (0/255, 0/255, 100/255),
        'tan': (228/255, 94/255, 50/255)
    }

    alpha_active = []
    alpha_inactive = []


    paths = [
                  ("/path/to/data", "/path/to/motion_levels", "recorded time", "start(min)", "end(min)")
                ]

    for path, video, dur, start, end in paths:

        vid = process_video(None, max_duration=60 * 30 * 60 * 8, frame_skip=180, threshold=25, csv_file=video)   # Analyze the first 30 seconds, skipping 5 frames each time
        vid = np.array(vid) / 1000000

        vid_ = []
        for i in range(len(vid)//5):
            vid_.append(np.sum(vid[i * 5: i * 5 + 5]) / 5)
        vid = np.array(vid_)

        vid_lim = vid[start * 10 // 5: end * 10 // 5]

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
        time = np.arange(len(vid)) * 30 # Create a time array
        motion = [1 if _ > threshold else 0 for _ in vid]

        for i in range(len(time)):
            if time[i] / 60 < start or time[i] / 60 > end:
                motion[i] = 2

        time = time[:end * 2 + 1]
        motion = motion[:end * 2 + 1]

        d, t = loadData(path)
        d = [convertVolt(_, 430.0) for _ in d]
        t = [_ / 1000000.0 for _ in t]

        sps = len(d) / dur

        d_active = []
        d_inactive = []

        for t in range(min(len(d), int(len(motion) * 30 * sps))):
            if motion[int(t / sps / 30)] == 1:
                d_active.append(d[t])
            elif motion[int(t / sps / 30)] == 0:
                d_inactive.append(d[t])

        d_active_fft = fft(d_active)
        d_active_freq = fftfreq(len(d_active), 1/sps)[:len(d_active)//2]

        log_freq = np.log10(d_active_freq[1:])  # Exclude the DC component (0 Hz)
        log_spectrum = np.log10(np.abs(2.0/len(d_active) * d_active_fft[1:len(d_active)//2]))  # Exclude the DC component

        num_samples = 1024
        uniform_log_freq = np.linspace(log_freq.min(), log_freq.max(), num_samples)
        uniform_log_spectrum = np.interp(uniform_log_freq, log_freq, log_spectrum)

        max_freq = 5
        valid_indices = uniform_log_freq[:] <= np.log10(max_freq)
        restricted_log_freq = uniform_log_freq[valid_indices]
        restricted_log_spectrum = uniform_log_spectrum[valid_indices]
        baseline_fit = np.poly1d(np.polyfit(restricted_log_freq, restricted_log_spectrum, deg=1))
        print("alpha: ", baseline_fit[1])
        alpha_active.append(baseline_fit[1] * -1)


        d_inactive_fft = fft(d_inactive)
        d_inactive_freq = fftfreq(len(d_inactive), 1/sps)[:len(d_inactive)//2]

        log_freq = np.log10(d_inactive_freq[1:])  # Exclude the DC component (0 Hz)
        log_spectrum = np.log10(np.abs(2.0/len(d_inactive) * d_inactive_fft[1:len(d_inactive)//2]))  # Exclude the DC component

        num_samples = 1024
        uniform_log_freq = np.linspace(log_freq.min(), log_freq.max(), num_samples)
        uniform_log_spectrum = np.interp(uniform_log_freq, log_freq, log_spectrum)

        max_freq = 5
        valid_indices = uniform_log_freq[:] <= np.log10(max_freq)
        restricted_log_freq = uniform_log_freq[valid_indices]
        restricted_log_spectrum = uniform_log_spectrum[valid_indices]
        baseline_fit = np.poly1d(np.polyfit(restricted_log_freq, restricted_log_spectrum, deg=1))
        print("alpha: ", baseline_fit[1])
        alpha_inactive.append(baseline_fit[1] * -1)

        # Print results
        print("Threshold for separating Gaussian core and outliers:", threshold)

    group1 = alpha_inactive
    group2 = alpha_active

    print(f"mean inactive: {np.mean(alpha_inactive)}, mean active: {np.mean(alpha_active)}")

    # Perform paired t-test
    t_stat, p_value = ttest_rel(group1, group2)
    print(f"t-statistic: {t_stat}, p-value: {p_value}")

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
    ax.set_yticks([0.7, 0.6, 0.5])
    ax.set_ylim([0.45, 0.75])
    ax.set_xticklabels(['Immobile', 'Behaving'],  rotation = 30, fontsize=AXIS_TITLE_SIZE, fontname=FONT_FAMILY)

    # Set title
    ax.set_title('Alpha', fontsize=AXIS_TITLE_SIZE, fontname=FONT_FAMILY)

    # Save and show figure
    plt.tight_layout()
    plt.savefig('paired_t_alpha_boxplot.tiff', format='tiff', dpi=300)
    plt.show()
    

if __name__ == '__main__':
    main()
