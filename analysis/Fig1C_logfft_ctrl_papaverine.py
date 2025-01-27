import os
import struct
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

np.seterr(divide = 'ignore') 

path_papa = "/path/to/data/papaverine"
path_ctrl = "/path/to/data/control"

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


def main():
    d, t = loadData(path_papa)
    d = [convertVolt(_, 430.0) for _ in d]
    d = d[0 * 60 * 100: 60 * 60 * 100]
    t = [_ / 1000000.0 for _ in t]
    d_fft = fft(d)
    d_freq = fftfreq(len(d), 1/100)[:len(d)//2]

    log_freq = np.log10(d_freq[1:])  # Exclude the DC component (0 Hz)
    log_spectrum = np.log10(np.abs(2.0/len(d) * d_fft[1:len(d)//2]))  # Exclude the DC component

    num_samples = 1024
    uniform_log_freq = np.linspace(log_freq.min(), log_freq.max(), num_samples)
    uniform_log_spectrum = np.interp(uniform_log_freq, log_freq, log_spectrum)

    max_freq = 5
    valid_indices = uniform_log_freq[:] <= np.log10(max_freq)
    restricted_log_freq = uniform_log_freq[valid_indices]
    restricted_log_spectrum = uniform_log_spectrum[valid_indices]
    baseline_fit = np.poly1d(np.polyfit(restricted_log_freq, restricted_log_spectrum, deg=1))

    print("alpha: ", baseline_fit[1])

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


    fig, ax = plt.subplots(figsize=(60 / MM_TO_INCH, 45 / MM_TO_INCH))
    ax.plot(uniform_log_freq, uniform_log_spectrum, color=COLORS['gray'], linewidth=TRACE_WIDTH)

    # Customize axis appearance
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(AXIS_LINEWIDTH)
    ax.spines['left'].set_linewidth(AXIS_LINEWIDTH)

    # Customize ticks
    ax.tick_params(axis='both', which='both', length=TICK_LENGTH, width=TICK_WIDTH, labelsize=TICK_LABEL_SIZE)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-7, -5, -3, -1])
    ax.set_xlim([-3.8, 2.2])
    ax.set_ylim([-6, -0.8])
    ax.set_xlabel('log(Frequency)', fontsize=AXIS_TITLE_SIZE, fontname=FONT_FAMILY)
    ax.set_ylabel('log(Amplitude)', fontsize=AXIS_TITLE_SIZE, fontname=FONT_FAMILY)

    # Customize legend
    ax.legend(fontsize=INSET_LABEL_SIZE, frameon=False)

    # Set title
    ax.set_title('Papaverine', fontsize=AXIS_TITLE_SIZE, fontname=FONT_FAMILY)

    plt.tight_layout()
    plt.savefig('figure_output_papa.tiff', format='tiff', dpi=300)


    d, t = loadData(path_ctrl)
    d = [convertVolt(_, 430.0) for _ in d]
    d = d[0: 90 * 60 * 100]
    t = [_ / 1000000.0 for _ in t]
    d_fft = fft(d)
    d_freq = fftfreq(len(d), 1/100)[:len(d)//2]

    log_freq = np.log10(d_freq[1:])  # Exclude the DC component (0 Hz)
    log_spectrum = np.log10(np.abs(2.0/len(d) * d_fft[1:len(d)//2]))  # Exclude the DC component

    uniform_log_freq = np.linspace(log_freq.min(), log_freq.max(), num_samples)
    uniform_log_spectrum = np.interp(uniform_log_freq, log_freq, log_spectrum)

    max_freq = 5
    valid_indices = uniform_log_freq[:] <= np.log10(max_freq)
    restricted_log_freq = uniform_log_freq[valid_indices]
    restricted_log_spectrum = uniform_log_spectrum[valid_indices]
    baseline_fit = np.poly1d(np.polyfit(restricted_log_freq, restricted_log_spectrum, deg=1))
    print("alpha: ", baseline_fit[1])

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
        'black': (0, 0, 0),
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


    fig, ax = plt.subplots(figsize=(60 / MM_TO_INCH, 45 / MM_TO_INCH))
    ax.plot(uniform_log_freq, uniform_log_spectrum, color=COLORS['black'], linewidth=TRACE_WIDTH)

    # Customize axis appearance
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(AXIS_LINEWIDTH)
    ax.spines['left'].set_linewidth(AXIS_LINEWIDTH)

    # Customize ticks
    ax.tick_params(axis='both', which='both', length=TICK_LENGTH, width=TICK_WIDTH, labelsize=TICK_LABEL_SIZE)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-7, -5, -3, -1])
    ax.set_xlim([-3.8, 2.2])
    ax.set_ylim([-6, -0.8])
    ax.set_xlabel('log(Frequency)', fontsize=AXIS_TITLE_SIZE, fontname=FONT_FAMILY)
    ax.set_ylabel('log(Amplitude)', fontsize=AXIS_TITLE_SIZE, fontname=FONT_FAMILY)

    # Customize legend
    ax.legend(fontsize=INSET_LABEL_SIZE, frameon=False)

    # Set title
    ax.set_title('Control', fontsize=AXIS_TITLE_SIZE, fontname=FONT_FAMILY)

    plt.tight_layout()
    plt.savefig('figure_output_ctrl.tiff', format='tiff', dpi=300)

    plt.show()


if __name__ == '__main__':
    main()
