import os
import struct
import matplotlib.pyplot as plt

path = "/path/to/datafile"

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

def main():
    d, t = loadData(path)
    d = [convertVolt(_, 430.0) for _ in d]
    t = [_ / 1000000.0 for _ in t]
    fig, ax = plt.subplots()
    ax.plot(t, d, color='C0', linestyle='-')
    ax.set_xlabel('time [sec]')
    ax.set_ylabel('Voltage [mV]')
    plt.show()

if __name__ == '__main__':
    main()