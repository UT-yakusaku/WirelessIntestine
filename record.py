import os
import threading
import serial
from serial.tools import list_ports
import asyncio
from bleak import BleakClient
from bleak.exc import BleakDeviceNotFoundError
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import time
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from collections import deque
from queue import Queue
import struct
import datetime
from dotenv import load_dotenv

load_dotenv() 

frame = None
time_scale = 5.0
packet_size = 9

adc_multiplier = 0.125
amp_resistance = 390.0
amp_gain = 1.0 + (100000.0 / amp_resistance)
volt_coeff = adc_multiplier / amp_gain

camera_width = 1080
camera_height = 1920

lock = threading.Lock()

ports = list_ports.comports()
devices = [info.device for info in ports]

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title('matplotlib graph')
        
        self.sps = 100
        
        self.th_camera = None
        self.cameraID = -1
        self.deviceID = ""
        self.video_display = False
        self.graph_display = False
        self.graph_interval = 10
        self.graph_interval_cnt = 0
        self.cap = None
        self.com = None

        self.graph_x = deque([])
        self.graph_y = deque([])
        self.time_loop_cnt = 0
        self.time_start_rec = 0
        self.graph_display_clfc = 0

        self.save_dir = ""
        self.sub_dir = "datafile"
        self.is_recording = False
        self.device_data_f = None
        self.video_data_f = None
        self.record_time_diff = None
        self.record_time_video = None
        self.record_n_row = 0

        self.ble_serial_start = False
        self.data_prev = bytearray()
        self.BLE_ATTRIBUTE_MAX_VALUE_LENGTH = 20
        self.data_idx = self.BLE_ATTRIBUTE_MAX_VALUE_LENGTH

        frame1 = ttk.Frame(self.master, padding=10)
        frame1.grid(row=2, column=0, sticky=tk.E)

        IDirLabel = ttk.Label(frame1, text="Save to folder", padding=(5, 2))
        IDirLabel.pack(side=tk.LEFT)

        self.entry1 = tk.StringVar(self.master)
        self.IDirEntry = ttk.Entry(frame1, textvariable=self.entry1, width=60)
        self.IDirEntry.pack(side=tk.LEFT)

        ISubDirLabel = ttk.Label(frame1, text="/"+self.sub_dir, padding=(0, 2))
        ISubDirLabel.pack(side=tk.LEFT)

        IDirButton = ttk.Button(frame1, text="Browse...", command=self.dirdialog_clicked)
        IDirButton.pack(side=tk.LEFT)

        frame2 = ttk.Frame(self.master, padding=10)
        frame2.grid(row=1, column=0, sticky=tk.E)

        combo_string = tk.StringVar()
        self.combobox_dev = ttk.Combobox(frame2, textvariable=combo_string, values=devices)
        self.combobox_dev.pack(side=tk.LEFT)

        self.DevGraphButton = ttk.Button(frame2, text="Show Graph", command=self.showgraph_clicked)
        self.DevGraphButton.pack(side=tk.LEFT)

        camera_devices = self.CameraIndexes()
        combo_string_camera = tk.StringVar()
        self.combobox_camera = ttk.Combobox(frame2, textvariable=combo_string_camera, values=["Camera " + str(i) for i in camera_devices])
        self.combobox_camera.pack(side=tk.LEFT)

        self.CameraShowButton = ttk.Button(frame2, text="Show Preview", command=self.cameradisplay_clicked)
        self.CameraShowButton.pack(side=tk.LEFT)

        frame3 = ttk.Frame(self.master, padding=10)
        frame3.grid(row=4, column=1, sticky=tk.E)
        self.canvas1 = tk.Canvas(frame3, width=640, height=480)
        self.canvas1.pack()
        self.video_frame_timer()

        frame4 = ttk.Frame(self.master, padding=10)
        frame4.grid(row=4, column=0, sticky=tk.E)
        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.fig_canvas = FigureCanvasTkAgg(self.fig, frame4)
        self.toolbar = NavigationToolbar2Tk(self.fig_canvas, frame4)
        self.fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.anim = animation.FuncAnimation(self.fig, self.next_graph, interval=100, blit=False)

        frame5 = ttk.Frame(self.master, padding=10)
        frame5.grid(row=3, column=0, sticky=tk.E)
        self.RecordButton = ttk.Button(frame5, text="Record", command=self.record_clicked)
        self.RecordButton.pack(side=tk.LEFT)

        self.master.protocol("WM_DELETE_WINDOW", self.click_close)

        self.mac_address = os.getenv('MAC_ADDRESS')

        self.halt_thread = False
        self.th_camera = threading.Thread(target=self.CameraCapture)
        self.th_camera.start()
        self.th_serial = threading.Thread(target=self.run_ble_loop)
        self.bleak_thread_ready = threading.Event()
        self.th_serial.start()
        self.bleak_thread_ready.wait()

        self.connect_bleak(10)
        

    def dirdialog_clicked(self):
        iDir = os.path.abspath(os.path.dirname(__file__))
        iDirPath = filedialog.askdirectory(initialdir = iDir)
        self.entry1.set(iDirPath)

    def showgraph_clicked(self):
        if self.graph_display is False:
            #if self.changedevice():
            self.graph_display = True
            if self.is_recording is False:
                self.ble_serial_start = True
            self.anim.event_source.start()
            self.DevGraphButton.configure(text="Hide Graph")
        else:
            self.graph_display = False
            self.anim.event_source.stop()
            self.DevGraphButton.configure(text="Show Graph")

    def cameradisplay_clicked(self):
        if self.video_display is False:
            if self.changecamera():
                self.video_display = True
                self.CameraShowButton.configure(text="Hide Preview")
        else:
            self.video_display = False
            self.CameraShowButton.configure(text="Show Preview")

    def record_clicked(self):
        if self.is_recording is False:
            if self.changecamera() is False:
                return False
            if self.changedevice() is False:
                return False
            if self.setSaveDirectory() is False:
                return False
            if self.recordConstructor() is False:
                return False
            self.is_recording = True
            if self.graph_display is False:
                self.ble_serial_start = True
            self.RecordButton.configure(text="Record Stop")
        else:
            self.is_recording = False
            self.recordDestructor()
            self.RecordButton.configure(text="Record Start")    

    def click_close(self):
        self.halt_thread = True
        if self.video_display is True:
            self.video_display = False
        self.ble_loop.stop()
        self.th_camera.join()
        exit()

    def next_frame(self):
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            x = self.canvas1.winfo_reqwidth()/pil.width
            y = self.canvas1.winfo_reqheight()/pil.height
            ratio = x if x<y else y
            pil = pil.resize((int(ratio*pil.width),int(ratio*pil.height)))
            self.image_video_display = ImageTk.PhotoImage(pil, master=self.master)
            self.canvas1.create_image(0,0, image=self.image_video_display, anchor =tk.NW)

    def video_frame_timer(self):
        if self.video_display:
            self.next_frame()
        self.master.after(50, self.video_frame_timer)

    def next_graph(self, i):
        if len(self.graph_x) == len(self.graph_y):
            self.graph_display_clfc += 1
            if self.graph_display_clfc % 30 == 0:
                self.ax.cla()

            while len(self.graph_x) > 1 and self.graph_x[-1] - self.graph_x[0] > time_scale:
                self.graph_x.popleft()
                self.graph_y.popleft()

            if len(self.graph_x) > 0:
                self.ax.set_xlim((self.graph_x[-1] - time_scale, self.graph_x[-1]))
                self.ax.plot(self.graph_x, self.graph_y, color='C0', linestyle='-')

                self.ax.set_xlabel('time [sec]')
                self.ax.set_ylabel('Voltage [mV]')

    def graph_display_timer(self):
        if self.graph_display:
            self.next_graph()
        self.master.after(200, self.graph_display_timer)

    def getCameraDevID(self):
        s = self.combobox_camera.get()
        if len(s) < 7:
            return -1
        else:
            return int(s[7:])
        
    def changecamera(self):
        camID = self.getCameraDevID()
        if camID != -1:
            if self.cameraID == camID:
                return True
            with lock:
                if self.cap != None:
                    self.cap.release()
                self.cap = cv2.VideoCapture(camID)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cameraID = camID
            return True
        else:
            return False

    def changedevice(self):
        dev = self.combobox_dev.get()
        if dev != "":
            if self.deviceID == dev:
                return True
            with lock:
                self.com = serial.Serial(dev, 115200)
            self.deviceID = dev
            return True
        else:
            return False

    def CameraIndexes(self):
        index = 0
        arr = []
        i = 10
        while i > 0:
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                arr.append(index)
                cap.release()
            index += 1
            i -= 1
        return arr
    
    def setSaveDirectory(self):
        self.save_dir = self.IDirEntry.get()
        if self.save_dir != "":
            dir = self.save_dir + "/" + self.sub_dir
            try:
                os.makedirs(dir, exist_ok=False)
            except:
                tk.messagebox.showinfo(title="Error", message="The directory has already existed, or permission denied.")
                return False
            return True
        return False
    
    def recordConstructor(self):
        try:
            dir = self.save_dir + "/" + self.sub_dir + "/"
            self.device_data_f = open(dir + "data.dat", "xb")
            fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
            self.video_data_f = cv2.VideoWriter(dir + "video.mp4", fourcc, float(self.cap.get(cv2.CAP_PROP_FPS)), (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            self.record_time_diff = None
            self.record_time_video = None
            return True
        except:
            return False
        
    def recordDestructor(self):
        self.device_data_f.seek(0x4, os.SEEK_SET)
        self.device_data_f.write(struct.pack("<L", self.record_n_row))
        self.device_data_f.close()
        self.video_data_f.release()
        self.record_time_diff = None

    def CameraCapture(self):
        global frame
        while True:
            if self.halt_thread is True:
                break
            if self.video_display or self.is_recording:
                ret, frame = self.cap.read()
                if self.is_recording:
                    self.video_data_f.write(frame)
                    if self.record_time_video is None:
                        self.record_time_video = time.time_ns()
            else: 
                time.sleep(0.1)

    def decodeCOBS(self, data):
        zero = data[0]
        if zero > 8: 
            return False, data
        for i in range(1, 8):
            zero -= 1
            if zero == 0:
                zero = data[i]
                data[i] = 0
        # check sum
        sum = 0
        for i in range(1, 7):
            sum += data[i]
        if sum & 0xFF != data[7]:
            return False, data
        return True, data[1:7]

    def recordDevice(self, dat):
        self.device_data_f.write(dat)
        self.record_n_row += 1

    def recordInit(self, dat):
        f = self.device_data_f
        f.seek(0, os.SEEK_SET)
        f.write(b"DAT\x00\x00\x00\x00\x00")
        f.write(struct.pack('<l', self.record_time_diff))
        f.write(b"\x00\x00\x00\x00")
        f.write(time.strftime('%Y%m%d%H%M%S', time.localtime()).encode("ascii"))
        f.write(b"\x00" * (0x20 - f.tell()))
        self.record_n_row = 0

        # For Displaying Graph
        self.time_loop_cnt = 0
        self.time_start_rec = struct.unpack('<H', dat[18:20])[0]
        self.graph_display_reset = False
        self.graph_x.clear()
        self.graph_y.clear()

    # def appendRecord2Graph(self, val, time):
    #     time = time + self.time_loop_cnt * 65536 - self.time_start_rec
    #     if len(self.graph_x) > 0 and time / 32500.0 < self.graph_x[-1]:
    #         self.time_loop_cnt += 1
    #         time += 65536
    #     self.graph_x.append(time / 32500.0)
    #     self.graph_y.append(val * volt_coeff)

    def appendRecord2Graph(self, val, time, idx):
        time = time + self.time_loop_cnt * 65536 - self.time_start_rec
        if len(self.graph_x) > 0 and time * 9 / 100 < self.graph_x[-1]:
            self.time_loop_cnt += 1
            time += 65536
        self.graph_x.append(time * 9 / 100 + idx / 100)
        self.graph_y.append(val * volt_coeff)

    """
    def ReadSerial(self):
        try:
            packet = b""
            for i in range(9):
                buf = self.com.read()
                if buf == b'\x00' and i != 8:
                    return
                packet += buf
            ret, dat = self.decodeCOBS(bytearray(packet))
            if ret is False:
                return
            
            if self.is_recording:
                if self.record_time_diff is not None:
                    self.recordDevice(dat + b"\x00\x00")
                elif self.record_time_video is not None:
                    self.record_time_diff = time.time_ns() - self.record_time_video
                    self.recordInit(dat)
                    self.recordDevice(dat + b"\x00\x00")

            if self.graph_display:
                self.graph_interval_cnt += 1
                if self.graph_interval_cnt % self.graph_interval == 0:
                    tim = struct.unpack('<L', dat[0:4])[0]
                    val = struct.unpack('<h', dat[4:6])[0]
                    self.appendRecord2Graph(val, tim)
            
        except:
            print("Could not communicate with device.")

    """

    # def SerialProgram(self):
    #     while True:
    #         if self.halt_thread is True:
    #             break
    #         if self.ble_serial_start:
    #             self.ble_loop.run_until_complete(self.ble_run(self.mac_address, self.ble_loop))
    #             self.ble_serial_start = False
    #         else: time.sleep(0.001)

    
    def run_ble_loop(self):
        self.ble_loop = asyncio.new_event_loop()
        self.bleak_thread_ready.set()
        self.ble_loop.run_forever()


    async def _connect(self, timeout: float) -> None:
        client = BleakClient(self.mac_address)
        try:
            await client.connect(timeout=timeout)
        except asyncio.TimeoutError:
            raise BleakDeviceNotFoundError("Failed to connect: timeout") from asyncio.TimeoutError
        x = client.is_connected
        print("Connected: {0}".format(x))
        await client.start_notify("6e400003-b5a3-f393-e0a9-e50e24dcca9e", self.notification_handler)
        return client

    def connect_bleak(self, timeout=None):
        future = asyncio.run_coroutine_threadsafe(self._connect(timeout), self.ble_loop)
        return future.result(timeout)


    def notification_handler(self, sender, data: bytearray):

        try:
            # if len(data) < 9: return
            # for i in range(9):
            #     if data[i] == b'\x00' and i != 8:
            #         return
            # ret, dat = self.decodeCOBS(bytearray(data))
            # if ret is False:
            #     return

            val = []

            if len(data) < 20: return
            for i in range(9):
                val.append(struct.unpack('>H', data[2 * i: 2 * i + 2])[0])
                time = struct.unpack('<H', data[18:20])[0]

            # data(2b) * 9 + timestamp(2b)
            
            if self.is_recording:
                if self.record_time_diff is not None:
                    self.recordDevice(data + b"\x00\x00")
                elif self.record_time_video is not None:
                    self.record_time_diff = time.time_ns() - self.record_time_video
                    self.recordInit(data)
                    self.recordDevice(data + b"\x00\x00")

            if self.graph_display:
                for i in range(9):
                    self.graph_interval_cnt += 1
                    if self.graph_interval_cnt % self.graph_interval == 0:
                        self.appendRecord2Graph(val[i], time, i)
            #print(datetime.datetime.now())
            
        except:
            print("Could not communicate with device.")

    # async def ble_run(self, address, loop):
    #     async with BleakClient(address, loop=loop) as client:
    #         x = client.is_connected
    #         print("Connected: {0}".format(x))
    #         await client.start_notify("6e400003-b5a3-f393-e0a9-e50e24dcca9e", self.notification_handler)

    #         while (self.graph_display or self.is_recording) and self.halt_thread is False:
    #             await asyncio.sleep(0.001)


root = tk.Tk()
app = Application(master=root)
app.mainloop()
