# =============================================================================
# BAGIAN A: IMPORT & KONFIGURASI GLOBAL test
# =============================================================================
import threading
import time
import threading
import queue
import subprocess

import sys
import socket
import os
IS_WINDOWS = os.name == "nt"
import re
import serial
import datetime
import cv2
import mysql.connector
import customtkinter as ctk
import tkinter as tk
import tkinter.messagebox as messagebox
import cv2.aruco as aruco
import numpy as np
from PIL import Image, ImageTk
from detection.yolo_detector import YoloDetector
import pandas as pd
from playsound import playsound
from mysql.connector import Error
from kalibrasi import JendelaKalibrasi

from database.connection import connection as get_db_connection

def load_icon(path, size=(20, 20)):
    """Memuat file gambar untuk ikon."""
    try:
        image = Image.open(path).resize(size)
        return ctk.CTkImage(light_image=image, dark_image=image, size=size)
    except Exception as e:
        print(f"Failed to load icon {path}: {e}")
        return None

# =============================================================================
# BAGIAN B: KELAS UTAMA APLIKASI
# =============================================================================
class App(ctk.CTk):

    # -------------------------------------------------------------------------
    # 1. INISIALISASI & SETUP AWAL
    # -------------------------------------------------------------------------
    def __init__(self):
        """
        - Menginisialisasi window utama.
        - Mendefinisikan semua variabel sistem (state, timer, data, dll).
        - Menyiapkan koneksi hardware (Serial).
        - Memuat model (YOLO).
        - Memanggil self.setup_ui() untuk membangun tampilan.
        """
        super().__init__()
        self.title("Wire Selection System")
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        # 1. Keep your working scaling configuration
        ctk.set_widget_scaling(1.5)  
        ctk.set_window_scaling(1.5)  

        # 2. Tell Linux to toggle its native fullscreen mode safely
        try:
            self.state('zoomed')
        except Exception:
            self.attributes('-zoomed', True)
        # self.ser = serial.Serial('/dev/tty0', baudrate=115200, timeout=0.01)
        self.is_dark_mode = False
        # self.ser = serial.Serial('COM6', baudrate=115200, timeout=0.01)

        self.kalibrasi_app = None
        # System variables
        self.timer_running = False
        self.timer_paused = False
        self.start_time = None
        self.paused_time = 0
        self.elapsed_time = 0
        self.camera_on = False
        self.cap = None
        #  self.model = YOLO(r'GUI/best2.engine', task='detect')#Blue.engine
        self.model = YoloDetector(r'UiBerdikari/SRC/best2.pt', task='detect')
        self.ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.MARKER_SIZE = 5.0
        self.excel_file_path = r'/home/berdikari/HandDetection/UiBerdikari/Database/barcode_data.xlsx'
        self.excel_file_user = r'/home/berdikari/HandDetection/UiBerdikari/DataUser/data_user.xlsx'
        self.django_process = None

        # Inisialisasi Aruco API baru
        self.ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)#DICT_APRILTAG_36h11  #DICT_4X4_50
        self.ARUCO_PARAMETERS = cv2.aruco.DetectorParameters()
        self.ARUCO_PARAMETERS.adaptiveThreshWinSizeMin = 5
        self.ARUCO_PARAMETERS.adaptiveThreshWinSizeMax = 23
        self.ARUCO_PARAMETERS.adaptiveThreshWinSizeStep = 5
        self.ARUCO_PARAMETERS.minMarkerPerimeterRate = 0.08
        self.ARUCO_PARAMETERS.maxMarkerPerimeterRate = 4.0
        self.ARUCO_PARAMETERS.minCornerDistanceRate = 0.05
        self.ARUCO_PARAMETERS.minOtsuStdDev = 5.0
        self.ARUCO_PARAMETERS.polygonalApproxAccuracyRate = 0.03
        self.ARUCO_PARAMETERS.minMarkerDistanceRate = 0.05

        self.ARUCO_DETECTOR = cv2.aruco.ArucoDetector(self.ARUCO_DICT, self.ARUCO_PARAMETERS)

        # Variabel untuk logika anti-duplikat
        self.last_detected_slot = None

        self.blinking_tags = {}  # Dictionary untuk melacak tag dan status warnanya
        self.error_tag_counter = 0 # Untuk membuat nama tag yang unik
        self.is_blinking_active = False # Penanda apakah loop kedip sedang berjalan
        self.is_status_blinking = False #
        self.original_status_fg_color = None
        self.is_status_blinking_selection = False # Flag khusus untuk kedip kuning saat seleksi
        self.original_status_fg_color_selection = None # Untuk menyimpan warna asli status bar

        # Threading variables
        self.camera_thread = None
        self.detection_thread = None
        self.serial_thread = None
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)

        # --- Thread-safety helpers ---
        # Cache warna LED, dibaca dari detection_thread (background) agar
        # TIDAK perlu memanggil widget.cget() lintas-thread (tidak aman di Tkinter).
        # Cache ini selalu di-update lewat self._set_led_color() yang berjalan di main thread.
        self.led_color_cache = {}

        # Queue + worker thread khusus untuk insert log error ke database,
        # supaya koneksi/insert MySQL tidak memblok detection_thread (penyebab freeze kamera/GUI).
        self.db_log_queue = queue.Queue()
        self.db_worker_thread = threading.Thread(target=self._db_worker_loop, daemon=True)
        self.db_worker_thread.start()

        # Initialize camera calibration
        self.camera_matrix = np.array([[800, 0, 320],
                                     [0, 800, 240],
                                     [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1))

        # System state
        self.system_started = False
        self.cycle_active = False
        self.current_cycle = 0
        self.total_cycles = 0
        self.max_selection_count = 0
        self.is_manual_selection_mode = False  # Flag untuk mode seleksi
        self.manual_selected_slots = []      # List untuk menyimpan slot pilihan
        # self.current_start = 1
        # self.current_end = 12
        self.current_group = 'A'
        self.detected_objects = set()
        self.blinking = False
        self.all_wires_taken = False
        self.last_logged_error_slot = None
        self.plan_work_order = 0
        self.accumulation_total = 0
        self.accumulation_user = ""
        self.error_delay_threshold = 2.0
        self.slot_data = {
    # Slot A
    "1A": {"x1": 250, "x2": 315, "y1": 260, "y2": 350, "z_min": 180, "z_max": 205},
    "2A": {"x1": 185, "x2": 245, "y1": 260, "y2": 350, "z_min": 180, "z_max": 205},
    "3A": {"x1": 110, "x2": 180, "y1": 260, "y2": 350, "z_min": 180, "z_max": 204},
    "4A": {"x1": 35, "x2": 105, "y1": 260, "y2": 356, "z_min": 181, "z_max": 208},
    "5A": {"x1": 245, "x2": 315, "y1": 152, "y2": 246, "z_min": 172, "z_max": 199},
    "6A": {"x1": 175, "x2": 240, "y1": 152, "y2": 246, "z_min": 172, "z_max": 199},
    "7A": {"x1": 95, "x2": 170, "y1": 152, "y2": 246, "z_min": 172, "z_max": 199},
    "8A": {"x1": 16, "x2": 90, "y1": 152, "y2": 249, "z_min": 172, "z_max": 200},
    "9A": {"x1": 240, "x2": 315, "y1": 27, "y2": 141, "z_min": 160, "z_max": 189},
    "10A": {"x1": 155, "x2": 235, "y1": 27, "y2": 141, "z_min": 160, "z_max": 189},
    "11A": {"x1": 75, "x2": 150, "y1": 27, "y2": 141, "z_min": 160, "z_max": 189},
    "12A": {"x1": 2, "x2": 70, "y1": 27, "y2": 141, "z_min": 174, "z_max": 195},
    # Slot B
    "1B":  {"x1": 530, "x2": 600, "y1": 260, "y2": 350, "z_min": 163, "z_max": 209},
    "2B":  {"x1": 457, "x2": 525, "y1": 260, "y2": 350, "z_min": 168, "z_max": 212},
    "3B":  {"x1": 387, "x2": 450, "y1": 260, "y2": 350, "z_min": 168, "z_max": 208},
    "4B":  {"x1": 322, "x2": 385, "y1": 260, "y2": 350, "z_min": 168, "z_max": 211},
    "5B":  {"x1": 545, "x2": 615, "y1": 152, "y2": 246, "z_min": 160, "z_max": 200},
    "6B":  {"x1": 465, "x2": 540, "y1": 152, "y2": 246, "z_min": 160, "z_max": 199},
    "7B":  {"x1": 395, "x2": 460, "y1": 152, "y2": 246, "z_min": 160, "z_max": 205},
    "8B":  {"x1": 322, "x2": 390, "y1": 152, "y2": 246, "z_min": 163, "z_max": 194},
    "9B":  {"x1": 560, "x2": 635, "y1": 27,  "y2": 141, "z_min": 152, "z_max": 197},
    "10B": {"x1": 480, "x2": 555, "y1": 27,  "y2": 141, "z_min": 138, "z_max": 188},
    "11B": {"x1": 400, "x2": 475, "y1": 27,  "y2": 141, "z_min": 153, "z_max": 179},
    "12B": {"x1": 322, "x2": 395, "y1": 27,  "y2": 141, "z_min": 155, "z_max": 182},
    
}
        # Setup UI
        self.setup_ui()
    # -------------------------------------------------------------------------
    # 2. PEMBUATAN & PENGATURAN TAMPILAN (UI)
    # -------------------------------------------------------------------------
    def setup_ui(self):
        """Membangun semua widget utama (sidebar, main_frame)."""
        self.sidebar = ctk.CTkFrame(self, width=250, fg_color="#d8ebff")
        self.sidebar.pack(side="left", fill="y")

        # Add logo
        try:
            logo_frame = tk.Frame(self.sidebar, bg="#d8ebff")
            logo_frame.pack(pady=20)
            self.logo_img1 = Image.open("UiBerdikari/SRC/polteklogo.png").resize((110, 98))
            self.logo_photo1 = ImageTk.PhotoImage(self.logo_img1)
            logo1_label = tk.Label(logo_frame, image=self.logo_photo1, bg="#d8ebff")
            logo1_label.pack(side="left", padx=10)

            self.logo_img2 = Image.open("UiBerdikari/SRC/sbilogo.png").resize((110, 98))
            self.logo_photo2 = ImageTk.PhotoImage(self.logo_img2)
            logo1_label = tk.Label(logo_frame, image=self.logo_photo2, bg="#d8ebff")
            logo1_label.pack(side="left", padx=10)
        except:
            pass

        # Load icons
        self.home_icon = load_icon("UiBerdikari/SRC/IMG/home.png")
        self.info_icon = load_icon("UiBerdikari/SRC/IMG/info.png")
        self.mode_icon = load_icon("UiBerdikari/SRC/IMG/mode.png")
        self.start_icon = load_icon("UiBerdikari/SRC/IMG/start.png")
        self.pause_icon = load_icon("UiBerdikari/SRC/IMG/pause.png")
        self.reset_icon = load_icon("UiBerdikari/SRC/IMG/Reset.png")
        self.camera_icon = load_icon("UiBerdikari/SRC/IMG/camera.png")
        self.complete_icon = load_icon("UiBerdikari/SRC/IMG/complete.png")
        self.server_icon = load_icon("UiBerdikari/SRC/IMG/server.png")
        self.server_icon = load_icon("UiBerdikari/SRC/IMG/complete.png")
        self.exit_icon = load_icon("UiBerdikari/SRC/IMG/exit.png")
        self.cal_icon = load_icon("UiBerdikari/SRC/IMG/calibration.png")

        # Navigation buttons
        self.home_btn = ctk.CTkButton(self.sidebar, text="Home", anchor="w", image=self.home_icon,
                                    border_color="#EBF9FF", border_width=3, compound="left", command=self.show_home)
        self.home_btn.pack(fill="x", pady=5, padx=10)

        self.info_btn = ctk.CTkButton(self.sidebar, text="Information", anchor="w", image=self.info_icon,
                                    border_color="#EBF9FF", border_width=3, compound="left", command=self.show_info)
        self.info_btn.pack(fill="x", pady=5, padx=10)

        self.mode_btn = ctk.CTkButton(self.sidebar, text="Mode", anchor="w", image=self.mode_icon,
                                    border_color="#EBF9FF", border_width=3, compound="left", command=self.toggle_mode)
        self.mode_btn.pack(fill="x", pady=5, padx=10)

        self.cal_btn = ctk.CTkButton(self.sidebar, text="Calibration", anchor="w", image=self.cal_icon,
                                    border_color="#EBF9FF", border_width=3, compound="left", command=self.buka_kalibrasi)
        self.cal_btn.pack(fill="x", pady=5, padx=10)

        # Control panel
        self.control_frame = ctk.CTkFrame(self.sidebar, fg_color="#d8ebff")
        self.control_frame.pack(pady=10, fill="x", padx=10)

        self.control_title = ctk.CTkLabel(self.control_frame, text="CONTROL", font=("Arial", 18, "bold"))
        self.control_title.pack(anchor="w")

        self.django_btn = ctk.CTkButton(self.control_frame, text="Run Server", anchor="w", 
                                border_color="#EBF9FF", border_width=3, command=self.run_django_server, image=self.server_icon, compound="left") 
        self.django_btn.pack(fill="x", pady=5)

        self.start_btn = ctk.CTkButton(self.control_frame, text="Start", anchor="w", image=self.start_icon,
                                     border_color="#EBF9FF", border_width=3, compound="left", command=self.start_system)
        self.start_btn.pack(fill="x", pady=5)

        self.reset_btn = ctk.CTkButton(self.control_frame, text="Reset", anchor="w", image=self.reset_icon,
                                     border_color="#EBF9FF", border_width=3, compound="left", command=self.reset_system)

        self.reset_btn.pack(fill="x", pady=5)
        self.pause_btn = ctk.CTkButton(self.control_frame, text="Pause", anchor="w", image=self.pause_icon,
                                     border_color="#EBF9FF", border_width=3, compound="left", command=self.pause_system)
        self.pause_btn.pack(fill="x", pady=5)

        self.complete_btn = ctk.CTkButton(self.control_frame, text="Complete", anchor="w", image=self.complete_icon,
                                        border_color="#EBF9FF", border_width=3, compound="left", command=self.complete_system)                  
        self.complete_btn.pack(fill="x", pady=5)
        self.camera_btn = ctk.CTkButton(self.control_frame, text="Start Camera", anchor="w", image=self.camera_icon,
                                      border_color="#EBF9FF", border_width=3, compound="left", command=self.toggle_camera)
        self.camera_btn.pack(fill="x", pady=5)

        self.delay_label = ctk.CTkLabel(self.control_frame, text="Delays :", anchor="w")
        self.delay_label.pack(fill="x", pady=(10, 0)) # pady=(top, bottom)

        self.delay_options = ["1.0", "1.5", "2.0", "2.5", "3.0"]
        self.delay_option_menu = ctk.CTkOptionMenu(
            self.control_frame,
            button_color="#3B8ED0",
            values=self.delay_options,
            command=self.set_error_delay_threshold # This function will handle the selection
        )
        self.delay_option_menu.set("1.0") # Set the default displayed value
        self.delay_option_menu.pack(fill="x", pady=5)

        self.exit_btn = ctk.CTkButton(self.control_frame, text="Exit", anchor="w", image=self.exit_icon,
                              border_color="#EBF9FF", border_width=3, compound="left", 
                              command=self.on_closing, fg_color="#c21111", hover_color="#990e0e")
        self.exit_btn.pack(fill="x", pady=5)

        # Main panel
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(side="left", expand=True, fill="both", padx=5, pady=5)

        self.create_home_layout()

    def create_home_layout(self):
        """Menciptakan dan menata semua widget di layar utama."""
        if hasattr(self, '_after_id'):
            self.after_cancel(self._after_id)
        # Clear existing widgets
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Top frame
        self.top_frame = ctk.CTkFrame(self.main_frame, height=100, fg_color="#d8ebff", border_color="#FFFFFF", border_width=5)
        self.top_frame.pack(fill="x")

        self.title_label = ctk.CTkLabel(self.top_frame, text="WIRE SELECTION SYSTEM",
                                      font=("Arial", 24, "bold"))
        self.title_label.pack(pady=20)

        # Middle frame
        self.middle_frame = ctk.CTkFrame(self.main_frame)
        self.middle_frame.pack(expand=True, fill="both")

        # Left box (LED indicators)
        self.left_box = ctk.CTkFrame(self.middle_frame, fg_color="#d8ebff", corner_radius=15, width=560, border_color="#FFFFFF", border_width=5)
        self.left_box.pack(side="left", fill="y", padx=10, pady=10, ipadx=10, ipady=10)
        self.left_box.pack_propagate(False)

        # LED indicator setup
        led_container = ctk.CTkFrame(self.left_box, fg_color="transparent")
        led_container.pack(expand=True, pady=10)

        led_frame = ctk.CTkFrame(led_container, fg_color="transparent")
        led_frame.pack()

        # Group headers
        header_frame = ctk.CTkFrame(led_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, columnspan=9, pady=(0, 10))

        self.grup_a_label = ctk.CTkLabel(header_frame, text="Group A", font=("Arial", 20, "bold"))
        self.grup_a_label.grid(row=0, column=1, columnspan=3, sticky="n")

        self.grup_b_label = ctk.CTkLabel(header_frame, text="Group B", font=("Arial", 20, "bold"))
        self.grup_b_label.grid(row=0, column=5, columnspan=3, sticky="n")

        ctk.CTkLabel(header_frame, text="\t\t\t\t", width=10).grid(row=0, column=4)

        # Create LED buttons
        self.leds_a = []
        self.leds_b = []
        led_numbers = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

        for row in range(3):
            for col in range(4):
                idx = row * 4 + col
                if idx >= len(led_numbers):
                    continue

                led_number = led_numbers[idx]

                # Group A LED
                label_a = f"{led_number}A" # Ambil label untuk command
                led_a = ctk.CTkButton(
                  led_frame,
                  width=40,
                  height=40,
                  text=f"{led_number}A",
                  fg_color="gray",
                  hover_color="lightgray",
                  corner_radius=20,
                  font=("Arial", 12),
                  command=lambda label=label_a: self.on_led_click(label)
              )
                led_a.grid(row=row + 1, column=col, padx=2, pady=5)
                self.leds_a.append(led_a)
                self.led_color_cache[label_a] = "gray"

                # Group B LED
                label_b = f"{led_number}B" # Ambil label untuk command
                led_b = ctk.CTkButton(
                  led_frame,
                  width=40,
                  height=40,
                  text=f"{led_number}B",
                  fg_color="gray",
                  hover_color="lightgray",
                  corner_radius=20,
                  font=("Arial", 12),
                  command=lambda label=label_b: self.on_led_click(label)
              )
                led_b.grid(row=row + 1, column=col + 5, padx=2, pady=5)
                self.leds_b.append(led_b)
                self.led_color_cache[label_b] = "gray"

            # Separator
            separator = ctk.CTkLabel(led_frame, text="|", font=("Arial", 35), text_color="gray")
            separator.grid(row=row + 1, column=4, padx=5)

        for i in range(9):
            led_frame.grid_columnconfigure(i, weight=1)

        # Right box (camera)
        self.right_box = ctk.CTkFrame(self.middle_frame, fg_color="#d8ebff", corner_radius=15, width=720, border_color="#FFFFFF", border_width=5)
        self.right_box.pack(side="left", fill="y", padx=10, pady=10)
        self.right_box.pack_propagate(False)

        self.camera_frame = tk.Frame(self.right_box, width=720, height=480, bg="black")
        self.camera_frame.pack(pady=10)
        self.camera_frame.pack_propagate(False)

        self.camera_label = tk.Label(self.camera_frame, bg="black")
        self.camera_label.pack(expand=True)

        self.camera_placeholder = tk.Label(self.camera_frame, text="Camera is OFF",
                                         fg="white", bg="black", font=("Arial", 24))
        self.camera_placeholder.pack(expand=True)

        # Bottom frame
        self.bottom_frame = ctk.CTkFrame(self.main_frame)
        self.bottom_frame.pack(fill="x", padx=10, pady=10)

        # Wire info section
        self.wire_info = ctk.CTkScrollableFrame(
            self.bottom_frame, 
            label_text="WIRE INFORMATION",  # Judul langsung di sini
            label_font=("Arial", 16, "bold"),
            fg_color="#d8ebff", 
            width=200, 
            border_color="#FFFFFF", 
            border_width=5
        )
        self.wire_info.pack(side="left", padx=5, fill="y")

        self.wire_info_labels = []
        self.wire_info_entries = []

        # User entry
        lbl = ctk.CTkLabel(self.wire_info, text="User")
        lbl.pack(anchor="w", padx=10)
        user_entry = ctk.CTkEntry(self.wire_info)
        user_entry.pack(fill="x", padx=10, pady=(0,5))
        self.wire_info_labels.append(lbl)
        self.wire_info_entries.append(user_entry)

        # Barcode entry
        lbl = ctk.CTkLabel(self.wire_info, text="ID Barcode")
        lbl.pack(anchor="w", padx=10)
        barcode_entry = ctk.CTkEntry(self.wire_info)
        barcode_entry.pack(fill="x", padx=10, pady=(0,5))
        self.wire_info_labels.append(lbl)
        self.wire_info_entries.append(barcode_entry)

        # Current Data entry
        lbl = ctk.CTkLabel(self.wire_info, text="Data Current")
        lbl.pack(anchor="w", padx=10)
        current_entry = ctk.CTkEntry(self.wire_info, state="readonly")
        current_entry.pack(fill="x", padx=10, pady=(0,5))
        self.wire_info_labels.append(lbl)
        self.wire_info_entries.append(current_entry)

        lbl = ctk.CTkLabel(self.wire_info, text="Total Work Order")
        lbl.pack(anchor="w", padx=10)
        total_wo_entry = ctk.CTkEntry(self.wire_info, state="readonly")
        total_wo_entry.pack(fill="x", padx=10, pady=(0,5))
        self.wire_info_labels.append(lbl)
        self.wire_info_entries.append(total_wo_entry)

        # Remaining entry
        lbl = ctk.CTkLabel(self.wire_info, text="Total Bundle")
        lbl.pack(anchor="w", padx=10)
        remaining_entry = ctk.CTkEntry(self.wire_info, state="readonly")
        remaining_entry.pack(fill="x", padx=10, pady=(0,5))
        self.wire_info_labels.append(lbl)
        self.wire_info_entries.append(remaining_entry)

        # Connection info
        self.connect_info = ctk.CTkFrame(self.bottom_frame, fg_color="#d8ebff", border_color="#FFFFFF", border_width=5)
        self.connect_info.pack(side="left", expand=True, fill="both", padx=5)

        self.connect_info_label = ctk.CTkLabel(self.connect_info, text="CONNECTIFY INFORMATIONS",
                                             font=("Arial", 16, "bold"))
        self.connect_info_label.pack(anchor="n", padx=10, pady=5)

        status_frame = ctk.CTkFrame(self.connect_info, fg_color="#d8ebff")
        status_frame.pack(anchor="w", padx=10, pady=(0,5))

        self.status_label = ctk.CTkLabel(status_frame, text="Status", font=("Arial", 14, "bold"))
        self.status_label.pack(side="left")

        self.led_canvas = tk.Canvas(status_frame, width=20, height=20, bg="#d8ebff", highlightthickness=0)
        self.led = self.led_canvas.create_oval(2, 2, 18, 18, fill="gray")
        self.led_canvas.pack(side="left", padx=10)

        self.status_entry = ctk.CTkEntry(self.connect_info, font=("Arial", 14, "bold"),
                                       justify="center", state="readonly")
        self.status_entry.pack(fill="x", padx=10, pady=(0, 10))
        self.status_entry.insert(0, "Not Connected")

        self.history_frame = ctk.CTkFrame(self.connect_info, fg_color="transparent")
        self.history_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Frame untuk History
        self.history_box = ctk.CTkFrame(self.history_frame, fg_color="#f0f0f0", height=60, border_width=2)
        self.history_box.pack(side="left", expand=True, fill="both", padx=(0, 5))
        self.history_label = ctk.CTkLabel(self.history_box, text="History", anchor="center", font=("Arial", 14, "bold"))
        self.history_label.pack(fill="x", padx=5, pady=2)
        self.history_text = ctk.CTkTextbox(self.history_box, height=50, state="disabled")
        self.history_text.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        # Frame untuk Barcode Info
        self.barcode_info_box = ctk.CTkFrame(self.history_frame, fg_color="#f0f0f0", height=60, border_width=2)
        self.barcode_info_box.pack(side="left", expand=True, fill="both", padx=(5, 0))
        self.barcode_info_label = ctk.CTkLabel(self.barcode_info_box, text="Information ID Barcode", anchor="center", font=("Arial", 14, "bold"))
        self.barcode_info_label.pack(fill="x", padx=5, pady=2)
        self.barcode_info_text = ctk.CTkTextbox(self.barcode_info_box, height=50, state="disabled")
        self.barcode_info_text.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        # Process frame
        self.process_frame = ctk.CTkScrollableFrame(self.bottom_frame,   label_text="TIME",  label_font=("Arial", 16, "bold"), width=250, fg_color="#d8ebff",  border_color="#FFFFFF",  border_width=5)
        self.process_frame.pack(side="left", padx=5, fill="y")

        # --- Konfigurasi Grid di dalam Scrollable Frame ---
        self.process_frame.grid_columnconfigure(0, weight=1)

        # Timer (Baris 0 dan 1)
        timer_label = ctk.CTkLabel(self.process_frame, text="Timer")
        timer_label.grid(row=0, column=0, padx=10, pady=(5, 0), sticky="w")
        self.timer_entry = ctk.CTkEntry(self.process_frame, justify="center", state="readonly")
        self.timer_entry.insert(0, "00:00:00")
        self.timer_entry.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="ew")

        # Actual Time (Baris 2 dan 3)
        self.time_label = ctk.CTkLabel(self.process_frame, text="Actual Time")
        self.time_label.grid(row=2, column=0, padx=10, pady=(5, 0), sticky="w")
        self.time_entry = ctk.CTkEntry(self.process_frame, state="readonly", justify="center")
        self.time_entry.grid(row=3, column=0, padx=10, pady=(0, 5), sticky="ew")

        # Remaining Bundle entry (Baris 4 dan 5)
        counter_label = ctk.CTkLabel(self.process_frame, text="Remaining Bundle", text_color="black")
        counter_label.grid(row=4, column=0, padx=10, pady=(5, 0), sticky="w")
        self.counter_entry = ctk.CTkEntry(self.process_frame, justify="center", state="readonly")
        self.counter_entry.grid(row=5, column=0, padx=10, pady=(0, 5), sticky="ew")

        # Work Order entry (Baris 6 dan 7)
        currently_label = ctk.CTkLabel(self.process_frame, text="Remaining Work Order")
        currently_label.grid(row=6, column=0, padx=10, pady=(5, 0), sticky="w")
        self.currently_entry = ctk.CTkEntry(self.process_frame, justify="center", state="readonly")
        self.currently_entry.grid(row=7, column=0, padx=10, pady=(0, 5), sticky="ew")

        # Accumulation entry (Baris 8 dan 9)
        accumulation_label = ctk.CTkLabel(self.process_frame, text="Accumulation")
        accumulation_label.grid(row=8, column=0, padx=10, pady=(5, 0), sticky="w")
        self.accumulation_entry = ctk.CTkEntry(self.process_frame, justify="center", state="readonly")
        self.accumulation_entry.grid(row=9, column=0, padx=10, pady=(0, 5), sticky="ew")

        self.update_time()

    def show_home(self):
        self.create_home_layout()
        self.reset_system()

    def buka_kalibrasi(self):
        if self.kalibrasi_app is not None and self.kalibrasi_app.top.winfo_exists():
            self.kalibrasi_app.top.lift()
            return

        self.kalibrasi_app = JendelaKalibrasi(self)

    def show_info(self):
        self.reset_system()

        # Hentikan update_time sebelum menghancurkan widget
        if hasattr(self, '_after_id'):
            self.after_cancel(self._after_id)

        # Clear existing widgets
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        info_frame = ctk.CTkFrame(self.main_frame, border_color="#FFFFFF", border_width=3)
        info_frame.pack(expand=True, fill="both", padx=10, pady=10)

        self.tabview = ctk.CTkTabview(info_frame)
        self.tabview.pack(expand=True, fill="both", padx=5, pady=5)

        self.tabview.add("Information 1")
        self.tabview.add("Information 2") 
        self.tabview.add("Information 3")

        self.tabview.tab("Information 1").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Information 2").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Information 3").grid_columnconfigure(0, weight=1)

        # Tab 1
        label1 = ctk.CTkLabel(self.tabview.tab("Information 1"), text="Otomatisasi Sistem Pemilih Kabel pada Line Produksi Otomotif Wiring System",
                            font=("Arial", 25, "bold"))
        label1.pack(pady=20)
        label1 = ctk.CTkLabel(self.tabview.tab("Information 1"), text="Invensi ini merupakan sistem berbasis computer vision untuk memandu operator dalam pemilihan kabel pada proses produksi wire harness. Sistem menggunakan\nkamera dan komputer yang dilengkapi teknologi YOLO untuk mendeteksi sarung tangan operator secara real-time, serta ArUco marker untuk menentukan posisi sarung\ntangan secara presisi. Deteksi ini memverifikasi kesesuaian pengambilan kabel dengan slot yang ditentukan. Sistem juga dilengkapi pemindai barcode untuk menentukan\nslot kabel yang harus diambil sesuai perintah kerja.",
                            font=("Arial", 14))
        label1.pack(pady=20)

        label1 = ctk.CTkLabel(self.tabview.tab("Information 1"), text="Panduan Penggunaan Alat",
                            font=("Arial", 25, "bold"))
        label1.pack(pady=20)

        # Create horizontal frame for side-by-side images
        image_container = ctk.CTkFrame(self.tabview.tab("Information 1"), fg_color="transparent")
        image_container.pack(pady=10)

        # First image frame
        frame_img1 = ctk.CTkFrame(image_container, fg_color="transparent")
        frame_img1.pack(side="left", padx=10, expand=True)

        try:
            # Load and resize the first landscape image (smaller width for side-by-side)
            landscape_img1 = Image.open("UiBerdikari/SRC/Flowalat.png")
            landscape_img1 = landscape_img1.resize((680, 400), Image.LANCZOS)  # Reduced width
            landscape_photo1 = ImageTk.PhotoImage(landscape_img1)

            # Create label for first image
            landscape_label1 = tk.Label(frame_img1, image=landscape_photo1, bg="#f0f0f0")
            landscape_label1.image = landscape_photo1
            landscape_label1.pack()

            # Caption for first image
            caption1 = ctk.CTkLabel(frame_img1, 
                                text="Diagram alur - Bagian 1",
                                font=("Arial", 12))
            caption1.pack(pady=(5, 0))
        except Exception as e:
            print(f"Error loading first landscape image: {e}")
            ctk.CTkLabel(frame_img1, 
                        text="[Diagram bagian 1]",
                        font=("Arial", 12, "italic")).pack()

        # Second image frame
        frame_img2 = ctk.CTkFrame(image_container, fg_color="transparent")
        frame_img2.pack(side="left", padx=10, expand=True)

        try:
            # Load and resize the second landscape image (same size as first)
            landscape_img2 = Image.open("UiBerdikari/SRC/Flowalat2.png")
            landscape_img2 = landscape_img2.resize((680, 400), Image.LANCZOS)  # Same dimensions
            landscape_photo2 = ImageTk.PhotoImage(landscape_img2)

            # Create label for second image
            landscape_label2 = tk.Label(frame_img2, image=landscape_photo2, bg="#f0f0f0")
            landscape_label2.image = landscape_photo2
            landscape_label2.pack()

            # Caption for second image
            caption2 = ctk.CTkLabel(frame_img2, 
                                text="Diagram alur - Bagian 2",
                                font=("Arial", 12))
            caption2.pack(pady=(5, 0))
        except Exception as e:
            print(f"Error loading second landscape image: {e}")
            ctk.CTkLabel(frame_img2, 
                        text="[Diagram bagian 2]",
                        font=("Arial", 12, "italic")).pack()

        # Rest of the code remains the same...
        # Tab 2
        label2 = ctk.CTkLabel(self.tabview.tab("Information 2"), text="Politeknik Negeri Batam",
                            font=("Arial", 25, "bold"))
        label2.pack(pady=20)
        label2 = ctk.CTkLabel(self.tabview.tab("Information 2"), text="Politeknik Negeri Batam (Polibatam) merupakan satu-satunya Perguruan Tinggi Negeri (PTN) Vokasi di kawasan perdagangan dan pelabuhan bebas Batam, Bintan, dan\nKarimun Provinsi Kepulauan Riau. Selain terletak di salah satu kawasan pusat pertumbuhan ekonomi nasional, Polibatam juga terletak di wilayah terdepan dan terluar\nwilayah Negara Kesatuan republik Indonesia yang berbatasan langsung dengan perairan internasional.",
                            font=("Arial", 14))
        label2.pack(pady=5)

        # Tab 3
        label3 = ctk.CTkLabel(self.tabview.tab("Information 3"), text="PT Sumitomo Wiring System Batam Indonesia (SWSBI)", 
                            font=("Arial", 25, "bold"))
        label3.pack(pady=20)
        label3 = ctk.CTkLabel(self.tabview.tab("Information 3"), text="PT Sumitomo Wiring System Batam Indonesia (SWSBI) adalah perusahaan asal Jepang telah berdiri sejak tahun 1990 yang memproduksi kabel harness untuk mobil.\nKapasitas produksi wire harness 260.000 set per tahun. Sebesar 100 persen hasil produksinya diserap pasar ekspor, seperti ke Thailand, Vietnam, dan Tiongkok.", 
                            font=("Arial", 14))
        label3.pack(pady=5)

        if hasattr(self, 'title_label'):
            del self.title_label

    def _set_led_color(self, led, label, color):
        """
        Helper terpusat untuk mengubah warna LED. HARUS dipanggil dari main thread
        (lewat callback langsung atau self.after). Selain mengubah widget,
        fungsi ini juga memperbarui led_color_cache agar detection_thread bisa
        membaca status warna tanpa menyentuh widget Tkinter secara langsung.
        """
        led.configure(fg_color=color)
        self.led_color_cache[label] = color

    def reset_led_colors(self):
        """Mengembalikan warna semua LED ke default (abu-abu)."""
        for led in self.leds_a:
            self._set_led_color(led, led.cget("text"), "gray")
        for led in self.leds_b:
            self._set_led_color(led, led.cget("text"), "gray")

    def get_led_by_label(self, label):
        """Mencari dan mengembalikan objek widget LED berdasarkan teks labelnya."""
        all_leds = self.leds_a + self.leds_b
        for led in all_leds:
            if led.cget("text") == label:
                return led
        return None

    def on_led_click(self, label):
        """Handler saat tombol LED diklik."""
        if not self.is_manual_selection_mode:
            return

        if not label.endswith(self.current_group):
            return # Jika grup salah, jangan lakukan apa-apa

        led_widget = self.get_led_by_label(label)
        if not led_widget:
            return

        # Logika untuk DESELECT (membatalkan pilihan)
        if label in self.manual_selected_slots:
            self._set_led_color(led_widget, label, "gray")
            self.manual_selected_slots.remove(label)
        # Logika untuk SELECT (memilih)
        else:
            if len(self.manual_selected_slots) >= self.max_selection_count:
                messagebox.showwarning("Batas Tercapai", 
                                    f"Anda hanya dapat memilih maksimal {self.max_selection_count} slot.", 
                                    parent=self)
                return # Hentikan fungsi jika sudah mencapai batas

            self._set_led_color(led_widget, label, "#FFD700") # Warna kuning
            self.manual_selected_slots.append(label)

        # Update tampilan Data Current setiap kali ada perubahan
        self.wire_info_entries[2].configure(state="normal")
        self.wire_info_entries[2].delete(0, "end")
        self.wire_info_entries[2].insert(0, str(len(self.manual_selected_slots)))
        self.wire_info_entries[2].configure(state="readonly")

        self.update_status(f"Pilih {self.max_selection_count} slot untuk Grup {self.current_group} ({len(self.manual_selected_slots)}/{self.max_selection_count})")

        # --- SOLUSI MASALAH 3: Pop-up Otomatis ---
        if len(self.manual_selected_slots) == self.max_selection_count:
            self._trigger_confirmation_popup()

    def _trigger_confirmation_popup(self):
        """Menampilkan pop-up konfirmasi dan memulai siklus jika 'Yes'."""
        confirm = messagebox.askyesno("Konfirmasi",
                                    f"Anda telah memilih {len(self.manual_selected_slots)} slot.\n\nLanjutkan proses?",
                                    parent=self)
        if confirm:
            self.is_status_blinking_selection = False
            if self.original_status_fg_color_selection:
                self.status_entry.configure(fg_color=self.original_status_fg_color_selection)
            self.is_manual_selection_mode = False
            self.reset_btn.configure(state="normal")
            self.pause_btn.configure(state="normal")
            self.django_btn.configure(state="normal")
            self.complete_btn.configure(state="normal")
            self.camera_btn.configure(state="normal")
            self.delay_option_menu.configure(state="normal")
            self.start_cycle()
        # Jika 'No', tidak terjadi apa-apa, user bisa lanjut mengubah pilihan.

        # -------------------------------------------------------------------------
        # 3. LOGIKA INTI & KONTROL ALUR SISTEM
        # -------------------------------------------------------------------------
    def start_system(self):
        """
        - Dipicu oleh tombol 'Start'.
        - Memvalidasi input User & Barcode.
        - Membaca data dari file Excel.
        - Memulai proses logging ke database.
        - Memanggil start_cycle() untuk pertama kali.
        """
        if self.is_manual_selection_mode:
            if not self.manual_selected_slots:
                messagebox.showwarning("Peringatan", "Anda harus memilih setidaknya satu slot kabel.", parent=self)
                return

            confirm = messagebox.askyesno("Konfirmasi",
                                          f"Anda telah memilih {len(self.manual_selected_slots)} slot.\n\nLanjutkan proses?",
                                          parent=self)
            if confirm:
                self.is_manual_selection_mode = False
                self.start_btn.configure(text="Start", state="disabled")
                self.reset_btn.configure(state="normal")
                self.pause_btn.configure(state="normal")

                self.wire_info_entries[2].configure(state="normal")
                self.wire_info_entries[2].delete(0, "end")
                self.wire_info_entries[2].insert(0, str(len(self.manual_selected_slots)))
                self.wire_info_entries[2].configure(state="readonly")

                # Panggil start_cycle untuk memulai proses
                self.start_cycle()
            return # Hentikan fungsi di sini setelah konfirmasi

        user_name = self.wire_info_entries[0].get().strip()
        barcode = self.wire_info_entries[1].get().strip()

        if user_name != self.accumulation_user:
            # Jika user berganti, reset akumulasi ke 0
            print(f"User changed from '{self.accumulation_user}' to '{user_name}'. Resetting accumulation.")
            self.accumulation_total = 0
            self.accumulation_user = user_name

        # Selalu perbarui tampilan akumulasi setiap kali proses dimulai
        self.accumulation_entry.configure(state="normal")
        self.accumulation_entry.delete(0, "end")
        self.accumulation_entry.insert(0, str(self.accumulation_total))
        self.accumulation_entry.configure(state="readonly")

        if not user_name:
            self.update_status("Please enter User name")
            return

        if not barcode:
            self.update_status("Please enter Barcode ID")
            return

        # Tampilkan message box pilihan group
        choice = messagebox.askquestion("Select Group", 
                                      "Please select working group:\n\n'Yes' for Group A\n'No' for Group B", 
                                      parent=self)

        # Set group berdasarkan pilihan user
        self.current_group = 'A' if choice == 'yes' else 'B'
        self.update_status(f"Group {self.current_group} selected")

        try:
            # Load Excel data
            df = pd.read_excel(self.excel_file_path)#, sheet_name='From 2026-06-16 to 2026-06-19')

            df_u = pd.read_excel(self.excel_file_user)
            user_data = df_u[df_u['id_user'] == user_name]

            # Filter rows where efu_ser_no matches the barcode
            barcode_data = df[df['efu_ser_no'] == barcode]

            if user_data.empty:
                self.update_status("BarcodeUser ID not found in database")
                return

            if barcode_data.empty:
                self.update_status("Barcode not found in database")
                return

            # Get required data
            prod_no = barcode_data['prod_no'].iloc[0]
            group_no = barcode_data['ckt_grp_no'].iloc[0]
            plan_iss_date = barcode_data['plan_iss_date'].iloc[0]
            lot_no = barcode_data['lot_no'].iloc[0]
            wire_bndle_qty = int(barcode_data['wire_bndle_qty'].iloc[0])
            self.plan_work_order = int(barcode_data['plan_work_ord_qty'].iloc[0])
            plan_work_order = self.plan_work_order
            work_order_no = barcode_data['work_ord_no'].iloc[0]
            line_cd = barcode_data['line_cd'].iloc[0]
            loc_cd = barcode_data['loc_cd'].iloc[0]

            user_n = user_data['Nama'].iloc[0]

            # Menghitung wire_t
            lot_no_v = df[(df['prod_no'] == prod_no) & (df['ckt_grp_no'] == group_no) & 
                          (df['plan_iss_date'] == plan_iss_date) & (df['lot_no'] == lot_no)
                          & (df['work_ord_no'] == work_order_no)]
            wire_t = len(lot_no_v)

            self.max_selection_count = wire_t # Simpan batas maksimal pilihan

            barcode_info = f"System started with barcode: {barcode}\nBarcode: {barcode}\nProduct: {prod_no}\nGroup: {group_no}\nIssue Date: {plan_iss_date}\nLot: {lot_no}\nPlan Work Order: {plan_work_order}\nWork Order No : {work_order_no}\nLine cd: {line_cd}\nLoc cd: {loc_cd}\n User ID: {user_name}\n User Name: {user_n}"
            self.add_barcode_info(barcode_info)

            # Update current data field (wire_t)
            self.wire_info_entries[2].configure(state="normal")
            self.wire_info_entries[2].delete(0, "end")
            self.wire_info_entries[2].insert(0, "0")
            self.wire_info_entries[2].configure(state="readonly")

            # total work order
            self.wire_info_entries[3].configure(state="normal")
            self.wire_info_entries[3].delete(0, "end")
            self.wire_info_entries[3].insert(0, str(self.plan_work_order))
            self.wire_info_entries[3].configure(state="readonly")

            # Update remaining field (wire_bndle_qty)
            self.wire_info_entries[4].configure(state="normal")
            self.wire_info_entries[4].delete(0, "end")
            self.wire_info_entries[4].insert(0, str(wire_bndle_qty))
            self.wire_info_entries[4].configure(state="readonly")

            self.currently_entry.configure(state="normal")
            self.currently_entry.delete(0, "end")
            self.currently_entry.insert(0, f"0/{self.plan_work_order}")
            self.currently_entry.configure(state="readonly")

            # Lock input fields
            self.wire_info_entries[0].configure(state="readonly")
            self.wire_info_entries[1].configure(state="readonly")

            # Simpan data awal ke DB
            conn = get_db_connection()
            if conn is None: self.update_status("Koneksi Database Gagal!"); return
            try:
                cursor = conn.cursor()
                query = ("INSERT INTO logproses (user, id_barcode, data_current, remaining, waktu_mulai, total_work_order) "
                         "VALUES (%s, %s, %s, %s, %s, %s)")
                data_to_insert = (user_name, barcode, str(wire_t), wire_bndle_qty, datetime.datetime.now(), self.plan_work_order)
                cursor.execute(query, data_to_insert)
                conn.commit()
                self.db_process_id = cursor.lastrowid
            except Error as e:
                self.update_status(f"DB Error: {e}"); print(f"DB Error: {e}"); return
            finally:
                if conn.is_connected(): cursor.close(); conn.close()

            # Set the range based on wire_t (1 to wire_t)
            # self.current_start = 1
            # self.current_end = wire_t
            self.total_cycles = wire_bndle_qty
            self.current_cycle = 0
            self.counter_entry.configure(state="normal")
            self.counter_entry.delete(0, "end")
            self.counter_entry.insert(0, str(self.total_cycles))
            self.counter_entry.configure(state="readonly")
            self.wire_info_entries[0].configure(state="readonly")
            self.wire_info_entries[1].configure(state="readonly")
            # self.start_btn.configure(state="disabled")
            # self.start_cycle()
            self.is_manual_selection_mode = True
            self.manual_selected_slots.clear()
            self.update_status(f"Pilih {self.max_selection_count} slot untuk Grup {self.current_group} ({len(self.manual_selected_slots)}/{self.max_selection_count})")
            self.start_btn.configure(state="disabled")
            self.reset_btn.configure(state="disabled")
            self.pause_btn.configure(state="disabled")
            self.django_btn.configure(state="disabled")
            self.complete_btn.configure(state="disabled")
            self.camera_btn.configure(state="disabled")
            self.delay_option_menu.configure(state="disabled")

            if not self.is_status_blinking_selection:
                # Simpan warna asli sebelum memulai kedip
                self.original_status_fg_color_selection = self.status_entry.cget("fg_color")
                self.is_status_blinking_selection = True
                self._blink_status_for_selection()

        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            return

    def start_cycle(self):
        """
        - Memulai sebuah siklus kerja baru.
        - Mereset LED yang relevan menjadi hijau.
        - Mereset variabel deteksi untuk siklus ini.
        - Memperbarui status UI.
        """
        if self.current_cycle >= self.plan_work_order:
            # Sekarang blok akumulasi sudah tidak ada di sini
            self.counter_entry.configure(state="normal")
        if self.current_cycle >= self.plan_work_order:

            self.counter_entry.configure(state="normal")
            self.counter_entry.delete(0, "end")
            self.counter_entry.insert(0, "0")
            self.counter_entry.configure(state="readonly")

            self.reset_led_colors()
            self.stop_blinking()

            # Unlock input fields when done
            self.wire_info_entries[0].configure(state="normal")
            self.wire_info_entries[1].configure(state="normal")
            self.wire_info_entries[2].configure(state="readonly")
            self.wire_info_entries[3].configure(state="readonly")

            self.update_status("All cycles completed")
            if self.db_process_id: self.finish_db_process()
            self.reset_system()
            self.start_btn.configure(state="normal")
            if self.current_cycle >= self.total_cycles:
                if self.db_process_id:
                    self.finish_db_process()
            return

        self.cycle_active = True
        # self.current_cycle += 1
        self.update_status(f"Cycle {self.current_cycle}/{self.total_cycles} (Group {self.current_group})")

        # Reset LED colors for the new cycle
        self.reset_led_colors()

        # Light up LEDs based on selected group
        for label in self.manual_selected_slots:
            led = self.get_led_by_label(label)
            if led:
                self._set_led_color(led, label, "green")

        self.detected_objects = set()
        self.all_wires_taken = False

        if not self.timer_running:
            self.start_timer()

        self.start_blinking()

        # Update counter
        remaining = max(0, self.total_cycles - self.current_cycle)
        self.counter_entry.configure(state="normal")
        self.counter_entry.delete(0, "end")
        self.counter_entry.insert(0, str(remaining))
        self.counter_entry.configure(state="readonly")

        self.system_started = True
        self.home_btn.configure(state="disabled")
        self.info_btn.configure(state="disabled")
        self.start_btn.configure(state="disabled")

    def check_cycle_completion(self):
        """
        - Mengecek apakah semua LED hijau sudah berubah menjadi merah.
        - Jika ya, menyelesaikan siklus dan memanggil start_cycle() lagi.
        """
        if not self.cycle_active or self.all_wires_taken:
            return

        all_detected = True
        for label in self.manual_selected_slots:
            led = self.get_led_by_label(label)
            if led and led.cget("fg_color") != "red":
                all_detected = False
                break

        if all_detected:
            self.all_wires_taken = True
            self.current_cycle += 1
            self.currently_entry.configure(state="normal")
            self.currently_entry.delete(0, "end")
            self.currently_entry.insert(0, f"{self.current_cycle}/{self.plan_work_order}")
            self.currently_entry.configure(state="readonly")
            remaining = max(0, self.total_cycles - self.current_cycle)
            self.counter_entry.configure(state="normal")
            self.counter_entry.delete(0, "end")
            self.counter_entry.insert(0, str(remaining))
            self.counter_entry.configure(state="readonly")
            self.add_history_message(f"✅ Siklus {self.current_cycle} selesai - {datetime.datetime.now().strftime('%H:%M:%S')}")
            self.after(1000, self.start_cycle)

    def pause_system(self):
        """Menjeda atau melanjutkan timer dan proses."""
        if self.timer_running:
            if not self.timer_paused:
                self.timer_paused = True
                self.pause_start = datetime.datetime.now()
                self.pause_btn.configure(text="Resume")
                self.start_btn.configure(state="disabled")
                self.stop_blinking()
                self.update_status("Paused system")
                if self.original_status_fg_color is None:
                    self.original_status_fg_color = self.status_entry.cget("fg_color")

                # Mulai kedipkan background status entry
                self.is_status_blinking = True
                self._blink_status_entry()
            else:
                self.timer_paused = False
                pause_end = datetime.datetime.now()
                self.paused_time += (pause_end - self.pause_start).total_seconds()
                self.pause_btn.configure(text="Pause")
                self.start_btn.configure(state="disabled")
                self.is_status_blinking = False

                # Kembalikan warna background ke warna aslinya
                if self.original_status_fg_color:
                    self.status_entry.configure(fg_color=self.original_status_fg_color)
                    self.original_status_fg_color = None # Reset agar bisa disimpan lagi nanti
                self.start_blinking()
                self.update_status(f"Cycle {self.current_cycle}/{self.total_cycles}")
                self.update_timer()

    def update_text_colors(self):
        """Updates text colors of various widgets based on the current theme."""
        # Hanya update jika kita berada di home layout
        if not hasattr(self, 'title_label') or not self.title_label.winfo_exists():
            return

        text_color = "black" if self.is_dark_mode else "black"
        widgets = [
            self.title_label,
            self.control_frame.winfo_children()[0],
            self.wire_info_label,
            *self.wire_info_labels,
            self.connect_info_label,
            self.status_label,
            self.history_label,
            self.barcode_info_label,
            self.process_label,
            *[lbl for lbl, _ in self.process_labels],
            self.time_label
        ]

        # Filter hanya widget yang masih ada
        widgets = [w for w in widgets if hasattr(w, 'winfo_exists') and w.winfo_exists()]

        for widget in widgets:
            try:
                widget.configure(text_color=text_color)
            except Exception as e:
                print(f"Error updating widget color: {e}")

        if hasattr(self, 'grup_a_label') and hasattr(self, 'grup_b_label'):
            if self.grup_a_label.winfo_exists() and self.grup_b_label.winfo_exists():
                self.grup_a_label.configure(text_color=text_color)
                self.grup_b_label.configure(text_color=text_color)

    def reset_system(self):
        """
        - Menghentikan semua proses.
        - Menyimpan data akhir ke database (memanggil finish_db_process).
        - Mereset semua variabel, timer, dan UI ke kondisi awal.
        """
        self._stop_all_blinking()
        if hasattr(self, 'db_process_id') and self.db_process_id:
            self.finish_db_process()

        """Resets the entire system to its initial state."""
        self.update_status("Not Connected")
        self.stop_blinking()
        self.blinking_tags.clear()
        self.reset_timer()
        self.reset_led_colors()

        # Clear history and barcode info
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", "end")
        self.history_text.configure(state="disabled")

        self.barcode_info_text.configure(state="normal")
        self.barcode_info_text.delete("1.0", "end")
        self.barcode_info_text.configure(state="disabled")

        # Reset all input fields to default
        self.wire_info_entries[0].configure(state="normal")  # User field

        self.wire_info_entries[1].configure(state="normal")  # Barcode field
        self.wire_info_entries[1].delete(0, "end")

        self.wire_info_entries[2].configure(state="normal")  # Current data field
        self.wire_info_entries[2].delete(0, "end")
        self.wire_info_entries[2].configure(state="readonly")

        self.wire_info_entries[3].configure(state="normal")
        self.wire_info_entries[3].delete(0, "end")
        self.wire_info_entries[3].configure(state="readonly")

        self.wire_info_entries[4].configure(state="normal")  # Remaining field
        self.wire_info_entries[4].delete(0, "end")
        self.wire_info_entries[4].configure(state="readonly")

        # Reset counter
        self.counter_entry.configure(state="normal")
        self.counter_entry.delete(0, "end")
        self.counter_entry.configure(state="readonly")

        self.currently_entry.configure(state="normal")
        self.currently_entry.delete(0, "end")
        self.currently_entry.configure(state="readonly")

        # Reset buttons
        self.start_btn.configure(state="normal")
        self.pause_btn.configure(text="Pause", state="normal")
        self.home_btn.configure(state="normal")
        self.info_btn.configure(state="normal")

        # Reset system state
        self.system_started = False
        self.cycle_active = False
        self.is_manual_selection_mode = False # <-- SANGAT PENTING
        self.manual_selected_slots.clear()  
        self.current_cycle = 0
        self.total_cycles = 0
        self.db_process_id = None
        self.all_wires_taken = False
        self.last_logged_error_slot = None
        self.plan_work_order = 0

        for button in [self.start_btn, self.reset_btn, self.pause_btn, self.home_btn, 
                       self.info_btn, self.mode_btn, self.django_btn, 
                       self.complete_btn, self.camera_btn, self.delay_option_menu]:
            button.configure(state="normal")
        self.pause_btn.configure(text="Pause")

    def complete_system(self):
        self._stop_all_blinking()
        """Dipicu tombol 'Complete', meminta konfirmasi lalu memanggil reset_system."""
        if not self.system_started:
            self.update_status("System not started")
            return

        confirm = messagebox.askyesno("Complete System", 
                                    "Are you sure you want to complete this process?",
                                    parent=self)
        if confirm:
            # Cukup panggil reset_system. Fungsi ini akan menangani penyimpanan DB dan reset UI.
            self.update_status("Proses selesai, sistem direset.")
            self.reset_system()

    # -------------------------------------------------------------------------
    # 4. MANAJEMEN KAMERA & PEMROSESAN GAMBAR (COMPUTER VISION)
    # -------------------------------------------------------------------------
    def toggle_camera(self):
        """Menyalakan atau mematikan stream kamera dan thread terkait."""
        if self.camera_on:

            self.stop_event.set()

            if self.camera_thread and self.camera_thread.is_alive():
                self.camera_thread.join(timeout=1.0)
            """
            if self.camera_thread.is_alive():  # Jika thread masih hidup setelah timeout
                self.camera_thread.terminate()  # atau handle khusus
            if self.camera_thread is not None:
                self.camera_thread.join()
            if self.detection_thread is not None:
                self.detection_thread.join()
            if self.serial_thread is not None:
                self.serial_thread.join()
            
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            self.camera_on = False
            self.camera_btn.configure(text="Start Camera")
            self.camera_label.pack_forget()
            self.camera_placeholder.pack(expand=True)
            """
            self.camera_on = False
            self.camera_btn.configure(text="Start Camera")
            self.camera_label.pack_forget()
            self.camera_placeholder.pack(expand=True)

            if self.cap: #is not None and self.cap.isOpened():
                self.cap.release()
                self.cap = None
                print("Kamera dimatikan")

            self.update_status("Camera OFF")
        else:
            self.stop_event.clear()
            self.camera_on = True
            self.update_status("Wait, camera loading...")
            self.update_idletasks() # Update the GUI to show the loading message
            init_thread = threading.Thread(target=self.init_camera, daemon=True)
            init_thread.start()

    def set_camera_state(self, on):
        """Update camera state from main thread."""
        self.camera_on = on
        self.camera_btn.configure(text="Stop Camera" if on else "Start Camera")
        self.camera_placeholder.pack_forget() if on else self.camera_label.pack_forget()
        self.camera_label.pack(expand=True) if on else self.camera_placeholder.pack(expand=True)
        self.update_status("Camera ON" if on else "Camera OFF")

    def init_camera(self):
        """Inisialisasi objek VideoCapture di thread terpisah."""
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720) #640
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #480
                self.cap.set(cv2.CAP_PROP_FPS, 120)#dari 120 FPS

                # Update GUI dari main thread
                self.after(0, lambda: [
                    self.set_camera_state(True),
                    self.start_camera_threads(),
                    self.start_serial_thread()
                ])
            else:
                self.after(0, lambda: self.update_status("Camera Error"))
        except Exception as e:
            self.after(0, lambda: self.update_status(f"Camera Error: {str(e)}"))

    def start_camera_threads(self):
        """Memulai thread untuk capture_frames dan process_frames."""
        self.camera_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.camera_thread.start()

        self.detection_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.detection_thread.start()

        self.update_camera_display()

    def capture_frames(self):
        """Thread worker: Terus menerus mengambil frame dari kamera."""
        while not self.stop_event.is_set() and self.camera_on:
            ret, frame = self.cap.read()
            if ret:

                # Terapkan koreksi distorsi fisheye pada frame
                frame = cv2.flip(frame, -1)
                h, w = frame.shape[:2]
                fx_val = 0.89#1.17
                k1_val = -0.34#-0.52  # Adjust this value for barrel distortion
                k2_val = 0.49#0.16  # Adjust this value for barrel distortion    

                fx = w  *max(fx_val, 0.1)  # Ensure fx is not too small
                fy = fx  # Keep aspect ratio
                cx = w / 2.00
                cy = h / 2.00

                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
                D = np.array([k1_val, k2_val, 0.0, 0.0], dtype=np.float64)

                try:

                    # Hitung matriks koreksi
                    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                        K, D, (w, h), np.eye(3), balance=0.5
                    )
                    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
                    )

                    # Terapkan koreksi pada frame realtime
                    frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

                    if self.frame_queue.empty():
                        self.frame_queue.put(frame)
                except queue.Full:
                    pass

    def process_frames(self):
        """Thread worker: Memproses frame untuk deteksi ArUco/YOLO."""
        while not self.stop_event.is_set() and self.camera_on:

            try:
                frame = self.frame_queue.get(timeout=0.01)

                # Process frame
                processed_frame = frame.copy()

                # self.classify_pipe(0, 0, 0, processed_frame, 0, 0, St_Aruco, self.data_)

                St_Aruco = 0  # initial aruco status

                # Process with YOLO
                results = self.model.predict(frame)
                yolo_boxes = []  # list of (x1, y1, x2, y2) hasil deteksi YOLO (koordinat piksel)
                if len(results):
                    processed_frame = results[0].plot()
                    if results[0].boxes is not None and len(results[0].boxes) > 0:
                        for box in results[0].boxes.xyxy.cpu().numpy():
                            bx1, by1, bx2, by2 = box[:4]
                            yolo_boxes.append((float(bx1), float(by1), float(bx2), float(by2)))

                # Process with ArUco
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # corners, ids, rejected = aruco.detectMarkers(gray, self.ARUCO_DICT)
                corners, ids, rejected = self.ARUCO_DETECTOR.detectMarkers(gray)

                if ids is not None:
                    St_Aruco = 1  # ArUco detected
                    self.classify_pipe_sensor(self.x_s, self.y_s, self.z_s, processed_frame, 0, 0, St_Aruco)

                    for corner, marker_id in zip(corners, ids.flatten()):

                        half_size = self.MARKER_SIZE / 2.0
                        obj_points = np.array([
                            [-half_size,  half_size, 0],
                            [ half_size,  half_size, 0],
                            [ half_size, -half_size, 0],
                            [-half_size, -half_size, 0]
                        ], dtype=np.float32)

                        # Hitung pose rvec dan tvec menggunakan solvePnP
                        success, rvec, tvec = cv2.solvePnP(
                            obj_points, corner, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
                        )

                        if success:
                            aruco.drawDetectedMarkers(processed_frame, [corner])
                            cv2.drawFrameAxes(processed_frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 2)

                        aruco.drawDetectedMarkers(processed_frame, [corner])
                        cv2.drawFrameAxes(processed_frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 2)

                        x_c = int(np.mean(corner[0][:, 0]))
                        y_c = int(np.mean(corner[0][:, 1]))
                        distance = np.linalg.norm(tvec)

                        self.classify_pipe(x_c, y_c, distance, processed_frame, x_c, y_c, St_Aruco, yolo_boxes)

                        text = f"x:{x_c} y:{y_c} Dist:{distance:.2f} cm"
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        text_x = int(corner[0][0][0]) - text_size[0] - 10
                        text_y = int(corner[0][0][1]) - 10
                        text_x = max(text_x, 0)
                        cv2.putText(processed_frame, text, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw slot rectangles
                for label, coord in self.slot_data.items():
                    cv2.rectangle(processed_frame, (coord["x1"], coord["y1"]), (coord["x2"], coord["y2"]),(216, 235, 255), 3)

                # Put processed frame in result queue
                if self.result_queue.empty():
                    self.result_queue.put(processed_frame)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

    def validate_remaining(self, value):
        """Validates that the 'Remaining' input is a number between 1 and 999."""
        if value == "":
            return True
        try:
            num = int(value)
            return 1 <= num <= 999
        except ValueError:
            return False

    def update_camera_display(self):
        """Mengambil frame dari result_queue dan menampilkannya di UI."""
        if self.camera_on: #and not self.stop_event.is_set():
            try:
                frame = self.result_queue.get_nowait()

                # Convert to RGB and display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=image)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)

            except queue.Empty:
                pass

            self.after(1, self.update_camera_display)

    def _is_point_in_yolo_box(self, x_c, y_c, yolo_boxes):
        """
        Mengecek apakah titik (x_c, y_c) berada di dalam salah satu
        bounding box hasil deteksi YOLO. Dipakai untuk logika AND
        antara ArUco dan YOLO.
        """
        if not yolo_boxes:
            return False
        for (bx1, by1, bx2, by2) in yolo_boxes:
            if bx1 <= x_c <= bx2 and by1 <= y_c <= by2:
                return True
        return False

    def classify_pipe(self, x, y, z, img, x_c, y_c, St_Aruco_, yolo_boxes=None):
        """
        - Menerima koordinat dari deteksi kamera.
        - Menentukan slot mana yang terdeteksi.
        - Logika AND: marker ArUco harus berada di slot tertentu DAN
          titik tersebut harus berada di dalam salah satu bounding box hasil YOLO,
          baru dianggap terkonfirmasi.
        - Menangani logika benar/salah ambil berdasarkan warna LED.
        - Memanggil update_led_color() jika pengambilan benar.

        CATATAN THREAD-SAFETY:
        Fungsi ini dipanggil dari detection_thread (background thread), BUKAN main thread.
        Karena itu:
        - Status warna LED dibaca dari self.led_color_cache (dict biasa, aman lintas-thread),
          BUKAN dengan memanggil widget.cget() langsung.
        - Semua aksi yang menyentuh widget Tkinter (update_led_color, add_history_message,
          log_error_detection_to_db) dijadwalkan lewat self.after(0, ...) agar benar-benar
          dieksekusi di main thread.
        - cv2.putText pada `img` tetap aman dipanggil di sini karena hanya menulis ke
          array numpy lokal, bukan ke widget GUI.
        """
        if yolo_boxes is None:
            yolo_boxes = []

        # Cek apakah titik tengah marker ArUco (x_c, y_c) berada di dalam
        # salah satu bounding box hasil deteksi YOLO.
        yolo_confirmed = self._is_point_in_yolo_box(x_c, y_c, yolo_boxes)

        for label, coord in self.slot_data.items():

            if St_Aruco_ == 1:
                condition = (coord["x1"] <= x <= coord["x2"] and 
                            coord["y1"] <= y <= coord["y2"] and
                            coord["z_min"] <= z <= coord["z_max"])
            else:
                condition = (coord["x1"] <= x <= coord["x2"] and 
                            coord["y1"] <= y <= coord["y2"])

            # LOGIKA AND: slot ArUco cocok DAN dikonfirmasi oleh bounding box YOLO
            condition = condition and yolo_confirmed

            if condition:
                # Tampilkan label di kamera (aman: hanya menulis ke array gambar lokal)
                cv2.putText(img, label, (x_c, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Baca status warna LED dari cache (thread-safe), bukan dari widget langsung
                current_color = self.led_color_cache.get(label, "gray")

                # Tambahkan teks status di kamera
                if current_color == "red":
                    cv2.putText(img, "SUDAH DIAMBIL", (x_c, y_c+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                elif current_color == "gray":
                    cv2.putText(img, "TIDAK DIAMBIL/TIDAK TERSEDIA", (x_c, y_c+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

                # Jika sistem aktif dan tidak pause
                if self.system_started and not self.timer_paused and self.cycle_active:
                    if label not in self.detected_objects:
                        self.pic_time = datetime.datetime.now()
                        # self.add_history_message(f"waktu ambil{self.pic_time}")
                        self.detected_objects.add(label)
                        # Sentuhan ke widget LED -> jadwalkan ke main thread
                        self.after(0, lambda l=label: self.update_led_color(l))

                        # print("Masuk state dtc")
                    else:
                        if current_color == "red":
                            print("Masuk state dtc_salah")
                            now_time = datetime.datetime.now()
                            selisih = now_time - self.pic_time
                            selisih_s = selisih.total_seconds()
                            self.last_logged_error_slot = label
                            if selisih_s > self.error_delay_threshold:
                                self.play_sound('buzzer.mp3')
                                warning_msg = f"⚠️ Peringatan: Objek terdeteksi di {label} (kabel sudah diambil) - {datetime.datetime.now().strftime('%H:%M:%S')}"
                                self.pic_time = datetime.datetime.now()
                                self.after(0, lambda m=warning_msg: self.add_history_message(m, error_type='red_slot'))
                                self.after(0, lambda l=label: self.log_error_detection_to_db(l))
                        elif current_color == "gray":
                            if label != self.last_logged_error_slot:
                                warning_msg = f"⚠️ Peringatan: Objek terdeteksi di {label} (area seharusnya kosong) - {datetime.datetime.now().strftime('%H:%M:%S')}"
                                self.last_logged_error_slot = label
                                self.play_sound('buzzer.mp3')
                                self.after(0, lambda m=warning_msg: self.add_history_message(m, error_type='gray_slot'))
                                self.after(0, lambda l=label: self.log_error_detection_to_db(l))

                return True
        return False

    # -------------------------------------------------------------------------
    # 5. KOMUNIKASI HARDWARE (SERIAL)
    # -------------------------------------------------------------------------
    def start_serial_thread(self):
        """Memulai thread untuk membaca data dari port serial."""
        self.serial_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.serial_thread.start()
        self.update_serial() 

    def update_serial(self):
        """Thread worker: Terus menerus membaca data dari serial dan menerjemahkannya."""
        self.x_s = 0
        self.y_s = 0
        self.z_s = 0

        # ==================== BAGIAN YANG DIGANTI / DIPERBAIKI ====================
        data = None

        # Cek apakah port serial terkonfigurasi dan dalam keadaan terbuka
        if hasattr(self, 'ser') and self.ser is not None and self.ser.is_open:
            try:
                if self.ser.in_waiting > 0:
                    data = self.ser.readline().decode('utf-8', errors='ignore').strip()
            except Exception as e:
                print(f"Error membaca serial: {e}")

        # Jika tidak ada data yang terbaca dari serial, hentikan proses parsing dan lanjut ke loop berikutnya
        if not data:
            self.after(10, self.update_serial)
            return
        # ==========================================================================

        # Logika Parsing Data Serial Kamu (Tetap Sama)
        try:
            self.data_ = data.split(",")
            if len(self.data_) >= 2:
                val1 = int(self.data_[0])
                val2 = int(self.data_[1])

                # if val1 <= 51 and val1 >= 43: # untuk sensor sebelah kiri
                if 55 <= val1 <= 64: # untuk sensor sebelah kanan
                    print("1A")
                    self.x_s = 260
                    self.y_s = 350
                    self.z_s = 190
                # elif val1 <= 41 and val1 >= 32: # untuk sensor sebelah kiri
                elif 67 <= val1 <= 75: # untuk sensor sebelah kanan
                    print("2A")
                    self.x_s = 200
                    self.y_s = 350
                    self.z_s = 190  
                # elif val1 <= 99 and val1 >= 91: # untuk sensor sebelah kiri
                elif 11 <= val1 <= 19: # untuk sensor sebelah kanan
                    print("1B")
                    self.x_s = 515
                    self.y_s = 350
                    self.z_s = 200                
                # elif val1 <= 89 and val1 >= 80: # untuk sensor sebelah kiri
                elif 21 <= val1 <= 31: # untuk sensor sebelah kanan
                    print("2B")
                    self.x_s = 460
                    self.y_s = 350
                    self.z_s = 200                

                # val2 belum diperbaharui nilai sensornya
                if 37 <= val2 <= 44:
                    print("2A")
                    self.x_s = 260
                    self.y_s = 250
                    self.z_s = 150
                elif 24 <= val2 <= 32:
                    print("6A")
                    self.x_s = 200
                    self.y_s = 250
                    self.z_s = 160

                print(f"Data: {data}")
        except (ValueError, IndexError) as e:
            print(f"Format data serial tidak valid ({data}): {e}")

        self.after(10, self.update_serial)

    def classify_pipe_sensor(self, x, y, z, img, x_c, y_c, St_Aruco_):
        """
        - Menerima koordinat dari hasil terjemahan data sensor.
        - Menangani logika benar/salah ambil (mirip dengan classify_pipe).
        """
        for label, coord in self.slot_data.items():

            if St_Aruco_ == 1:
                condition = (coord["x1"] <= x <= coord["x2"] and 
                            coord["y1"] <= y <= coord["y2"] and
                            coord["z_min"] <= z <= coord["z_max"])
            else:
                condition = (coord["x1"] <= x <= coord["x2"] and 
                            coord["y1"] <= y <= coord["y2"])

            if condition:

                # Baca status warna LED dari cache (thread-safe), bukan dari widget langsung
                current_color = self.led_color_cache.get(label, "gray")

                # Jika sistem aktif dan tidak pause
                if self.system_started and not self.timer_paused and self.cycle_active:
                    if label not in self.detected_objects:
                        self.pic_time = datetime.datetime.now()
                        # self.add_history_message(f"waktu ambil{self.pic_time}")
                        self.detected_objects.add(label)
                        # Sentuhan ke widget LED -> jadwalkan ke main thread
                        self.after(0, lambda l=label: self.update_led_color(l))
                    else:
                        if current_color == "red":
                            print("Masuk state dtc_salah")
                            now_time = datetime.datetime.now()
                            selisih = now_time - self.pic_time
                            selisih_s = selisih.total_seconds()
                            if selisih_s > self.error_delay_threshold:
                                self.play_sound('buzzer.mp3')
                                warning_msg = f"⚠️ Peringatan: Objek terdeteksi di {label} (kabel sudah diambil) - {datetime.datetime.now().strftime('%H:%M:%S')}"
                                self.pic_time = datetime.datetime.now()
                                self.after(0, lambda m=warning_msg: self.add_history_message(m, error_type='red_slot'))
                                self.after(0, lambda l=label: self.log_error_detection_to_db(l))
                        elif current_color == "gray":
                            if label != self.last_logged_error_slot:
                                warning_msg = f"⚠️ Peringatan: Objek terdeteksi di {label} (area seharusnya kosong) - {datetime.datetime.now().strftime('%H:%M:%S')}"
                                self.last_logged_error_slot = label
                                self.play_sound('buzzer.mp3')
                                self.after(0, lambda m=warning_msg: self.add_history_message(m, error_type='gray_slot'))
                                self.after(0, lambda l=label: self.log_error_detection_to_db(l))

                return True
        return False

    # -------------------------------------------------------------------------
    # 6. MANAJEMEN DATABASE & PROSES EKSTERNAL (DJANGO)
    # -------------------------------------------------------------------------
    def finish_db_process(self):
        """Menyimpan data final (waktu selesai, jumlah siklus) ke database."""
        if not self.db_process_id: return
        self.accumulation_total += self.current_cycle

        # Perbarui juga tampilan di GUI agar langsung terlihat
        self.accumulation_entry.configure(state="normal") 
        self.accumulation_entry.delete(0, "end")
        self.accumulation_entry.insert(0, str(self.accumulation_total))
        self.accumulation_entry.configure(state="readonly")
        conn = get_db_connection()
        if not conn: print("DB connection failed."); return

        try:
            cursor = conn.cursor()
            query = ("UPDATE logproses SET waktu_selesai=%s, timer=%s, durasi_detik=%s, sisa_bundle=%s, work_order=%s WHERE id=%s")

            work_order = self.current_cycle

            # HITUNG SISA SIKLUS DI SINI
            sisa_bundle = max(0, self.total_cycles - work_order)

            data_to_update = (datetime.datetime.now(), self.timer_entry.get(), int(self.elapsed_time), sisa_bundle, work_order, self.db_process_id)
            cursor.execute(query, data_to_update)
            conn.commit()
            print(f"Proses DB ID {self.db_process_id} difinalisasi dengan counter: {work_order}.")
        except Error as e:
            print(f"DB Update Error: {e}")
        finally:
            if conn.is_connected(): cursor.close(); conn.close()
            self.db_process_id = None

    def log_error_detection_to_db(self, posisi):
        """
        Mencatat deteksi yang salah ke tabel log deteksi di database.
        PENTING: Fungsi ini HARUS dipanggil dari main thread (mis. lewat self.after(0, ...)),
        karena di sini kita membaca nilai dari widget Tkinter (wire_info_entries, timer_entry).
        Setelah data dikumpulkan, proses insert ke database (operasi I/O yang bisa lambat/blocking)
        didelegasikan ke db_worker_thread lewat db_log_queue, supaya tidak memblok
        main thread ataupun detection_thread.
        """
        if not self.db_process_id: return
        data_to_insert = (
            self.db_process_id,
            self.wire_info_entries[0].get(),
            self.wire_info_entries[1].get(),
            self.wire_info_entries[2].get(),
            posisi,"salah",
            datetime.datetime.now(),
            self.timer_entry.get()
        )
        try:
            self.db_log_queue.put_nowait(data_to_insert)
        except queue.Full:
            print("DB log queue penuh, log error dilewati.")

    def _db_worker_loop(self):
        """
        Thread worker khusus untuk menulis log error ke database.
        Berjalan terus selama aplikasi aktif, mengambil data dari db_log_queue
        dan melakukan insert MySQL secara sinkron DI THREAD INI SAJA,
        sehingga koneksi/insert yang lambat tidak akan membekukan kamera atau GUI.
        """
        while not self.stop_event.is_set():
            try:
                data_to_insert = self.db_log_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._insert_log_error_to_db(data_to_insert)

    def _insert_log_error_to_db(self, data_to_insert):
        """Melakukan insert aktual ke tabel logdeteksi. Hanya dipanggil dari db_worker_thread."""
        conn = get_db_connection()
        if not conn: return
        try:
            cursor = conn.cursor()
            query = ("INSERT INTO logdeteksi (id_proses, username, id_barcode, data_current, posisi_terdeteksi, kondisi, time_actual, timer) "
                     "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)")
            cursor.execute(query, data_to_insert)
            conn.commit()
        except Error as e:
            print(f"DB Log Detection Error: {e}")
        finally:
            if conn.is_connected(): cursor.close(); conn.close()

    def run_django_server(self):
        """Menjalankan server Django sebagai proses terpisah."""
        print("Mencoba menghubungkan ke database untuk memeriksa status MariaDB...")
        conn = get_db_connection()
        if conn is None:
            self.update_status("Database tidak aktif!")
            messagebox.showerror("Koneksi Gagal", "Tidak bisa terhubung ke database. Pastikan MariaDB sudah berjalan.")
            return
        else:
            conn.close() 
            print("Koneksi database berhasil. Melanjutkan pengecekan port...")

        if self.is_port_in_use(8000):
            self.update_status("Server sudah berjalan (Port 8000 digunakan).")
            print("Operasi dibatalkan: Port 8000 sudah digunakan oleh server lain.")
            messagebox.showwarning("Peringatan", "Server terdeteksi sudah berjalan dari sesi sebelumnya. Tombol kini berfungsi sebagai 'Stop Server'.")
            self.django_btn.configure(text="Stop Server", command=self.stop_django_server)
            return

        try:
            # /home/berdikari/Documents/wireselection/Djangoberdikari/venv/bin
            python_executable = "/home/berdikari/HandDetection/venv/bin/python"
            manage_py_path = "/home/berdikari/HandDetection/Djangoberdikari/dashboard_project/manage.py"
            command = [python_executable, manage_py_path, "runserver", "0.0.0.0:8000"]

            # Pengecekan OS: Gunakan bendera Windows HANYA jika di Windows
            if IS_WINDOWS:
                self.django_process = subprocess.Popen(command, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                self.django_process = subprocess.Popen(command) # Linux tidak butuh bendera ini

            self.update_status("Server sedang berjalan...")
            print("Server dimulai di latar belakang.")
            self.django_btn.configure(text="Stop Server", command=self.stop_django_server)

        except Exception as e:
            self.update_status(f"Gagal memulai server: {e}")
            print(f"Error saat mencoba memulai server: {e}")

    def stop_django_server(self):
        """Menghentikan proses server Django."""
        pid_to_kill = None

        if self.django_process and self.django_process.poll() is None:
            pid_to_kill = self.django_process.pid
            print(f"Menggunakan PID tersimpan: {pid_to_kill}")
        else:
            print("Tidak ada PID tersimpan. Mencari proses yang menggunakan port 8000...")
            try:
                if IS_WINDOWS:
                    result = subprocess.run(
                        ['netstat', '-aon'], capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW
                    )
                    for line in result.stdout.splitlines():
                        if "8000" in line and "LISTENING" in line:
                            match = re.search(r'(\d+)$', line.strip())
                            if match:
                                pid_to_kill = match.group(1)
                                print(f"Ditemukan proses dengan PID {pid_to_kill} di port 8000.")
                                break
                else:
                    # Perintah khusus Linux (lsof) untuk menemukan PID di port 8000
                    result = subprocess.run(['lsof', '-t', '-i:8000'], capture_output=True, text=True)
                    if result.stdout:
                        pid_to_kill = result.stdout.strip().split('\n')[0]
                        print(f"Ditemukan proses dengan PID {pid_to_kill} di port 8000.")
            except Exception as e:
                print(f"Gagal mencari PID di port 8000: {e}")
                self.update_status("Gagal mencari server.")
                return

        if pid_to_kill:
            print(f"Mencoba menghentikan server Django dengan PID: {pid_to_kill}")
            try:
                if IS_WINDOWS:
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid_to_kill)], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    # Perintah khusus Linux (kill) untuk menghentikan proses
                    subprocess.run(['kill', '-9', str(pid_to_kill)], check=True)

                self.update_status("Server berhasil dimatikan.")
                print("Server telah dihentikan.")
            except Exception as e:
                print(f"Gagal menghentikan proses dengan PID {pid_to_kill}: {e}")
                self.update_status(f"Gagal mematikan server.")
            finally:
                self.django_process = None
                self.django_btn.configure(text="Run Server", command=self.run_django_server)
        else:
            print("Tidak ada proses server yang aktif untuk dihentikan.")
            self.update_status("Tidak ada server yang berjalan.")
            self.django_process = None
            self.django_btn.configure(text="Run Server", command=self.run_django_server)

    def is_port_in_use(self, port: int):
        """Utility untuk memeriksa apakah port jaringan sedang digunakan."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Atur timeout agar tidak menunggu terlalu lama
            s.settimeout(0.5)
            # connect_ex() akan mengembalikan 0 jika koneksi berhasil (artinya port TERISI)
            # dan akan mengembalikan kode error jika koneksi gagal (artinya port KOSONG).
            try:
                if s.connect_ex(('127.0.0.1', port)) == 0:
                    return True # Port sedang digunakan
            except Exception:
                return False # Terjadi error lain, anggap port tidak digunakan
        return False # Port bebas

    # -------------------------------------------------------------------------
    # 7. FUNGSI UTILITAS & PEMBANTU (HELPERS)
    # -------------------------------------------------------------------------
    def play_sound(self, sound_file):
        """
        Memainkan file suara di thread terpisah menggunakan mpg123.

        REVISI: diganti dari library 'playsound' ke subprocess 'mpg123'
        karena playsound sering gagal diam-diam di Linux (butuh gstreamer/gi
        yang tidak selalu terinstall). mpg123 jauh lebih reliable di Ubuntu/Linux.

        Cara install mpg123 (cukup sekali):
            sudo apt install mpg123

        Path file suara dibuat absolut (relatif ke lokasi file .py ini)
        agar tidak error meski program dijalankan dari direktori berbeda.
        """
        try:
            base = os.path.dirname(os.path.abspath(__file__))
            path_to_sound = os.path.join(base, sound_file)
            if not os.path.exists(path_to_sound):
                print(f"[play_sound] File tidak ditemukan: {path_to_sound}")
                return
            threading.Thread(
                target=lambda: __import__('subprocess').run(
                    ["mpg123", "-q", path_to_sound],
                    stdout=__import__('subprocess').DEVNULL,
                    stderr=__import__('subprocess').DEVNULL
                ),
                daemon=True
            ).start()
        except Exception as e:
            print(f"[play_sound] Gagal memutar suara: {e}")

    def update_status(self, text):
        """
        Memperbarui teks di kotak status.
        Dijadwalkan lewat self.after(0, ...) agar aman dipanggil dari thread mana pun
        (mis. dari detection_thread), karena widget Tkinter h9anya boleh diubah dari main thread.
        """
        def _do_update():
            self.status_entry.configure(state="normal")
            self.status_entry.delete(0, "end")
            self.status_entry.insert(0, text)
            self.status_entry.configure(state="readonly")
        self.after(0, _do_update)

    def add_history_message(self, message, delay=0, error_type=None):
        """
        Menambahkan pesan ke kotak histori.
        Selalu dijadwalkan lewat self.after(...) (minimal delay 0) supaya aman
        dipanggil dari thread mana pun, termasuk detection_thread.
        """
        def add_message():
            # ==> SEMUA LOGIKA DI BAWAH INI SEKARANG BERADA DI DALAM add_message <==
            self.history_text.configure(state="normal")
            start_index = self.history_text.index("end-1c")
            self.history_text.insert("end", message + "\n")

            # Jika ini adalah pesan error, tambahkan tag dengan warna spesifik
            if error_type:
                # ==> BAGIAN YANG HILANG SUDAH SAYA TAMBAHKAN KEMBALI <==
                end_index = self.history_text.index("end-2c")
                tag_name = f"error_{self.error_tag_counter}"
                self.error_tag_counter += 1
                self.history_text.tag_add(tag_name, start_index, end_index)

                # Tentukan warna highlight berdasarkan tipe error
                if error_type == 'red_slot':
                    highlight_color = "#FFA500"  # Oranye untuk slot merah
                elif error_type == 'gray_slot':
                    highlight_color = "#DB1514"  # Merah untuk slot abu-abu
                else:
                    highlight_color = "#B6580C"  # Warna default jika tipe tidak spesifik

                # Highlight border history box
                self.history_box.configure(border_color="red", border_width=3)
                self.after(3000, self.reset_history_box_highlight)

                # Konfigurasi tag dan blinking
                self.history_text.tag_config(tag_name, foreground="black", background=highlight_color)
                self.blinking_tags[tag_name] = {"state": "active", "color": highlight_color}

                if not self.is_blinking_active:
                    self.is_blinking_active = True
                    self._blink_history_errors()

            self.history_text.configure(state="disabled")
            self.history_text.see("end")

        # Selalu jadwalkan lewat self.after, baik delay 0 ataupun lebih,
        # supaya aman dipanggil dari thread background sekalipun.
        self.after(max(delay, 0), add_message)

    def add_barcode_info(self, info):
        """Menampilkan informasi barcode di UI."""
        self.barcode_info_text.configure(state="normal")
        self.barcode_info_text.delete("1.0", "end")
        self.barcode_info_text.insert("end", info)
        self.barcode_info_text.configure(state="disabled")

    def update_led_color(self, label):
        """
        Mengubah warna LED dari hijau ke merah setelah pengambilan benar.
        PENTING: Fungsi ini menyentuh widget Tkinter, jadi HARUS selalu dipanggil
        dari main thread (mis. lewat self.after(0, ...) dari detection_thread),
        jangan dipanggil langsung dari thread background.
        """
        if label.endswith(self.current_group):  # Hanya proses group yang aktif
            led_list = self.leds_a if self.current_group == 'A' else self.leds_b

            for led in led_list:
                if led.cget("text") == label:
                    current_color = led.cget("fg_color")

                    # Jika LED hijau (kabel belum diambil) - KONDISI BENAR
                    if current_color == "green":
                        self._set_led_color(led, label, "red")  # Ubah ke merah
                        success_msg = f"✅ Kabel {label} berhasil diambil - {datetime.datetime.now().strftime('%H:%M:%S')}"
                        self.ls_dtc = label
                        # self.add_history_message(success_msg, 400)
                        self.last_logged_error_slot = None
                        self.check_cycle_completion()
                        return # Tidak ada suara di sini

    # --- Helper untuk Timer & Waktu ---
    def start_timer(self):
        """Memulai timer proses."""
        if not self.timer_running:
            self.timer_running = True
            self.timer_paused = False
            self.start_time = datetime.datetime.now()
            self.paused_time = 0
            self.update_timer()

    def reset_timer(self):
        """Mereset timer ke 00:00:00."""
        self.timer_running = False
        self.timer_paused = False
        self.start_time = None
        self.paused_time = 0
        self.elapsed_time = 0
        self.timer_entry.configure(state="normal")
        self.timer_entry.delete(0, "end")
        self.timer_entry.insert(0, "00:00:00")
        self.timer_entry.configure(state="readonly")
        self.pause_btn.configure(text="Pause")
        self.start_btn.configure(state="normal")

    def update_timer(self):
        """Dipanggil setiap detik untuk memperbarui tampilan timer."""
        if self.timer_running and not self.timer_paused:
            current_time = datetime.datetime.now()
            elapsed = (current_time - self.start_time).total_seconds() - self.paused_time
            self.elapsed_time = elapsed

            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            self.timer_entry.configure(state="normal")
            self.timer_entry.delete(0, "end")
            self.timer_entry.insert(0, time_str)
            self.timer_entry.configure(state="readonly")

            self.after(1000, self.update_timer)

    def start_blinking(self):
        """Starts the blinking of the status LED."""
        if not self.blinking:
            self.blinking = True
            self._blink()

    def update_time(self):
        """Dipanggil setiap detik untuk memperbarui jam aktual."""
        try:
            now = datetime.datetime.now().strftime("%H:%M:%S | %d-%m-%Y")
            if hasattr(self, 'time_entry') and self.time_entry.winfo_exists():
                self.time_entry.configure(state="normal")
                self.time_entry.delete(0, "end")
                self.time_entry.insert(0, now)
                self.time_entry.configure(state="readonly")

            self.after(1000, self.update_time)
        except Exception as e:
            print(f"Error in update_time: {e}")  # Debugging
            return  # Hentikan jika ada error

    # --- Helper untuk Efek Kedip (Blinking) ---
    def start_blinking(self):
        """Memulai animasi kedip pada LED status."""
        if not self.blinking:
            self.blinking = True
            self._blink()

    def stop_blinking(self):
        """Menghentikan animasi kedip pada LED status."""
        self.blinking = False
        self.led_canvas.itemconfig(self.led, fill="gray")

    def _blink(self):
        """Logika untuk mengubah warna LED status secara periodik."""
        if self.blinking:
            current_color = self.led_canvas.itemcget(self.led, "fill")
            new_color = "green" if current_color == "gray" else "gray"
            self.led_canvas.itemconfig(self.led, fill=new_color)
            self.after(500, self._blink)

    def _blink_status_entry(self):
        """Logika untuk kedip background status entry saat dijeda."""
        if self.is_status_blinking:
            # Tentukan warna untuk kedipan (merah atau oranye seperti history box)
            blink_color = "#DB1514" # Warna oranye yang sama dengan history error
            current_color = self.status_entry.cget("fg_color")

            # Ganti warna antara warna asli dan warna kedip
            new_color = self.original_status_fg_color if current_color == blink_color else blink_color

            self.status_entry.configure(fg_color=new_color)

            # Jadwalkan untuk pemanggilan berikutnya
            self.after(500, self._blink_status_entry)

    def _blink_status_for_selection(self):
        """Logika untuk kedip background status entry menjadi kuning saat pemilihan slot."""
        if self.is_status_blinking_selection:
            blink_color = "#E8C822" # Warna kuning keemasan
            current_color = self.status_entry.cget("fg_color")

            # Ganti warna antara warna asli dan warna kedip
            new_color = self.original_status_fg_color_selection if current_color == blink_color else blink_color

            self.status_entry.configure(fg_color=new_color)

            # Jadwalkan pemanggilan berikutnya
            self.after(500, self._blink_status_for_selection)

    def reset_history_box_highlight(self):
        """Mengembalikan tampilan border history box ke default."""
        # Gunakan warna yang sesuai dengan tema Anda, misal "gray" atau warna border default CTk
        self.history_box.configure(border_color="gray", border_width=2)

    def clear_history(self):
        """Membersihkan kotak History dan mereset status error blinking."""
        # Hentikan semua tag error yang sedang berkedip dan hapus datanya
        self.blinking_tags.clear()

        # Bersihkan teks di dalam history box
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", "end")
        self.history_text.configure(state="disabled")

    def _blink_history_errors(self):
        """Logika untuk kedip background pesan error di histori."""
        default_bg = "#f0f0f0" 

        for tag, data in list(self.blinking_tags.items()):
            state = data["state"]
            color = data["color"]

            if state == "active":
                # Matikan sorotan: kembalikan ke latar belakang default
                self.history_text.tag_config(tag, background=default_bg)
                # PERBAIKAN: Perbarui 'state' di dalam dictionary
                self.blinking_tags[tag]['state'] = "inactive" 
            else: # state == "inactive"
                # Nyalakan sorotan: atur ke warna error spesifiknya
                self.history_text.tag_config(tag, background=color)
                # PERBAIKAN: Perbarui 'state' di dalam dictionary
                self.blinking_tags[tag]['state'] = "active"

        # Jadwalkan pemanggilan berikutnya jika masih ada tag yang harus berkedip
        if self.blinking_tags:
            self.after(500, self._blink_history_errors)
        else:
            self.is_blinking_active = False

    def _stop_all_blinking(self):
        """Fungsi sentral untuk menghentikan semua efek kedip status bar."""
        # 1. Hentikan kedip kuning saat pemilihan slot
        self.is_status_blinking_selection = False
        if self.original_status_fg_color_selection:
            self.status_entry.configure(fg_color=self.original_status_fg_color_selection)

        # 2. Hentikan kedip saat sistem di-pause
        self.is_status_blinking = False
        if self.original_status_fg_color:
            self.status_entry.configure(fg_color=self.original_status_fg_color)

    # -------------------------------------------------------------------------
    # 8. EVENT HANDLERS & CALLBACKS
    # -------------------------------------------------------------------------
    def toggle_mode(self):
        """Mengganti tema antara Light/Dark Mode."""
        self.is_dark_mode = not self.is_dark_mode
        mode = "dark" if self.is_dark_mode else "light"
        ctk.set_appearance_mode(mode)
        self.update_text_colors()

    def set_error_delay_threshold(self, choice: str):
        """Callback yang dipicu saat pilihan delay diubah."""
        try:
            # Convert the selected string value to a float and store it
            self.error_delay_threshold = float(choice)
            self.update_status(f"Error delay set to {self.error_delay_threshold}s")
            print(f"Error detection delay threshold updated to: {self.error_delay_threshold} seconds")
        except ValueError:
            # This should not happen with predefined options, but it's safe to have
            print(f"Error: Could not convert '{choice}' to a float.")

    def on_closing(self):
        """
        - Dipicu saat jendela aplikasi ditutup.
        - Memastikan semua thread dan proses berhenti dengan aman.
        """
        self.stop_event.set()
        if self.camera_thread is not None:
            self.camera_thread.join()
        if self.detection_thread is not None:
            self.detection_thread.join()
        if self.serial_thread is not None:
            self.serial_thread.join()
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.cap.release()

        # Hentikan semua scheduled events
        self.after_cancel(self._after_id) if hasattr(self, '_after_id') else None
        self.destroy()

# =============================================================================
# BAGIAN C: TITIK MASUK APLIKASI
# =============================================================================
if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
