import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import os
from PIL import Image, ImageTk

class JendelaKalibrasi:
    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
        self.top.title("Kalibrasi Sistem & Area Persegi 4 Tag")
        self.top.geometry("1450x650")

        # Nama file teks untuk menyimpan data koordinat
        self.config_file = "config_kalibrasi.txt"

        # Frame UI
        frame_kiri = tk.Frame(self.top)
        frame_kiri.pack(side=tk.LEFT, padx=10, pady=10)

        frame_kanan = tk.Frame(self.top)
        frame_kanan.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.label_video = tk.Label(frame_kiri)
        self.label_video.pack()

        lbl_info = tk.Label(frame_kanan, text="Status & Data Kalibrasi:", font=("Arial", 10, "bold"))
        lbl_info.pack(anchor="w")

        self.txt_data = tk.Text(frame_kanan, width=45, height=18, font=("Consolas", 9))
        self.txt_data.pack(fill=tk.BOTH, expand=True, pady=5)

        # --- PANEL INPUT MANUAL 4 TITIK BIRU ---
        frame_manual = tk.LabelFrame(frame_kanan, text="Input Manual Persegi Biru (x, y)", font=("Arial", 9, "bold"))
        frame_manual.pack(fill=tk.X, pady=5)

        self.entries = []
        titik_label = ["TL (Top-Left):", "TR (Top-Right):", "BR (Bottom-Right):", "BL (Bottom-Left):"]
        
        # Load nilai dari file teks jika ada, jika tidak pakai default
        saved_values = self.muat_konfigurasi()
        
        for i, label_text in enumerate(titik_label):
            row_f = tk.Frame(frame_manual)
            row_f.pack(fill=tk.X, padx=5, pady=2)
            
            lbl = tk.Label(row_f, width=15, text=label_text, anchor="w", font=("Arial", 8))
            lbl.pack(side=tk.LEFT)
            
            ent = tk.Entry(row_f, width=15, font=("Consolas", 9))
            ent.insert(0, saved_values[i])
            ent.config(state="disabled")  # Default terkunci
            ent.pack(side=tk.LEFT, padx=2)
            self.entries.append(ent)

        # Tombol Toggle Edit / Simpan
        self.is_editable = False
        self.btn_toggle = tk.Button(frame_manual, text="Enable Edit", bg="#007ACC", fg="white", font=("Arial", 9, "bold"), command=self.toggle_edit)
        self.btn_toggle.pack(fill=tk.X, padx=5, pady=5)

        # Inisialisasi awal manual_pts berdasarkan nilai yang dimuat
        temp_init_pts = []
        for val_str in saved_values:
            x_s, y_s = val_str.split(',')
            temp_init_pts.append([float(x_s.strip()), float(y_s.strip())])
        self.manual_pts = np.array(temp_init_pts, dtype="float32")

        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1100)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        self.top.protocol("WM_DELETE_WINDOW", self.tutup)

        self.proses_frame()

    def muat_konfigurasi(self):
        """Membaca data konfigurasi dari file teks baris per baris"""
        default_values = ["21,129", "934,130", "911,467", "44,462"]
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    if len(lines) == 4:
                        return lines
            except Exception:
                pass
        return default_values

    def simpan_konfigurasi(self, values_list):
        """Menyimpan data koordinat ke file teks (baris per baris)"""
        try:
            with open(self.config_file, "w") as f:
                for val in values_list:
                    f.write(f"{val}\n")
        except Exception as e:
            print(f"Gagal menyimpan konfigurasi: {e}")

    def toggle_edit(self):
        """Mengubah status entry antara bisa diedit atau dikunci"""
        if not self.is_editable:
            for ent in self.entries:
                ent.config(state="normal")
            self.btn_toggle.config(text="Simpan & Terapkan", bg="#28A745")
            self.is_editable = True
        else:
            try:
                temp_pts = []
                raw_values = []
                for ent in self.entries:
                    val = ent.get().strip()
                    if not val:
                        raise ValueError("Semua titik harus diisi!")
                    x_str, y_str = val.split(',')
                    temp_pts.append([float(x_str.strip()), float(y_str.strip())])
                    raw_values.append(val)
                
                self.manual_pts = np.array(temp_pts, dtype="float32")
                
                # Simpan permanen ke file teks (.txt)
                self.simpan_konfigurasi(raw_values)
                
                for ent in self.entries:
                    ent.config(state="disabled")
                self.btn_toggle.config(text="Enable Edit", bg="#007ACC")
                self.is_editable = False
                messagebox.showinfo("Berhasil", "Titik manual berhasil disimpan ke file TXT dan diterapkan!")
            except Exception as e:
                messagebox.showerror("Error Input", f"Format salah! Gunakan format x,y (Contoh: 150,200)\nDetail: {e}")

    def urutkan_titik_persegi(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def proses_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, -1)
                h, w = frame.shape[:2]
                fx_val = 0.89
                k1_val = -0.34
                k2_val = 0.49

                fx = w * max(fx_val, 0.1)
                fy = fx
                cx = w / 2.00
                cy = h / 2.00

                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
                D = np.array([k1_val, k2_val, 0.0, 0.0], dtype=np.float64)

                new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K, D, (w, h), np.eye(3), balance=0.5
                )
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
                )
                
                frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
                
                teks_output = ""

                # Deteksi AprilTag 16h5 / ArUco
                try:
                    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
                    parameters = cv2.aruco.DetectorParameters()
                    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                    corners, ids, rejected = detector.detectMarkers(frame)
                except AttributeError:
                    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_16h5)
                    parameters = cv2.aruco.DetectorParameters_create()
                    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

                center_points = []

                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    
                    teks_output += f"Total Tag Terdeteksi: {len(ids)}\n"
                    teks_output += "="*37 + "\n"

                    for i, tag_id in enumerate(ids.flatten()):
                        pts = corners[i][0]
                        cx = int(pts[:, 0].mean())
                        cy = int(pts[:, 1].mean())
                        center_points.append([cx, cy])

                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                        teks_output += f"Tag ID [{tag_id}] : ({cx}, {cy})\n"

                    if len(center_points) >= 4:
                        pts_arr = np.array(center_points[:4], dtype="float32")
                        rect_ordered = self.urutkan_titik_persegi(pts_arr)
                        rect_int = rect_ordered.astype(int)

                        cv2.polylines(frame, [rect_int], isClosed=True, color=(0, 255, 0), thickness=2)
                        overlay = frame.copy()
                        cv2.fillPoly(overlay, [rect_int], color=(0, 255, 0))
                        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

                        lebar_atas = np.linalg.norm(rect_ordered[0] - rect_ordered[1])
                        lebar_bawah = np.linalg.norm(rect_ordered[3] - rect_ordered[2])
                        tinggi_kiri = np.linalg.norm(rect_ordered[0] - rect_ordered[3])
                        tinggi_kanan = np.linalg.norm(rect_ordered[1] - rect_ordered[2])
                        
                        lebar_avg = (lebar_atas + lebar_bawah) / 2
                        tinggi_avg = (tinggi_kiri + tinggi_kanan) / 2
                        area = cv2.contourArea(rect_int)

                        teks_output += "="*37 + "\n"
                        teks_output += "STATUS: PERSEGI HIJAU (ARUCO) TERKUNCI\n"
                        teks_output += f"Rata-rata Lebar  : {lebar_avg:.1f} px\n"
                        teks_output += f"Rata-rata Tinggi : {tinggi_avg:.1f} px\n"
                        teks_output += f"Estimasi Luas    : {area:.1f} px^2\n"
                    else:
                        teks_output += "\n[!] Perlu minimal 4 marker untuk persegi hijau.\n"
                else:
                    teks_output = "Menunggu tag AprilTag 16h5...\n"

                # Render Persegi Manual Biru
                if self.manual_pts is not None:
                    manual_int = self.manual_pts.astype(int)
                    cv2.polylines(frame, [manual_int], isClosed=True, color=(255, 0, 0), thickness=1)
                    
                    overlay_b = frame.copy()
                    cv2.fillPoly(overlay_b, [manual_int], color=(255, 0, 0))
                    cv2.addWeighted(overlay_b, 0.15, frame, 0.85, 0, frame)

                    for pt in manual_int:
                        cv2.circle(frame, tuple(pt), 1, (255, 0, 0), -1)

                    teks_output += "\n" + "="*37 + "\n"
                    teks_output += "STATUS: PERSEGI BIRU MANUAL AKTIF\n"
                    m_area = cv2.contourArea(self.manual_pts.astype(int))
                    teks_output += f"Luas Manual (Biru): {m_area:.1f} px^2\n"

                self.txt_data.delete("1.0", tk.END)
                self.txt_data.insert(tk.END, teks_output)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(image=img)
                self.label_video.img_tk = img_tk
                self.label_video.configure(image=img_tk)

        if self.top.winfo_exists():
            self.top.after(15, self.proses_frame)

    def tutup(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.top.destroy()