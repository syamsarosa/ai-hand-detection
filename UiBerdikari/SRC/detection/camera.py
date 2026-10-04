"""Camera capture and frame-preprocessing pipeline."""

import queue
import threading

import cv2
import numpy as np

class Camera:
    """Owns camera capture, capture-thread lifecycle, and frame preprocessing."""

    def __init__(self, index=0, width=720, height=480, fps=120):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.thread = None
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=1)

    def start(self):
        """Open the camera and start the capture worker."""
        self.stop_event.clear()
        self.cap = cv2.VideoCapture(self.index)

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        """Stop the capture worker and release the camera device."""
        self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.thread = None

    def get_frame(self, timeout=0.01):
        """Return the next preprocessed frame from the capture queue."""
        return self.frame_queue.get(timeout=timeout)

    def _capture_loop(self):
        """Continuously capture and preprocess frames."""
        while not self.stop_event.is_set():
            if self.cap is None:
                break

            ret, frame = self.cap.read()
            if not ret:
                continue

            try:
                frame = self._preprocess_frame(frame)

                if self.frame_queue.empty():
                    self.frame_queue.put(frame)
            except queue.Full:
                pass
            except Exception:
                continue

    @staticmethod
    def _preprocess_frame(frame):
        """Apply the existing flip and fisheye correction without changing its parameters."""
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

        return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
