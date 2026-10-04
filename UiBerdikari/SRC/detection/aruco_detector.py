import cv2
import cv2.aruco as aruco
import numpy as np


class ArucoDetector:
    """Handles ArUco/AprilTag detection and pose calculation."""

    def __init__(
        self,
        camera_matrix,
        dist_coeffs,
        marker_size=5.0,
    ):
        self.marker_size = marker_size
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16h5)

        self.parameters = aruco.DetectorParameters()
        self.parameters.adaptiveThreshWinSizeMin = 5
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 5
        self.parameters.minMarkerPerimeterRate = 0.08
        self.parameters.maxMarkerPerimeterRate = 4.0
        self.parameters.minCornerDistanceRate = 0.05
        self.parameters.minOtsuStdDev = 5.0
        self.parameters.polygonalApproxAccuracyRate = 0.03
        self.parameters.minMarkerDistanceRate = 0.05

        self.detector = aruco.ArucoDetector(
            self.dictionary,
            self.parameters,
        )

    def detect(self, frame, output_frame=None):
        """
        Detect markers and calculate their center and distance.

        Returns:
            list[dict]: One result for each detected marker.
        """
        if output_frame is None:
            output_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)

        detections = []

        if ids is None:
            return detections

        for corner, marker_id in zip(corners, ids.flatten()):
            half_size = self.marker_size / 2.0

            obj_points = np.array(
                [
                    [-half_size, half_size, 0],
                    [half_size, half_size, 0],
                    [half_size, -half_size, 0],
                    [-half_size, -half_size, 0],
                ],
                dtype=np.float32,
            )

            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                corner,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )

            if success:
                aruco.drawDetectedMarkers(
                    output_frame,
                    [corner],
                )
                cv2.drawFrameAxes(
                    output_frame,
                    self.camera_matrix,
                    self.dist_coeffs,
                    rvec,
                    tvec,
                    2,
                )

            x_c = int(np.mean(corner[0][:, 0]))
            y_c = int(np.mean(corner[0][:, 1]))
            distance = np.linalg.norm(tvec)

            text = f"x:{x_c} y:{y_c} Dist:{distance:.2f} cm"
            text_size, _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                2,
            )

            text_x = int(corner[0][0][0]) - text_size[0] - 10
            text_y = int(corner[0][0][1]) - 10
            text_x = max(text_x, 0)

            cv2.putText(
                output_frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            detections.append(
                {
                    "id": int(marker_id),
                    "corner": corner,
                    "x": x_c,
                    "y": y_c,
                    "distance": distance,
                    "rvec": rvec,
                    "tvec": tvec,
                }
            )

        return detections
