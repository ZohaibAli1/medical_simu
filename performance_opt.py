import cv2
import numpy as np
import time
from collections import deque


class PerformanceOptimizer:
    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.frame_times = deque(maxlen=30)
        self.last_time = time.time()

    def should_skip_frame(self):
        """Adaptive frame skipping based on performance"""
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.frame_times.append(frame_time)

        avg_frame_time = np.mean(self.frame_times) if self.frame_times else frame_time
        target_frame_time = 1.0 / self.target_fps

        self.last_time = current_time
        return avg_frame_time < target_frame_time

    def optimize_frame(self, frame, scale_factor=0.8):
        """Reduce frame resolution temporarily"""
        if scale_factor < 1.0:
            h, w = frame.shape[:2]
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            frame = cv2.resize(frame, (new_w, new_h))
        return frame


class GestureSmoother:
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.position_buffer = deque(maxlen=buffer_size)
        self.rotation_buffer = deque(maxlen=buffer_size)

    def smooth_position(self, new_position):
        self.position_buffer.append(new_position)
        if len(self.position_buffer) < 2:
            return new_position

        weights = np.linspace(0.5, 1.0, len(self.position_buffer))
        weights /= weights.sum()

        x_avg = sum(pos[0] * w for pos, w in zip(self.position_buffer, weights))
        y_avg = sum(pos[1] * w for pos, w in zip(self.position_buffer, weights))

        return (int(x_avg), int(y_avg))

    def smooth_rotation(self, new_rotation):
        self.rotation_buffer.append(new_rotation)
        if len(self.rotation_buffer) < 2:
            return new_rotation

        return np.mean(list(self.rotation_buffer), axis=0)