import numpy as np
import cv2


class SurgicalProcedures:
    def __init__(self):
        self.procedures = {
            'incision': self.make_incision,
            'suture': self.apply_suture,
            'cauterize': self.cauterize,
            'extract': self.extract_organ
        }

    def make_incision(self, frame, start_pos, end_pos, depth=1):
        """Simulate surgical incision"""
        color = (255, 255, 255)  # White for incision
        thickness = max(1, 3 - depth)  # Thinner for deeper cuts

        cv2.line(frame, start_pos, end_pos, color, thickness)
        if depth == 1:
            self.add_blood_effect(frame, start_pos, end_pos)

        return frame

    def apply_suture(self, frame, points):
        """Simulate suture application"""
        color = (255, 255, 0)  # Yellow for suture

        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 2)

        return frame

    def cauterize(self, frame, position, size=20):
        """Simulate cauterization"""
        x, y = position

        overlay = frame.copy()
        cv2.circle(overlay, (x, y), size, (50, 50, 50), -1)  # Dark gray

        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        return frame

    def add_blood_effect(self, frame, start_pos, end_pos):
        """Add simple blood effect"""
        length = np.sqrt((end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2)
        num_droplets = max(3, int(length / 20))

        for i in range(num_droplets):
            t = i / (num_droplets - 1) if num_droplets > 1 else 0.5
            x = int(start_pos[0] + t * (end_pos[0] - start_pos[0]))
            y = int(start_pos[1] + t * (end_pos[1] - start_pos[1]))

            # Add random offset
            x += np.random.randint(-5, 5)
            y += np.random.randint(-5, 5)

            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # Red droplets