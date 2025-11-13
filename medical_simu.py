import cv2
import mediapipe as mp
import numpy as np
import math
from enum import Enum


class GestureType(Enum):
    NONE = 0
    ROTATE = 1
    ZOOM = 2
    SELECT = 3
    DISSECT = 4


class AnatomyLayer(Enum):
    SKIN = 0
    MUSCLES = 1
    ORGANS = 2
    SKELETON = 3


class MedicalTrainingSimulator:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # 3D Model state
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        self.zoom = 1.0
        self.current_layer = AnatomyLayer.SKIN

        # Gesture tracking
        self.prev_hand_positions = {}
        self.gesture_start_distance = None

        # Anatomical data
        self.organs = {
            'heart': {'position': (0, 50, 0), 'size': 80, 'color': (0, 0, 200)},
            'left_lung': {'position': (-70, 30, -20), 'size': 90, 'color': (200, 150, 150)},
            'right_lung': {'position': (70, 30, -20), 'size': 90, 'color': (200, 150, 150)},
            'liver': {'position': (40, 100, 0), 'size': 100, 'color': (100, 50, 50)},
            'stomach': {'position': (-30, 90, 10), 'size': 70, 'color': (150, 100, 100)},
        }

        self.selected_organ = None

    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def detect_gesture(self, hand_landmarks, hand_label, img_width, img_height):
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        thumb_pos = (int(thumb_tip.x * img_width), int(thumb_tip.y * img_height))
        index_pos = (int(index_tip.x * img_width), int(index_tip.y * img_height))
        thumb_index_dist = self.calculate_distance(thumb_pos, index_pos)
        if thumb_index_dist < 40:
            return GestureType.SELECT, index_pos
        fingers_up = sum([
            index_tip.y < middle_tip.y < wrist.y,
            middle_tip.y < wrist.y,
            ring_tip.y < wrist.y,
            pinky_tip.y < wrist.y
        ])

        if fingers_up >= 3:
            return GestureType.ROTATE, index_pos

        return GestureType.NONE, index_pos

    def draw_3d_organ(self, frame, organ_name, organ_data, center_x, center_y):
        x, y, z = organ_data['position']
        size = organ_data['size']
        color = organ_data['color']
        cos_x = math.cos(self.rotation_x)
        sin_x = math.sin(self.rotation_x)
        cos_y = math.cos(self.rotation_y)
        sin_y = math.sin(self.rotation_y)
        rotated_x = x * cos_y - z * sin_y
        rotated_z = x * sin_y + z * cos_y
        rotated_y = y * cos_x - rotated_z * sin_x
        scaled_size = int(size * self.zoom)

        # Project to 2D
        screen_x = int(center_x + rotated_x * self.zoom)
        screen_y = int(center_y + rotated_y * self.zoom)

        # Draw based on current layer
        alpha = 1.0
        if self.current_layer == AnatomyLayer.SKELETON:
            if organ_name == 'heart':
                alpha = 0.3
        elif self.current_layer == AnatomyLayer.ORGANS:
            alpha = 0.8
        elif self.current_layer == AnatomyLayer.MUSCLES:
            alpha = 0.5

        # Create overlay for transparency
        overlay = frame.copy()

        # Draw organ (simplified as ellipse for now)
        cv2.ellipse(overlay, (screen_x, screen_y),
                    (scaled_size, int(scaled_size * 0.8)),
                    0, 0, 360, color, -1)

        # Apply transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw outline
        cv2.ellipse(frame, (screen_x, screen_y),
                    (scaled_size, int(scaled_size * 0.8)),
                    0, 0, 360, (255, 255, 255), 2)

        # Highlight if selected
        if self.selected_organ == organ_name:
            cv2.ellipse(frame, (screen_x, screen_y),
                        (scaled_size + 10, int(scaled_size * 0.8) + 10),
                        0, 0, 360, (0, 255, 255), 3)

        return frame, (screen_x, screen_y)

    def draw_medical_info(self, frame, organ_name, position):
        """Medical information overlay karo"""
        info = {
            'heart': ['Myocardium: Cardiac muscle',
                      'Function: Pumps blood',
                      'Rate: 60-100 bpm'],
            'left_lung': ['Lobes: 2 (Superior, Inferior)',
                          'Capacity: ~1.5L',
                          'Function: Gas exchange'],
            'right_lung': ['Lobes: 3 (Superior, Middle, Inferior)',
                           'Capacity: ~1.8L',
                           'Function: Oxygenation'],
            'liver': ['Weight: ~1.5 kg',
                      'Function: Detoxification',
                      'Regenerates: Yes'],
            'stomach': ['Capacity: 1-2 L',
                        'pH: 1.5-3.5',
                        'Function: Digestion']
        }

        if organ_name in info:
            x, y = position
            y_offset = 0

            # Draw info box
            box_height = len(info[organ_name]) * 25 + 20
            cv2.rectangle(frame, (x + 100, y - 10),
                          (x + 400, y + box_height),
                          (40, 40, 40), -1)
            cv2.rectangle(frame, (x + 100, y - 10),
                          (x + 400, y + box_height),
                          (255, 255, 255), 2)

            # Title
            cv2.putText(frame, organ_name.upper().replace('_', ' '),
                        (x + 110, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Info lines
            for i, line in enumerate(info[organ_name]):
                cv2.putText(frame, line,
                            (x + 110, y + 45 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def draw_skeleton(self, frame, center_x, center_y):
        # Spine
        spine_top = (center_x, int(center_y - 150 * self.zoom))
        spine_bottom = (center_x, int(center_y + 150 * self.zoom))
        cv2.line(frame, spine_top, spine_bottom, (200, 200, 200), 3)

        # Ribs (simplified)
        for i in range(-5, 6):
            y_pos = int(center_y + i * 20 * self.zoom)
            rib_length = int(80 * self.zoom - abs(i) * 10)
            cv2.ellipse(frame, (center_x, y_pos),
                        (rib_length, 15), 0, 0, 180, (200, 200, 200), 2)

        # Skull
        cv2.circle(frame, (center_x, int(center_y - 180 * self.zoom)),
                   int(40 * self.zoom), (200, 200, 200), 2)

        return frame

    def draw_ui(self, frame):
        """UI controls draw karo"""
        h, w = frame.shape[:2]

        # Control panel
        cv2.rectangle(frame, (10, 10), (300, 200), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, 10), (300, 200), (255, 255, 255), 2)

        # Title
        cv2.putText(frame, "MEDICAL TRAINING SIMULATOR", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Layer info
        layer_name = self.current_layer.name
        cv2.putText(frame, f"Layer: {layer_name}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Zoom info
        cv2.putText(frame, f"Zoom: {self.zoom:.1f}x", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Controls
        cv2.putText(frame, "Controls:", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, "L - Change Layer", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "+/- - Zoom", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "Open Palm - Rotate", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Selected organ info
        if self.selected_organ:
            cv2.putText(frame, f"Selected: {self.selected_organ}",
                        (w - 300, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    def run(self):
        """Main simulation loop"""
        cap = cv2.VideoCapture(0)

        print("=" * 50)
        print("MEDICAL TRAINING SIMULATOR")
        print("=" * 50)
        print("Controls:")
        print("  L - Change anatomical layer")
        print("  +/- - Zoom in/out")
        print("  R - Reset view")
        print("  Q - Quit")
        print("=" * 50)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            center_x, center_y = w // 2, h // 2

            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # Process hand gestures
            if results.multi_hand_landmarks and results.multi_handedness:
                for idx, (hand_landmarks, handedness) in enumerate(
                        zip(results.multi_hand_landmarks, results.multi_handedness)):

                    hand_label = handedness.classification[0].label
                    gesture, position = self.detect_gesture(
                        hand_landmarks, hand_label, w, h)

                    # Handle gestures
                    if gesture == GestureType.ROTATE:
                        # Track hand movement for rotation
                        if hand_label in self.prev_hand_positions:
                            prev_pos = self.prev_hand_positions[hand_label]
                            dx = position[0] - prev_pos[0]
                            dy = position[1] - prev_pos[1]

                            self.rotation_y += dx * 0.01
                            self.rotation_x += dy * 0.01

                        self.prev_hand_positions[hand_label] = position

                        # Draw hand skeleton
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    elif gesture == GestureType.SELECT:
                        # Check if touching any organ
                        for organ_name, organ_data in self.organs.items():
                            # Simplified hit detection
                            organ_screen_pos = (
                                int(center_x + organ_data['position'][0] * self.zoom),
                                int(center_y + organ_data['position'][1] * self.zoom)
                            )
                            dist = self.calculate_distance(position, organ_screen_pos)

                            if dist < organ_data['size'] * self.zoom:
                                self.selected_organ = organ_name
                                break

            # Draw skeleton if in skeleton layer
            if self.current_layer == AnatomyLayer.SKELETON:
                frame = self.draw_skeleton(frame, center_x, center_y)

            # Draw organs
            organ_positions = {}
            for organ_name, organ_data in self.organs.items():
                frame, pos = self.draw_3d_organ(
                    frame, organ_name, organ_data, center_x, center_y)
                organ_positions[organ_name] = pos

            # Draw medical info for selected organ
            if self.selected_organ and self.selected_organ in organ_positions:
                frame = self.draw_medical_info(
                    frame, self.selected_organ, organ_positions[self.selected_organ])

            # Draw UI
            frame = self.draw_ui(frame)

            # Display
            cv2.imshow('Medical Training Simulator', frame)

            # Keyboard controls
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                # Cycle through layers
                layers = list(AnatomyLayer)
                current_idx = layers.index(self.current_layer)
                self.current_layer = layers[(current_idx + 1) % len(layers)]
            elif key == ord('+') or key == ord('='):
                self.zoom = min(self.zoom + 0.1, 3.0)
            elif key == ord('-'):
                self.zoom = max(self.zoom - 0.1, 0.5)
            elif key == ord('r'):
                # Reset
                self.rotation_x = 0
                self.rotation_y = 0
                self.zoom = 1.0
                self.selected_organ = None

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    simulator = MedicalTrainingSimulator()
    simulator.run()