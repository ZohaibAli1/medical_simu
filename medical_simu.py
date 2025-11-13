# [file name]: enhanced_medical_simulator.py
import cv2
import mediapipe as mp
import numpy as np
import math
from enum import Enum
import time


class GestureType(Enum):
    NONE = 0
    ROTATE = 1
    ZOOM = 2
    SELECT = 3
    DISSECT = 4
    SCALPEL = 5


class AnatomyLayer(Enum):
    SKIN = 0
    MUSCLES = 1
    ORGANS = 2
    SKELETON = 3
    NERVES = 4


    def __init__(self):
        # Optimized MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,  # Reduced for performance
            min_tracking_confidence=0.6
        )

        self.anatomy_layers = {
            AnatomyLayer.SKIN: {
                'visible': True,
                'opacity': 0.3,
                'color': (255, 218, 185)
            },
            AnatomyLayer.MUSCLES: {
                'visible': False,
                'opacity': 0.7,
                'color': (205, 133, 63)
            },
            AnatomyLayer.ORGANS: {
                'visible': True,
                'opacity': 0.8,
                'color': None  # Organ-specific colors
            },
            AnatomyLayer.SKELETON: {
                'visible': False,
                'opacity': 0.9,
                'color': (240, 240, 240)
            },
            AnatomyLayer.NERVES: {
                'visible': False,
                'opacity': 0.6,
                'color': (255, 255, 0)
            }
        }

        self.organs = {
            'heart': {
                'position': (0, 50, 0), 'size': 80, 'color': (0, 0, 200),
                'layer': AnatomyLayer.ORGANS, 'health': 100
            },
            'left_lung': {
                'position': (-70, 30, -20), 'size': 90, 'color': (200, 150, 150),
                'layer': AnatomyLayer.ORGANS, 'health': 100
            },
            'right_lung': {
                'position': (70, 30, -20), 'size': 90, 'color': (200, 150, 150),
                'layer': AnatomyLayer.ORGANS, 'health': 100
            },
            'liver': {
                'position': (40, 100, 0), 'size': 100, 'color': (100, 50, 50),
                'layer': AnatomyLayer.ORGANS, 'health': 100
            },
            'stomach': {
                'position': (-30, 90, 10), 'size': 70, 'color': (150, 100, 100),
                'layer': AnatomyLayer.ORGANS, 'health': 100
            },
            'brain': {
                'position': (0, -80, 0), 'size': 60, 'color': (255, 182, 193),
                'layer': AnatomyLayer.ORGANS, 'health': 100
            },
            'kidney_left': {
                'position': (-50, 120, 10), 'size': 40, 'color': (139, 69, 19),
                'layer': AnatomyLayer.ORGANS, 'health': 100
            },
            'kidney_right': {
                'position': (50, 120, 10), 'size': 40, 'color': (139, 69, 19),
                'layer': AnatomyLayer.ORGANS, 'health': 100
            }
        }

        # Simulation state
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom = 1.0
        self.current_layer = AnatomyLayer.SKIN
        self.selected_organ = None
        self.surgical_mode = False
        self.dissected_organs = set()

        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()

    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def detect_enhanced_gesture(self, hand_landmarks, hand_label, img_width, img_height):
        """Enhanced gesture detection with surgical tools"""
        landmarks = hand_landmarks.landmark

        # Get key points
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]

        # Convert to pixel coordinates
        thumb_pos = (int(thumb_tip.x * img_width), int(thumb_tip.y * img_height))
        index_pos = (int(index_tip.x * img_width), int(index_tip.y * img_height))
        middle_pos = (int(middle_tip.x * img_width), int(middle_tip.y * img_height))

        # Calculate distances
        thumb_index_dist = self.calculate_distance(thumb_pos, index_pos)
        index_middle_dist = self.calculate_distance(index_pos, middle_pos)

        # Gesture recognition
        fingers_up = sum([
            index_tip.y < middle_tip.y < wrist.y,  # Index up
            middle_tip.y < wrist.y,  # Middle up
            ring_tip.y < wrist.y,  # Ring up
            pinky_tip.y < wrist.y  # Pinky up
        ])

        # Scalpel gesture (index and middle finger together)
        if index_middle_dist < 30 and fingers_up >= 2:
            return GestureType.SCALPEL, index_pos

        # Selection gesture (pinch)
        elif thumb_index_dist < 40:
            return GestureType.SELECT, index_pos

        # Rotation gesture (open hand)
        elif fingers_up >= 3:
            return GestureType.ROTATE, index_pos

        # Dissection gesture (fist)
        elif fingers_up <= 1:
            return GestureType.DISSECT, index_pos

        return GestureType.NONE, index_pos

    def draw_optimized_organ(self, frame, organ_name, organ_data, center_x, center_y):
        if organ_name in self.dissected_organs:
            return frame, None

        x, y, z = organ_data['position']

        cos_x, sin_x = math.cos(self.rotation_x), math.sin(self.rotation_x)
        cos_y, sin_y = math.cos(self.rotation_y), math.sin(self.rotation_y)

        # Apply rotation
        rotated_x = x * cos_y - z * sin_y
        rotated_z = x * sin_y + z * cos_y
        rotated_y = y * cos_x - rotated_z * sin_x

        # Project to 2D
        screen_x = int(center_x + rotated_x * self.zoom)
        screen_y = int(center_y + rotated_y * self.zoom)
        scaled_size = int(organ_data['size'] * self.zoom)

        # Skip if outside view (early culling)
        if (screen_x < -scaled_size or screen_x > frame.shape[1] + scaled_size or
                screen_y < -scaled_size or screen_y > frame.shape[0] + scaled_size):
            return frame, None

        # Layer-based visibility
        layer = organ_data['layer']
        if not self.anatomy_layers[layer]['visible']:
            return frame, None

        # Calculate opacity based on layer
        base_opacity = self.anatomy_layers[layer]['opacity']
        if self.current_layer.value > layer.value:
            base_opacity *= 0.3  # Dim organs in deeper layers

        # Create overlay
        overlay = frame.copy()
        color = organ_data['color']

        # Draw organ with health-based coloring
        health = organ_data['health']
        if health < 70:
            color = (0, 0, 255)  # Red for damaged
        elif health < 90:
            color = (0, 165, 255)  # Orange for slightly damaged

        # Simple ellipse for organ
        cv2.ellipse(overlay, (screen_x, screen_y),
                    (scaled_size, int(scaled_size * 0.8)), 0, 0, 360, color, -1)

        # Apply transparency
        cv2.addWeighted(overlay, base_opacity, frame, 1 - base_opacity, 0, frame)

        # Outline
        cv2.ellipse(frame, (screen_x, screen_y),
                    (scaled_size, int(scaled_size * 0.8)), 0, 0, 360, (255, 255, 255), 2)

        # Selection highlight
        if self.selected_organ == organ_name:
            cv2.ellipse(frame, (screen_x, screen_y),
                        (scaled_size + 10, int(scaled_size * 0.8) + 10),
                        0, 0, 360, (0, 255, 255), 3)

        return frame, (screen_x, screen_y)

    def draw_enhanced_skeleton(self, frame, center_x, center_y):
        """Optimized skeleton drawing"""
        if not self.anatomy_layers[AnatomyLayer.SKELETON]['visible']:
            return frame

        scale = self.zoom
        color = self.anatomy_layers[AnatomyLayer.SKELETON]['color']
        opacity = self.anatomy_layers[AnatomyLayer.SKELETON]['opacity']

        # Create overlay for transparency
        overlay = frame.copy()

        # Spine (simplified)
        spine_top = (center_x, int(center_y - 150 * scale))
        spine_bottom = (center_x, int(center_y + 150 * scale))
        cv2.line(overlay, spine_top, spine_bottom, color, 3)

        # Ribs
        for i in range(-5, 6):
            y_pos = int(center_y + i * 20 * scale)
            rib_length = int(80 * scale - abs(i) * 10)
            cv2.ellipse(overlay, (center_x, y_pos),
                        (rib_length, 15), 0, 0, 180, color, 2)

        # Pelvis
        cv2.ellipse(overlay, (center_x, int(center_y + 160 * scale)),
                    (60, 25), 0, 0, 360, color, 2)

        # Skull
        cv2.circle(overlay, (center_x, int(center_y - 180 * scale)),
                   int(40 * scale), color, 2)

        # Apply transparency
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

        return frame

    def draw_nervous_system(self, frame, center_x, center_y):
        """Draw simplified nervous system"""
        if not self.anatomy_layers[AnatomyLayer.NERVES]['visible']:
            return frame

        scale = self.zoom
        color = self.anatomy_layers[AnatomyLayer.NERVES]['color']
        opacity = self.anatomy_layers[AnatomyLayer.NERVES]['opacity']

        overlay = frame.copy()

        # Spinal cord
        spine_top = (center_x, int(center_y - 150 * scale))
        spine_bottom = (center_x, int(center_y + 150 * scale))
        cv2.line(overlay, spine_top, spine_bottom, color, 2)

        # Nerve branches (simplified)
        for i in range(-4, 5):
            y_pos = int(center_y + i * 30 * scale)
            branch_length = int(50 * scale)
            cv2.line(overlay, (center_x, y_pos),
                     (center_x + branch_length, y_pos), color, 1)
            cv2.line(overlay, (center_x, y_pos),
                     (center_x - branch_length, y_pos), color, 1)

        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
        return frame

    def draw_muscular_system(self, frame, center_x, center_y):
        """Draw simplified muscular system"""
        if not self.anatomy_layers[AnatomyLayer.MUSCLES]['visible']:
            return frame

        scale = self.zoom
        color = self.anatomy_layers[AnatomyLayer.MUSCLES]['color']
        opacity = self.anatomy_layers[AnatomyLayer.MUSCLES]['opacity']

        overlay = frame.copy()

        # Major muscle groups (simplified)
        # Pectorals
        cv2.ellipse(overlay, (center_x, int(center_y + 20 * scale)),
                    (80, 40), 0, 0, 180, color, -1)

        # Abdominals
        for i in range(3):
            y_pos = int(center_y + 60 * scale + i * 20 * scale)
            cv2.rectangle(overlay,
                          (center_x - 40, y_pos - 8),
                          (center_x + 40, y_pos + 8), color, -1)

        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
        return frame

    def perform_surgery(self, organ_name, surgery_type):
        """Simulate surgical procedures"""
        if organ_name not in self.organs:
            return

        organ = self.organs[organ_name]

        if surgery_type == "dissect":
            self.dissected_organs.add(organ_name)
            print(f"Dissected {organ_name}")

        elif surgery_type == "repair":
            organ['health'] = min(100, organ['health'] + 30)
            print(f"Repaired {organ_name}, health: {organ['health']}%")

        elif surgery_type == "damage":
            organ['health'] = max(0, organ['health'] - 20)
            print(f"Damaged {organ_name}, health: {organ['health']}%")

    def draw_enhanced_ui(self, frame):
        """Enhanced UI with performance metrics"""
        h, w = frame.shape[:2]

        # Control panel
        cv2.rectangle(frame, (10, 10), (350, 250), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, 10), (350, 250), (255, 255, 255), 2)

        # Title
        cv2.putText(frame, "ENHANCED MEDICAL SIMULATOR", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Performance info
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Layer info
        layer_name = self.current_layer.name
        cv2.putText(frame, f"Layer: {layer_name}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Zoom info
        cv2.putText(frame, f"Zoom: {self.zoom:.1f}x", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Surgical mode
        mode_status = "ON" if self.surgical_mode else "OFF"
        mode_color = (0, 255, 0) if self.surgical_mode else (0, 0, 255)
        cv2.putText(frame, f"Surgical Mode: {mode_status}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)

        # Controls
        cv2.putText(frame, "Controls:", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        controls = [
            "L - Change Layer", "+/- - Zoom", "S - Surgical Mode",
            "R - Reset", "D - Dissect", "H - Heal"
        ]
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (20, 190 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Selected organ info
        if self.selected_organ:
            health = self.organs[self.selected_organ]['health']
            health_color = (0, 255, 0) if health > 80 else (0, 165, 255) if health > 60 else (0, 0, 255)

            cv2.putText(frame, f"Selected: {self.selected_organ}",
                        (w - 300, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Health: {health}%",
                        (w - 300, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, health_color, 2)

        return frame

    def update_performance_metrics(self):
        """Update FPS and performance metrics"""
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time

    def run_enhanced_simulation(self):
        """Enhanced main simulation loop"""
        cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("=" * 60)
        print("ENHANCED MEDICAL TRAINING SIMULATOR")
        print("=" * 60)
        print("Controls:")
        print("  L - Cycle through anatomical layers")
        print("  S - Toggle surgical mode")
        print("  D - Dissect selected organ")
        print("  H - Heal selected organ")
        print("  +/- - Zoom in/out")
        print("  R - Reset simulation")
        print("  Q - Quit")
        print("=" * 60)

        prev_hand_positions = {}

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.resize(frame, (800, 600))
            frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks and results.multi_handedness:
                for idx, (hand_landmarks, handedness) in enumerate(
                        zip(results.multi_hand_landmarks, results.multi_handedness)):

                    hand_label = handedness.classification[0].label
                    gesture, position = self.detect_enhanced_gesture(
                        hand_landmarks, hand_label, w, h)

                    # Handle gestures
                    if gesture == GestureType.ROTATE:
                        if hand_label in prev_hand_positions:
                            prev_pos = prev_hand_positions[hand_label]
                            dx = position[0] - prev_pos[0]
                            dy = position[1] - prev_pos[1]
                            self.rotation_y += dx * 0.01
                            self.rotation_x += dy * 0.01

                        prev_hand_positions[hand_label] = position
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    elif gesture == GestureType.SELECT and self.surgical_mode:
                        # Organ selection in surgical mode
                        for organ_name, organ_data in self.organs.items():
                            if organ_name in self.dissected_organs:
                                continue

                            organ_screen_pos = (
                                int(center_x + organ_data['position'][0] * self.zoom),
                                int(center_y + organ_data['position'][1] * self.zoom)
                            )
                            dist = self.calculate_distance(position, organ_screen_pos)

                            if dist < organ_data['size'] * self.zoom:
                                self.selected_organ = organ_name
                                break

                    elif gesture == GestureType.SCALPEL and self.surgical_mode and self.selected_organ:
                        # Scalpel action - damage organ
                        self.perform_surgery(self.selected_organ, "damage")

            # Draw anatomical systems in order
            frame = self.draw_enhanced_skeleton(frame, center_x, center_y)
            frame = self.draw_nervous_system(frame, center_x, center_y)
            frame = self.draw_muscular_system(frame, center_x, center_y)

            # Draw organs
            organ_positions = {}
            for organ_name, organ_data in self.organs.items():
                frame, pos = self.draw_optimized_organ(
                    frame, organ_name, organ_data, center_x, center_y)
                if pos:
                    organ_positions[organ_name] = pos

            if self.selected_organ and self.selected_organ in organ_positions:
                self.draw_medical_info(frame, self.selected_organ, organ_positions[self.selected_organ])

            frame = self.draw_enhanced_ui(frame)

            # Update performance metrics
            self.update_performance_metrics()

            cv2.imshow('Enhanced Medical Training Simulator', frame)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                # Cycle through layers
                layers = list(AnatomyLayer)
                current_idx = layers.index(self.current_layer)
                self.current_layer = layers[(current_idx + 1) % len(layers)]
            elif key == ord('s'):
                self.surgical_mode = not self.surgical_mode
            elif key == ord('d') and self.selected_organ and self.surgical_mode:
                self.perform_surgery(self.selected_organ, "dissect")
            elif key == ord('h') and self.selected_organ and self.surgical_mode:
                self.perform_surgery(self.selected_organ, "repair")
            elif key == ord('+') or key == ord('='):
                self.zoom = min(self.zoom + 0.1, 3.0)
            elif key == ord('-'):
                self.zoom = max(self.zoom - 0.1, 0.5)
            elif key == ord('r'):
                # Reset simulation
                self.rotation_x = 0
                self.rotation_y = 0
                self.zoom = 1.0
                self.selected_organ = None
                self.dissected_organs.clear()
                self.surgical_mode = False
                # Reset organ health
                for organ in self.organs.values():
                    organ['health'] = 100

        cap.release()
        cv2.destroyAllWindows()

    def draw_medical_info(self, frame, organ_name, position):
        """Enhanced medical information display"""
        medical_data = {
            'heart': {
                'name': 'HEART',
                'info': [
                    'Myocardium: Cardiac muscle tissue',
                    'Chambers: 4 (2 Atria, 2 Ventricles)',
                    'Function: Blood circulation',
                    'Rate: 60-100 bpm',
                    'Output: 5-6 L/min'
                ]
            },
            'brain': {
                'name': 'BRAIN',
                'info': [
                    'Weight: ~1.4 kg',
                    'Neurons: ~86 billion',
                    'Function: CNS control center',
                    'Oxygen use: 20% of total'
                ]
            },
            'lungs': {
                'name': 'LUNGS',
                'info': [
                    'Capacity: 4-6 liters',
                    'Alveoli: ~480 million',
                    'Function: Gas exchange',
                    'Surface area: ~70 m²'
                ]
            }
        }

        if organ_name in medical_data:
            data = medical_data[organ_name]
            x, y = position
            y_offset = 0

            # Draw info box
            box_height = len(data['info']) * 25 + 40
            cv2.rectangle(frame, (x + 120, y - 10),
                          (x + 450, y + box_height), (40, 40, 40), -1)
            cv2.rectangle(frame, (x + 120, y - 10),
                          (x + 450, y + box_height), (255, 255, 255), 2)

            # Title
            cv2.putText(frame, data['name'], (x + 130, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Info lines
            for i, line in enumerate(data['info']):
                cv2.putText(frame, line, (x + 130, y + 45 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame


if __name__ == "__main__":
    simulator = AnatomyLayer()
    simulator.run_enhanced_simulation()