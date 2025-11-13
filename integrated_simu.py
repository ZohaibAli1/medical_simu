# [file name]: integrated_medical_simulator.py
import cv2
import mediapipe as mp
import numpy as np
from unified_3d_anatomy import UnifiedAnatomySystem, UnifiedAnatomyController


class IntegratedMedicalSimulator:
    def __init__(self):
        # Initialize 3D anatomy system
        self.anatomy = UnifiedAnatomySystem()
        self.controller = UnifiedAnatomyController(self.anatomy)

        # Load complete anatomy
        self.anatomy.load_complete_anatomy()

        # MediaPipe for gesture control
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # Simulation state
        self.gesture_mode = True
        self.selected_organ = None
        self.show_labels = True

        # Performance tracking
        self.frame_count = 0
        self.fps = 0

    def detect_gestures(self, frame):
        """Detect hand gestures for controlling the 3D model"""
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get key points for gesture recognition
                landmarks = hand_landmarks.landmark

                thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                wrist = landmarks[self.mp_hands.HandLandmark.WRIST]

                # Convert to pixel coordinates
                thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_pos = (int(index_tip.x * w), int(index_tip.y * h))

                # Calculate distances for gesture recognition
                thumb_index_dist = np.sqrt((thumb_pos[0] - index_pos[0]) ** 2 +
                                           (thumb_pos[1] - index_pos[1]) ** 2)

                # Rotation gesture (open hand movement)
                if thumb_index_dist > 50:
                    # Use wrist movement for rotation
                    wrist_pos = (int(wrist.x * w), int(wrist.y * h))
                    if hasattr(self, 'prev_wrist_pos'):
                        dx = wrist_pos[0] - self.prev_wrist_pos[0]
                        dy = wrist_pos[1] - self.prev_wrist_pos[1]
                        self.controller.handle_rotation(dx * 0.5, dy * 0.5)

                    self.prev_wrist_pos = wrist_pos

                # Zoom gesture (pinch)
                else:
                    if thumb_index_dist < 30:
                        self.controller.handle_zoom(0.95)  # Zoom in
                    else:
                        self.controller.handle_zoom(1.05)  # Zoom out

        return frame

    def draw_ui_overlay(self, frame, anatomy_image):
        """Draw UI overlay on the combined view"""
        h, w = frame.shape[:2]

        # Resize anatomy image to fit alongside webcam
        anatomy_resized = cv2.resize(anatomy_image, (w // 2, h))

        # Combine webcam and anatomy view
        combined = np.hstack((frame, anatomy_resized))

        # Add UI elements
        cv2.putText(combined, f"Layer: {self.anatomy.current_layer.upper()}",
                    (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, f"FPS: {self.fps:.1f}",
                    (w + 20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Controls guide
        controls = [
            "1-5: Switch Layers",
            "R: Reset View",
            "G: Toggle Gestures",
            "L: Toggle Labels",
            "Q: Quit"
        ]

        for i, control in enumerate(controls):
            cv2.putText(combined, control,
                        (w + 20, 90 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return combined

    def update_performance_metrics(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        if hasattr(self, 'last_time'):
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_time)
                self.frame_count = 0
                self.last_time = current_time
        else:
            self.last_time = current_time

    def run(self):
        """Main simulation loop"""
        cap = cv2.VideoCapture(0)

        print("\n=== Integrated Medical Simulator ===")
        print("Features:")
        print("  - Complete 3D Anatomy System")
        print("  - Real-time Gesture Control")
        print("  - Layer-by-Layer Visualization")
        print("  - Anatomical Labels")
        print("  - Performance Optimized")
        print("====================================\n")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # Flip for mirror view
            frame = cv2.flip(frame, 1)

            # Process gestures if enabled
            if self.gesture_mode:
                frame = self.detect_gestures(frame)

            # Render 3D anatomy
            anatomy_image = self.anatomy.render_to_image()

            # Auto-rotate for demo
            self.anatomy.rotate_all(0, 0.5, 0)

            # Combine views and draw UI
            combined_view = self.draw_ui_overlay(frame, anatomy_image)

            # Display
            cv2.imshow('Integrated Medical Simulator', combined_view)

            # Update performance metrics
            self.update_performance_metrics()

            # Handle keyboard input
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.controller.switch_layer('skeleton')
            elif key == ord('2'):
                self.controller.switch_layer('organs')
            elif key == ord('3'):
                self.controller.switch_layer('muscles')
            elif key == ord('4'):
                self.controller.switch_layer('nerves')
            elif key == ord('5'):
                self.controller.switch_layer('skin')
            elif key == ord('r'):
                self.anatomy.reset_view()
            elif key == ord('g'):
                self.gesture_mode = not self.gesture_mode
            elif key == ord('l'):
                self.show_labels = not self.show_labels
                # Toggle label visibility
                for model_name in self.anatomy.models:
                    if model_name.startswith('label_'):
                        self.anatomy.set_model_visibility(model_name, self.show_labels)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    simulator = IntegratedMedicalSimulator()
    simulator.run()