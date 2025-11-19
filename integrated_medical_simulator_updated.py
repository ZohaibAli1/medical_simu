# enhanced_hand_3d_interaction.py
import cv2
import mediapipe as mp
import numpy as np
import time
from unified_3d_anatomy import UnifiedAnatomySystem, UnifiedAnatomyController
from procedure_scripting import create_appendectomy_script, ProcedureAction
from assessment_module import AssessmentModule
from surgical_procedures import SurgicalProcedures


class HandTo3DMapper:
    """Maps 2D hand position to 3D anatomy coordinates"""

    def __init__(self, frame_width, frame_height, anatomy_bounds):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.anatomy_bounds = anatomy_bounds  # 3D space boundaries

    def map_2d_to_3d(self, hand_x, hand_y, hand_z=0):
        """Convert 2D screen coordinates to 3D anatomy space"""
        # Normalize screen coordinates (0 to 1)
        norm_x = hand_x / self.frame_width
        norm_y = hand_y / self.frame_height

        # Map to 3D space
        # X: -100 to 100 (left-right)
        # Y: -100 to 100 (front-back)
        # Z: 0 to 200 (bottom-top)
        x_3d = (norm_x - 0.5) * 200  # -100 to 100
        y_3d = (norm_y - 0.5) * 200
        z_3d = 100 + hand_z * 100  # Depth from hand tracking

        return np.array([x_3d, y_3d, z_3d])


class OrganInteractionDetector:
    """Detects when hand is near/touching organs"""

    def __init__(self, anatomy_system):
        self.anatomy = anatomy_system
        self.interaction_radius = 30  # Detection radius
        self.highlighted_organ = None

    def check_proximity(self, hand_3d_pos):
        """Check if hand is near any organ"""
        closest_organ = None
        min_distance = float('inf')

        for organ_name, organ_data in self.anatomy.models.items():
            if not organ_data['visible']:
                continue

            actor = organ_data['actor']
            organ_pos = np.array(actor.GetPosition())

            # Calculate distance
            distance = np.linalg.norm(hand_3d_pos - organ_pos)

            if distance < self.interaction_radius and distance < min_distance:
                min_distance = distance
                closest_organ = organ_name

        # Highlight closest organ
        if closest_organ != self.highlighted_organ:
            self.reset_highlights()
            if closest_organ:
                self.highlight_organ(closest_organ)
            self.highlighted_organ = closest_organ

        return closest_organ, min_distance

    def highlight_organ(self, organ_name):
        """Highlight the organ being pointed at"""
        if organ_name in self.anatomy.models:
            actor = self.anatomy.models[organ_name]['actor']
            # Make it brighter/outlined
            actor.GetProperty().SetAmbient(0.5)
            actor.GetProperty().SetDiffuse(0.8)

    def reset_highlights(self):
        """Remove all highlights"""
        for organ_data in self.anatomy.models.values():
            actor = organ_data['actor']
            actor.GetProperty().SetAmbient(0.1)
            actor.GetProperty().SetDiffuse(0.6)


class EnhancedMedicalSimulator:
    def __init__(self):
        # Core systems
        self.procedure_script = create_appendectomy_script()
        self.assessment_module = AssessmentModule(self.procedure_script)
        self.surgical_procedures = SurgicalProcedures()

        # 3D Anatomy
        self.anatomy = UnifiedAnatomySystem()
        self.controller = UnifiedAnatomyController(self.anatomy)
        self.anatomy.load_complete_anatomy()

        # NEW: Interaction systems
        self.hand_mapper = None  # Initialize after camera
        self.interaction_detector = OrganInteractionDetector(self.anatomy)

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # State
        self.active_tool = "none"
        self.hand_3d_position = np.array([0, 0, 0])
        self.current_feedback = "Place your hand in view..."
        self.is_pinching = False
        self.last_pinch_time = 0

        # Visual guides
        self.target_organ_position = None
        self.show_3d_cursor = True

    def get_required_organ(self):
        """Get the organ needed for current step"""
        step = self.procedure_script.get_current_step()
        if step and step.target_organ:
            return step.target_organ
        return None

    def process_hand_interaction(self, frame):
        h, w = frame.shape[:2]

        # Initialize mapper if not done
        if self.hand_mapper is None:
            self.hand_mapper = HandTo3DMapper(w, h, anatomy_bounds=200)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        current_step = self.procedure_script.get_current_step()
        target_organ = self.get_required_organ()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark

                # Get hand positions
                index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
                wrist = landmarks[self.mp_hands.HandLandmark.WRIST]

                cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

                # Estimate depth from hand size
                hand_z = abs(wrist.z) if hasattr(wrist, 'z') else 0

                # Map to 3D space
                self.hand_3d_position = self.hand_mapper.map_2d_to_3d(cx, cy, hand_z)

                # Check proximity to organs
                closest_organ, distance = self.interaction_detector.check_proximity(
                    self.hand_3d_position
                )

                # Draw 2D visualization
                self.draw_hand_guidance(frame, (cx, cy), closest_organ, distance, target_organ)

                # Pinch detection
                pinch_distance = np.hypot(cx - tx, cy - ty)
                self.is_pinching = pinch_distance < 40

                if self.is_pinching:
                    cv2.circle(frame, (cx, cy), 20, (0, 255, 0), 3)

                    # Handle pinch action
                    current_time = time.time()
                    if current_time - self.last_pinch_time > 0.5:  # Debounce
                        self.handle_pinch_action(closest_organ, distance)
                        self.last_pinch_time = current_time
                else:
                    cv2.circle(frame, (cx, cy), 15, (255, 255, 0), 2)

                # Update feedback
                if target_organ and closest_organ:
                    if closest_organ == target_organ:
                        if distance < 20:
                            self.current_feedback = f"✓ CORRECT! Pinch to select {target_organ}"
                        else:
                            self.current_feedback = f"Move closer to {target_organ}"
                    else:
                        self.current_feedback = f"Wrong organ! Need: {target_organ}"
                elif target_organ:
                    self.current_feedback = f"Find and point at: {target_organ}"

        else:
            self.current_feedback = "Place your hand in view..."
            self.interaction_detector.reset_highlights()

        return frame

    def draw_hand_guidance(self, frame, hand_pos, closest_organ, distance, target_organ):
        """Draw visual guides showing hand-organ relationship"""
        h, w = frame.shape[:2]
        cx, cy = hand_pos

        # Draw crosshair at hand position
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 2)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 2)

        # If near an organ, show distance indicator
        if closest_organ and distance < 50:
            # Color changes based on correctness
            if target_organ and closest_organ == target_organ:
                color = (0, 255, 0)  # Green for correct
                label = f"{closest_organ} ✓"
            else:
                color = (0, 165, 255)  # Orange for wrong
                label = f"{closest_organ} ✗"

            # Distance bar
            bar_length = int(max(0, 100 - distance * 2))
            cv2.rectangle(frame, (cx - 50, cy - 40), (cx - 50 + bar_length, cy - 30), color, -1)
            cv2.rectangle(frame, (cx - 50, cy - 40), (cx + 50, cy - 30), (255, 255, 255), 2)

            # Label
            cv2.putText(frame, label, (cx - 40, cy - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Distance text
            cv2.putText(frame, f"{int(distance)}mm", (cx - 40, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw target indicator if we have a target organ
        if target_organ and target_organ in self.anatomy.models:
            # Show arrow pointing to general direction of target
            organ_data = self.anatomy.models[target_organ]
            organ_pos_3d = np.array(organ_data['actor'].GetPosition())

            # Project 3D to 2D (simplified)
            target_2d = self.project_3d_to_2d(organ_pos_3d, w, h)

            if target_2d:
                tx, ty = target_2d
                # Draw arrow from hand to target
                cv2.arrowedLine(frame, (cx, cy), (tx, ty), (0, 255, 0), 2, tipLength=0.2)
                cv2.circle(frame, (tx, ty), 10, (0, 255, 0), 2)
                cv2.putText(frame, "TARGET", (tx + 15, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def project_3d_to_2d(self, pos_3d, frame_w, frame_h):
        """Simple 3D to 2D projection for guidance"""
        # Normalize 3D position (-100 to 100) to screen space
        x_norm = (pos_3d[0] + 100) / 200
        y_norm = (pos_3d[1] + 100) / 200

        x_2d = int(x_norm * frame_w)
        y_2d = int(y_norm * frame_h)

        # Keep within bounds
        x_2d = max(0, min(frame_w - 1, x_2d))
        y_2d = max(0, min(frame_h - 1, y_2d))

        return (x_2d, y_2d)

    def handle_pinch_action(self, organ_name, distance):
        """Handle pinch gesture on an organ"""
        if not organ_name or distance > 30:
            return

        current_step = self.procedure_script.get_current_step()
        if not current_step:
            return

        # Create action data
        action_data = {
            'action': current_step.required_action,
            'target': organ_name,
            'tool': self.active_tool
        }

        # Validate with assessment module
        is_correct, feedback = self.assessment_module.check_action(action_data)
        self.current_feedback = feedback

        # Visual feedback
        if is_correct:
            print(f"✓ Correct action on {organ_name}")
        else:
            print(f"✗ Incorrect: {feedback}")

    def draw_ui(self, frame, anatomy_image):
        """Enhanced UI with 3D alignment"""
        h, w = frame.shape[:2]

        # Side-by-side view
        if anatomy_image is not None and anatomy_image.size > 0:
            anatomy_resized = cv2.resize(anatomy_image, (w // 2, h))

            # Draw 3D cursor on anatomy view
            if self.show_3d_cursor:
                cursor_2d = self.project_3d_to_2d(self.hand_3d_position, w // 2, h)
                if cursor_2d:
                    cv2.circle(anatomy_resized, cursor_2d, 8, (0, 255, 255), -1)
                    cv2.circle(anatomy_resized, cursor_2d, 12, (255, 255, 255), 2)

            combined = np.hstack((frame, anatomy_resized))
        else:
            combined = frame

        cw = combined.shape[1]

        # Header
        cv2.rectangle(combined, (0, 0), (cw, 60), (40, 40, 40), -1)
        cv2.putText(combined, "INTERACTIVE SURGICAL TRAINING", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Step info
        step = self.procedure_script.get_current_step()
        if step:
            # Instruction box
            cv2.rectangle(combined, (10, h - 120), (cw - 10, h - 10), (30, 30, 30), -1)
            cv2.rectangle(combined, (10, h - 120), (cw - 10, h - 10), (0, 255, 255), 2)

            cv2.putText(combined, f"STEP {step.step_id}: {step.description}",
                        (25, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if step.target_organ:
                cv2.putText(combined, f"TARGET: {step.target_organ.upper()}",
                            (25, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(combined, "👆 Point at organ | ✊ Pinch to interact",
                        (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(combined, "✓ PROCEDURE COMPLETE!",
                        (cw // 2 - 150, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Feedback overlay
        feedback_color = (0, 255, 0) if "✓" in self.current_feedback else (0, 165, 255)
        cv2.putText(combined, self.current_feedback, (20, h - 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2)

        # 3D position indicator
        pos_text = f"3D Pos: ({int(self.hand_3d_position[0])}, {int(self.hand_3d_position[1])}, {int(self.hand_3d_position[2])})"
        cv2.putText(combined, pos_text, (cw - 300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return combined

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera failed to open")
            return

        print("🎯 Enhanced Surgical Training System Starting...")
        print("Controls:")
        print("  - Point at organs to highlight them")
        print("  - Pinch (thumb + index) to interact")
        print("  - Q to quit")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)

            # Auto-rotate anatomy for better view
            self.anatomy.rotate_all(0, 0.3, 0)

            # Render 3D anatomy
            anatomy_image = self.anatomy.render_to_image()

            # Process interactions
            frame = self.process_hand_interaction(frame)

            # Draw combined UI
            display = self.draw_ui(frame, anatomy_image)

            cv2.imshow('Enhanced Surgical Training', display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Show final report
        report = self.assessment_module.finalize_assessment()
        print("\n" + "=" * 50)
        print("TRAINING SESSION COMPLETE")
        print("=" * 50)
        print(f"Procedure: {report['procedure_name']}")
        print(f"Score: {report['final_score']}/100")
        print(f"Time: {report['total_time_seconds']:.1f}s")
        print(f"Mistakes: {report['mistake_count']}")
        print("=" * 50)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sim = EnhancedMedicalSimulator()
    sim.run()