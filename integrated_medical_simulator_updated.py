# enhanced_hand_3d_interaction.py
import cv2
import mediapipe as mp
import numpy as np
import time
from enum import Enum
from unified_3d_anatomy import UnifiedAnatomySystem, UnifiedAnatomyController
from procedure_scripting import create_appendectomy_script, create_gallbladder_script, ProcedureAction
from assessment_module import AssessmentModule
from surgical_procedures import SurgicalProcedures

class AppState(Enum):
    MENU = 1
    TOOL_SELECT = 2
    SURGERY = 3
    REPORT = 4

class HandTo3DMapper:

    def __init__(self, frame_width, frame_height, anatomy_bounds):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.anatomy_bounds = anatomy_bounds

    def map_2d_to_3d(self, hand_x, hand_y, hand_z=0):
        norm_x = hand_x / self.frame_width
        norm_y = hand_y / self.frame_height
        x_3d = (norm_x - 0.5) * 200
        y_3d = (norm_y - 0.5) * 200
        z_3d = 100 + hand_z * 100
        return np.array([x_3d, y_3d, z_3d])


class OrganInteractionDetector:

    def __init__(self, anatomy_system):
        self.anatomy = anatomy_system
        self.interaction_radius = 30
        self.highlighted_organ = None

    def check_proximity(self, hand_3d_pos):
        closest_organ = None
        min_distance = float('inf')

        for organ_name, organ_data in self.anatomy.models.items():
            if not organ_data['visible']:
                continue

            actor = organ_data['actor']
            organ_pos = np.array(actor.GetPosition())
            distance = np.linalg.norm(hand_3d_pos - organ_pos)

            if distance < self.interaction_radius and distance < min_distance:
                min_distance = distance
                closest_organ = organ_name
        if closest_organ != self.highlighted_organ:
            self.reset_highlights()
            if closest_organ:
                self.highlight_organ(closest_organ)
            self.highlighted_organ = closest_organ

        return closest_organ, min_distance

    def highlight_organ(self, organ_name):
        if organ_name in self.anatomy.models:
            actor = self.anatomy.models[organ_name]['actor']
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
        self.state = AppState.MENU
        self.procedure_script = None
        self.assessment_module = None
        self.surgical_procedures = SurgicalProcedures()
        self.anatomy = UnifiedAnatomySystem()
        self.controller = UnifiedAnatomyController(self.anatomy)
        self.anatomy.load_complete_anatomy()

        self.hand_mapper = None
        self.interaction_detector = OrganInteractionDetector(self.anatomy)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.active_tool = "none"
        self.hand_3d_position = np.array([0, 0, 0])
        self.current_feedback = "Welcome to Hologram Training Center"
        self.is_pinching = False
        self.last_pinch_time = 0
        
        self.target_organ_position = None
        self.show_3d_cursor = True
        
        # Menu items
        self.menu_items = [
            {"name": "Appendectomy", "script": create_appendectomy_script},
            {"name": "Gallbladder Removal", "script": create_gallbladder_script}
        ]
        self.selected_menu_index = -1
        
        # Tool Tray
        self.tools = ["none", "scalpel", "forceps", "cauterizer", "needle_driver"]
        self.tool_icons = {
            "none": "Hand",
            "scalpel": "Scalpel",
            "forceps": "Forceps",
            "cauterizer": "Cauterizer",
            "needle_driver": "Needle"
        }

    def get_required_organ(self):
        if not self.procedure_script: return None
        step = self.procedure_script.get_current_step()
        if step and step.target_organ:
            return step.target_organ
        return None

    def process_hand_interaction(self, frame):
        h, w = frame.shape[:2]
        if self.hand_mapper is None:
            self.hand_mapper = HandTo3DMapper(w, h, anatomy_bounds=200)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

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
                
                # Pinch detection
                pinch_distance = np.hypot(cx - tx, cy - ty)
                self.is_pinching = pinch_distance < 40
                
                if self.is_pinching:
                    cv2.circle(frame, (cx, cy), 20, (0, 255, 0), 3)
                else:
                    cv2.circle(frame, (cx, cy), 15, (255, 255, 0), 2)

                # State-specific logic
                if self.state == AppState.MENU:
                    self.handle_menu_interaction(frame, cx, cy)
                elif self.state == AppState.TOOL_SELECT:
                    self.handle_tool_selection(frame, cx, cy)
                elif self.state == AppState.SURGERY:
                    self.handle_surgery_interaction(frame, cx, cy, hand_z)

        else:
            self.current_feedback = "Place your hand in view..."
            self.interaction_detector.reset_highlights()

        return frame

    def handle_menu_interaction(self, frame, cx, cy):
        h, w = frame.shape[:2]
        item_height = 80
        start_y = h // 2 - (len(self.menu_items) * item_height) // 2
        
        for i, item in enumerate(self.menu_items):
            y_pos = start_y + i * item_height
            # Check hover
            if w//4 < cx < 3*w//4 and y_pos < cy < y_pos + 60:
                cv2.rectangle(frame, (w//4, y_pos), (3*w//4, y_pos + 60), (0, 255, 0), 2)
                if self.is_pinching and (time.time() - self.last_pinch_time > 1.0):
                    self.procedure_script = item["script"]()
                    self.assessment_module = AssessmentModule(self.procedure_script)
                    self.state = AppState.SURGERY
                    self.last_pinch_time = time.time()
                    print(f"Selected: {item['name']}")

    def handle_tool_selection(self, frame, cx, cy):
        h, w = frame.shape[:2]
        # Draw tool tray
        tray_y = h - 150
        slot_width = w // len(self.tools)
        
        for i, tool in enumerate(self.tools):
            x_pos = i * slot_width
            # Check hover
            if x_pos < cx < x_pos + slot_width and tray_y < cy < h:
                cv2.rectangle(frame, (x_pos, tray_y), (x_pos + slot_width, h), (0, 255, 0), 2)
                if self.is_pinching and (time.time() - self.last_pinch_time > 0.5):
                    self.active_tool = tool
                    self.state = AppState.SURGERY
                    self.last_pinch_time = time.time()
                    self.current_feedback = f"Equipped: {tool}"
            
    def handle_surgery_interaction(self, frame, cx, cy, hand_z):
        # Check if tool change is needed
        current_step = self.procedure_script.get_current_step()
        if current_step:
            req_tool = current_step.required_tool
            if req_tool and req_tool != "none" and self.active_tool != req_tool:
                self.current_feedback = f"WRONG TOOL! Need {req_tool}. Pinch bottom to switch."
                # Check if hand is at bottom to switch tool
                if cy > frame.shape[0] - 100 and self.is_pinching:
                     self.state = AppState.TOOL_SELECT
                return

        # Normal surgery interaction
        closest_organ, distance = self.interaction_detector.check_proximity(self.hand_3d_position)
        target_organ = self.get_required_organ()
        
        self.draw_hand_guidance(frame, (cx, cy), closest_organ, distance, target_organ)
        
        # Render tool overlay
        if self.active_tool != "none":
            self.surgical_procedures.render_tool(frame, self.active_tool, (cx, cy))

        if self.is_pinching:
            current_time = time.time()
            if current_time - self.last_pinch_time > 0.5:
                self.handle_pinch_action(closest_organ, distance)
                self.last_pinch_time = current_time
                
        # Update feedback based on step
        if current_step:
             if target_organ and closest_organ:
                if closest_organ == target_organ:
                    if distance < 20:
                        self.current_feedback = f"✓ CORRECT! Pinch to {current_step.required_action.value}"
                    else:
                        self.current_feedback = f"Move closer to {target_organ}"
                else:
                    self.current_feedback = f"Wrong organ! Need: {target_organ}"
             elif target_organ:
                self.current_feedback = f"Find and point at: {target_organ}"

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

    def apply_hologram_effect(self, anatomy_img):
        """Apply sci-fi hologram style to the anatomy render"""
        if anatomy_img is None or anatomy_img.size == 0:
            return None
            
        # 1. Create mask of the anatomy (non-black pixels)
        gray = cv2.cvtColor(anatomy_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # 2. Colorize to Cyan/Blue (Hologram tint)
        # Create a cyan version of the image
        hologram = np.zeros_like(anatomy_img)
        hologram[:, :, 0] = 255  # Blue channel
        hologram[:, :, 1] = 255  # Green channel
        hologram[:, :, 2] = 0    # Red channel
        
        # Blend original intensity with cyan tint
        hologram = cv2.bitwise_and(hologram, hologram, mask=mask)
        
        # 3. Add Scanlines
        rows, cols = anatomy_img.shape[:2]
        scanline_mask = np.zeros((rows, cols), dtype=np.uint8)
        scanline_mask[::4, :] = 255  # Every 4th line
        
        # Darken scanlines
        hologram[scanline_mask == 0] = (hologram[scanline_mask == 0] * 0.7).astype(np.uint8)
        
        return hologram, mask

    def draw_ui(self, frame, anatomy_image):
        """Enhanced UI with Holographic AR Overlay"""
        h, w = frame.shape[:2]
        
        # --- Holographic Overlay ---
        if anatomy_image is not None and anatomy_image.size > 0:
            # Resize anatomy to fill the screen (or center it)
            # We'll make it cover the center 80%
            target_h = int(h * 0.8)
            scale = target_h / anatomy_image.shape[0]
            target_w = int(anatomy_image.shape[1] * scale)
            
            anatomy_resized = cv2.resize(anatomy_image, (target_w, target_h))
            
            # Calculate centering offsets
            y_offset = (h - target_h) // 2
            x_offset = (w - target_w) // 2
            
            # Apply Hologram Effect
            holo_img, holo_mask = self.apply_hologram_effect(anatomy_resized)
            
            # Overlay onto frame
            roi = frame[y_offset:y_offset+target_h, x_offset:x_offset+target_w]
            
            # Create inverse mask
            mask_inv = cv2.bitwise_not(holo_mask)
            
            # Black-out area of anatomy in ROI
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            
            # Take only region of anatomy from hologram image
            img_fg = cv2.bitwise_and(holo_img, holo_img, mask=holo_mask)
            
            # Add (blend) the two
            # We use addWeighted for transparency/glow effect
            dst = cv2.addWeighted(img_bg, 1.0, img_fg, 0.8, 0) # 0.8 opacity for hologram
            
            # Put back in frame
            frame[y_offset:y_offset+target_h, x_offset:x_offset+target_w] = dst
            
            # Update 3D cursor logic to match new coordinates
            if self.show_3d_cursor:
                # We need to map 3D pos to this new centered viewport
                # This is tricky without full projection matrix, but we'll approximate
                # The anatomy is centered in the view
                pass 

        # Header
        cv2.rectangle(frame, (0, 0), (w, 60), (40, 40, 40), -1)
        cv2.putText(frame, "HOLOGRAM TRAINING CENTER", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if self.state == AppState.MENU:
            self.draw_menu_ui(frame)
        elif self.state == AppState.TOOL_SELECT:
            self.draw_tool_select_ui(frame)
        elif self.state == AppState.SURGERY:
            self.draw_surgery_ui(frame)

        return frame

    def project_3d_to_2d(self, pos_3d, frame_w, frame_h):
        """Simple 3D to 2D projection for guidance"""
        # Normalize 3D position (-100 to 100) to screen space
        # Adjusted for the centered overlay logic
        # This is a rough approximation since we are overlaying a 2D render
        x_norm = (pos_3d[0] + 100) / 200
        y_norm = (pos_3d[1] + 100) / 200

        # Assume anatomy is centered and takes up 80% height
        target_h = int(frame_h * 0.8)
        # Aspect ratio of render window is 800x600 (4:3)
        target_w = int(target_h * (800/600)) 
        
        x_offset = (frame_w - target_w) // 2
        y_offset = (frame_h - target_h) // 2

        x_2d = int(x_norm * target_w) + x_offset
        y_2d = int(y_norm * target_h) + y_offset

        # Keep within bounds
        x_2d = max(0, min(frame_w - 1, x_2d))
        y_2d = max(0, min(frame_h - 1, y_2d))

        return (x_2d, y_2d)

    def draw_menu_ui(self, frame):
        h, w = frame.shape[:2]
        cv2.putText(frame, "SELECT PROCEDURE (Pinch to Start)", (w//2 - 200, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        item_height = 80
        start_y = h // 2 - (len(self.menu_items) * item_height) // 2
        
        for i, item in enumerate(self.menu_items):
            y_pos = start_y + i * item_height
            cv2.rectangle(frame, (w//4, y_pos), (3*w//4, y_pos + 60), (50, 50, 50), -1)
            cv2.rectangle(frame, (w//4, y_pos), (3*w//4, y_pos + 60), (0, 255, 255), 2)
            cv2.putText(frame, item["name"], (w//4 + 20, y_pos + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def draw_tool_select_ui(self, frame):
        h, w = frame.shape[:2]
        cv2.putText(frame, "SELECT TOOL (Pinch to Equip)", (w//2 - 150, h - 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        tray_y = h - 150
        slot_width = w // len(self.tools)
        
        for i, tool in enumerate(self.tools):
            x_pos = i * slot_width
            color = (0, 255, 0) if tool == self.active_tool else (100, 100, 100)
            cv2.rectangle(frame, (x_pos, tray_y), (x_pos + slot_width, h), (50, 50, 50), -1)
            cv2.rectangle(frame, (x_pos, tray_y), (x_pos + slot_width, h), color, 2)
            cv2.putText(frame, self.tool_icons[tool], (x_pos + 10, tray_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_surgery_ui(self, frame):
        h, w = frame.shape[:2]
        step = self.procedure_script.get_current_step()
        if step:
            # Instruction box
            cv2.rectangle(frame, (10, h - 120), (w - 10, h - 10), (30, 30, 30), -1)
            cv2.rectangle(frame, (10, h - 120), (w - 10, h - 10), (0, 255, 255), 2)

            cv2.putText(frame, f"STEP {step.step_id}: {step.description}",
                        (25, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if step.target_organ:
                cv2.putText(frame, f"TARGET: {step.target_organ.upper()}",
                            (25, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            tool_text = f"TOOL: {self.active_tool.upper()}"
            tool_color = (0, 255, 0) if self.active_tool == step.required_tool else (0, 0, 255)
            cv2.putText(frame, tool_text, (w - 250, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tool_color, 2)

        else:
            cv2.putText(frame, "✓ PROCEDURE COMPLETE!",
                        (w // 2 - 150, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Feedback overlay
        feedback_color = (0, 255, 0) if "✓" in self.current_feedback else (0, 165, 255)
        cv2.putText(frame, self.current_feedback, (20, h - 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera failed to open")
            return

        # Zoom in for better visibility
        self.anatomy.camera.Zoom(1.5)

        print("🎯 Enhanced Surgical Training System Starting...")
        print("Controls:")
        print("  - Point at organs to highlight them")
        print("  - Pinch (thumb + index) to interact")
        print("  - Z: Zoom In, X: Zoom Out")
        print("  - Q to quit")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)

            # Auto-rotate anatomy for better view
            self.anatomy.rotate_all(0, 0.3, 0)
            
            # Dynamic Camera Focus
            target_organ = self.get_required_organ()
            if target_organ:
                self.anatomy.focus_on_organ(target_organ)
            else:
                # Default view if no target
                self.anatomy.reset_view()
                self.anatomy.camera.Zoom(1.2) # Slight zoom for general view

            # Render 3D anatomy
            anatomy_image = self.anatomy.render_to_image()

            # Process interactions
            frame = self.process_hand_interaction(frame)

            # Draw combined UI
            display = self.draw_ui(frame, anatomy_image)

            cv2.imshow('Enhanced Surgical Training', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('z'):
                self.anatomy.camera.Zoom(1.1)
            elif key == ord('x'):
                self.anatomy.camera.Zoom(0.9)

        # Show final report
        if self.assessment_module:
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