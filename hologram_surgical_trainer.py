# ====================================
# HOLOGRAM SURGICAL TRAINING SYSTEM
# Complete Holographic Training with Hand & Instrument Detection
# ====================================

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

class InstrumentDetector:
    """Detect surgical instruments using color and shape recognition"""
    
    def __init__(self):
        self.instruments = {
            'scalpel': {'color_lower': np.array([0, 0, 150]), 'color_upper': np.array([180, 50, 255])},  # Silver/metallic
            'forceps': {'color_lower': np.array([0, 0, 120]), 'color_upper': np.array([180, 60, 200])},
            'cauterizer': {'color_lower': np.array([0, 100, 100]), 'color_upper': np.array([10, 255, 255])},  # Red tip
        }
        self.current_instrument = "none"
        
    def detect(self, frame):
        """Detect which instrument is in the frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        best_match = "none"
        max_area = 0
        
        for instrument, color_range in self.instruments.items():
            mask = cv2.inRange(hsv, color_range['color_lower'], color_range['color_upper'])
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > max_area and area > 500:  # Minimum size threshold
                    max_area = area
                    best_match = instrument
                    
                    # Draw bounding box
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{instrument.upper()}", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        self.current_instrument = best_match
        return best_match, frame


class HologramRenderer:
    """Create holographic display output for projection devices"""
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        
    def apply_hologram_effect(self, frame):
        """Apply holographic visual effects"""
        # Convert to cyan/blue tint
        hologram = np.zeros_like(frame)
        
        # Create cyan tinted version
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hologram[:, :, 0] = gray  # Blue
        hologram[:, :, 1] = gray  # Green
        hologram[:, :, 2] = gray * 0.3  # Red (minimal)
        
        # Add glow effect (Gaussian blur)
        glow = cv2.GaussianBlur(hologram, (21, 21), 0)
        hologram = cv2.addWeighted(hologram, 0.7, glow, 0.3, 0)
        
        # Add scanlines
        for i in range(0, self.height, 4):
            hologram[i:i+2, :] = hologram[i:i+2, :] * 0.8
        
        # Add edge detection overlay
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = np.zeros_like(hologram)
        edges_colored[edges > 0] = [255, 255, 0]  # Cyan edges
        
        hologram = cv2.addWeighted(hologram, 1.0, edges_colored, 0.3, 0)
        
        return hologram
    
    def create_stereo_view(self, left_frame, right_frame):
        """Create side-by-side stereo view for holographic displays"""
        stereo = np.hstack([left_frame, right_frame])
        return stereo
    
    def generate_hologram_output(self, anatomy_image, camera_frame):
        """Combine anatomy 3D view with camera feed for holographic projection"""
        
        # Apply holographic effect to anatomy
        holo_anatomy = self.apply_hologram_effect(anatomy_image)
        
        # Resize and prepare camera frame
        camera_small = cv2.resize(camera_frame, (self.width // 4, self.height // 4))
        
        # Create composite holographic display
        output = holo_anatomy.copy()
        
        # Place camera feed in corner (picture-in-picture)
        h, w = camera_small.shape[:2]
        output[10:10+h, 10:10+w] = camera_small
        
        # Add holographic grid overlay
        self.add_grid_overlay(output)
        
        return output
    
    def add_grid_overlay(self, frame):
        """Add futuristic grid lines"""
        h, w = frame.shape[:2]
        
        # Horizontal lines
        for y in range(0, h, 50):
            cv2.line(frame, (0, y), (w, y), (0, 255, 255), 1, cv2.LINE_AA)
        
        # Vertical lines
        for x in range(0, w, 50):
            cv2.line(frame, (x, 0), (x, h), (0, 255, 255), 1, cv2.LINE_AA)


class PerformanceMonitor:
    """Real-time surgical performance monitoring"""
    
    def __init__(self):
        self.start_time = time.time()
        self.actions = []
        self.mistakes = 0
        self.correct_actions = 0
        
    def record_action(self, action_type, is_correct, details=""):
        """Record each surgical action"""
        self.actions.append({
            'timestamp': time.time() - self.start_time,
            'type': action_type,
            'correct': is_correct,
            'details': details
        })
        
        if is_correct:
            self.correct_actions += 1
        else:
            self.mistakes += 1
    
    def get_live_stats(self):
        """Get current performance statistics"""
        total_time = time.time() - self.start_time
        total_actions = len(self.actions)
        accuracy = (self.correct_actions / total_actions * 100) if total_actions > 0 else 0
        
        return {
            'time': int(total_time),
            'actions': total_actions,
            'correct': self.correct_actions,
            'mistakes': self.mistakes,
            'accuracy': round(accuracy, 1)
        }
    
    def generate_report(self):
        """Generate final performance report"""
        stats = self.get_live_stats()
        
        # Calculate score (0-100)
        score = min(100, int(stats['accuracy'] * 0.7 + max(0, 100 - stats['mistakes'] * 10) * 0.3))
        
        feedback = ""
        if score >= 90:
            feedback = "Excellent! Professional level performance."
        elif score >= 75:
            feedback = "Good work! Keep practicing."
        elif score >= 60:
            feedback = "Fair performance. Review the procedure steps."
        else:
            feedback = "Needs improvement. Study and practice more."
        
        return {
            'score': score,
            'time': stats['time'],
            'accuracy': stats['accuracy'],
            'mistakes': stats['mistakes'],
            'feedback': feedback,
            'actions': self.actions
        }


class HologramSurgicalTrainer:
    """Main holographic surgical training system"""
    
    def __init__(self):
        self.state = AppState.MENU
        self.procedure_script = None
        self.assessment_module = None
        self.surgical_procedures = SurgicalProcedures()
        
        # 3D Anatomy System
        self.anatomy = UnifiedAnatomySystem()
        self.controller = UnifiedAnatomyController(self.anatomy)
        self.anatomy.load_complete_anatomy()
        
        # Hand Tracking
        self.hand_mapper = None
        self.interaction_detector = OrganInteractionDetector(self.anatomy)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Track both hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Instrument Detection
        self.instrument_detector = InstrumentDetector()
        
        # Hologram Rendering
        self.hologram_renderer = HologramRenderer()
        
        # Performance Monitoring
        self.performance_monitor = None
        
        # State
        self.active_tool = "none"
        self.hand_3d_position = np.array([0, 0, 0])
        self.current_feedback = "Welcome to Hologram Training Center"
        self.is_pinching = False
        self.last_pinch_time = 0
        self.target_organ_position = None
        
        # Menu
        self.menu_items = [
            {"name": "Appendectomy", "script": create_appendectomy_script},
            {"name": "Gallbladder Removal", "script": create_gallbladder_script}
        ]
        self.selected_menu_index = -1
    
    def get_required_organ(self):
        """Get target organ for current step"""
        if not self.procedure_script:
            return None
        step = self.procedure_script.get_current_step()
        if step and step.target_organ:
            return step.target_organ
        return None
    
    def process_hand_and_instrument(self, frame):
        """Process both hand tracking and instrument detection"""
        h, w = frame.shape[:2]
        if self.hand_mapper is None:
            self.hand_mapper = HandTo3DMapper(w, h, anatomy_bounds=200)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect instruments FIRST
        detected_instrument, frame = self.instrument_detector.detect(frame)
        
        # Then detect hands
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand skeleton
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2),
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                
                landmarks = hand_landmarks.landmark
                index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
                wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
                
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                
                # 3D position mapping
                hand_z = abs(wrist.z) if hasattr(wrist, 'z') else 0
                self.hand_3d_position = self.hand_mapper.map_2d_to_3d(cx, cy, hand_z)
                
                # Pinch detection
                pinch_distance = np.hypot(cx - tx, cy - ty)
                self.is_pinching = pinch_distance < 40
                
                # Visual feedback
                if self.is_pinching:
                    cv2.circle(frame, (cx, cy), 25, (0, 255, 0), 3)
                    cv2.putText(frame, "PINCH DETECTED", (cx + 30, cy),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.circle(frame, (cx, cy), 15, (255, 255, 0), 2)
                
                # State-specific logic
                if self.state == AppState.MENU:
                    self.handle_menu_interaction(frame, cx, cy)
                elif self.state == AppState.SURGERY:
                    self.handle_surgery_interaction(frame, cx, cy, hand_z, detected_instrument)
        else:
            self.current_feedback = "Place your hand in view..."
            self.interaction_detector.reset_highlights()
        
        # Update tool based on instrument detection
        if detected_instrument != "none" and self.state == AppState.SURGERY:
            self.active_tool = detected_instrument
        
        return frame
    
    def handle_menu_interaction(self, frame, cx, cy):
        """Menu selection with hand gestures"""
        h, w = frame.shape[:2]
        item_height = 80
        start_y = h // 2 - (len(self.menu_items) * item_height) // 2
        
        for i, item in enumerate(self.menu_items):
            y_pos = start_y + i * item_height
            if w//4 < cx < 3*w//4 and y_pos < cy < y_pos + 60:
                cv2.rectangle(frame, (w//4, y_pos), (3*w//4, y_pos + 60), (0, 255, 0), 3)
                if self.is_pinching and (time.time() - self.last_pinch_time > 1.0):
                    self.procedure_script = item["script"]()
                    self.assessment_module = AssessmentModule(self.procedure_script)
                    self.performance_monitor = PerformanceMonitor()
                    self.state = AppState.SURGERY
                    self.last_pinch_time = time.time()
                    print(f"✅ Selected: {item['name']}")
    
    def handle_surgery_interaction(self, frame, cx, cy, hand_z, detected_instrument):
        """Handle surgical interactions"""
        current_step = self.procedure_script.get_current_step()
        if not current_step:
            self.state = AppState.REPORT
            return
        
        # Check tool match
        req_tool = current_step.required_tool
        if req_tool and req_tool != "none":
            if detected_instrument != req_tool:
                self.current_feedback = f"⚠️ Wrong tool! Need {req_tool}. Detected: {detected_instrument}"
                return
            else:
                self.current_feedback = f"✅ Correct tool: {detected_instrument}"
        
        # Check organ interaction
        closest_organ, distance = self.interaction_detector.check_proximity(self.hand_3d_position)
        target_organ = self.get_required_organ()
        
        self.draw_hand_guidance(frame, (cx, cy), closest_organ, distance, target_organ)
        
        # Perform action on pinch
        if self.is_pinching and closest_organ:
            current_time = time.time()
            if current_time - self.last_pinch_time > 0.5:
                self.handle_pinch_action(closest_organ, distance, detected_instrument)
                self.last_pinch_time = current_time
    
    def handle_pinch_action(self, organ_name, distance, instrument):
        """Handle surgical action"""
        current_step = self.procedure_script.get_current_step()
        if not current_step or distance > 30:
            return
        
        action_data = {
            'action': current_step.required_action,
            'target': organ_name,
            'tool': instrument
        }
        
        is_correct, feedback = self.assessment_module.check_action(action_data)
        self.current_feedback = feedback
        
        # Record performance
        self.performance_monitor.record_action(
            action_type=current_step.required_action.value,
            is_correct=is_correct,
            details=f"Tool: {instrument}, Target: {organ_name}"
        )
        
        if is_correct:
            print(f"✅ Correct action on {organ_name}")
        else:
            print(f"❌ Incorrect: {feedback}")
    
    def draw_hand_guidance(self, frame, hand_pos, closest_organ, distance, target_organ):
        """Visual guidance overlay"""
        h, w = frame.shape[:2]
        cx, cy = hand_pos
        
        # Crosshair
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 2)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 2)
        
        # Distance indicator
        if closest_organ and distance < 50:
            color = (0, 255, 0) if closest_organ == target_organ else (0, 165, 255)
            label = f"{closest_organ} {'✓' if closest_organ == target_organ else '✗'}"
            
            cv2.putText(frame, label, (cx - 40, cy - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"{int(distance)}mm", (cx - 40, cy - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_ui(self, frame, anatomy_image):
        """Draw complete UI with holographic effects"""
        h, w = frame.shape[:2]
        
        # Apply holographic effect to anatomy
        if anatomy_image is not None and anatomy_image.size > 0:
            target_h = int(h * 0.6)
            scale = target_h / anatomy_image.shape[0]
            target_w = int(anatomy_image.shape[1] * scale)
            
            anatomy_resized = cv2.resize(anatomy_image, (target_w, target_h))
            
            # Holographic rendering
            holo_anatomy = self.hologram_renderer.apply_hologram_effect(anatomy_resized)
            
            y_offset = h - target_h - 150
            x_offset = (w - target_w) // 2
            
            # Overlay hologram
            frame[y_offset:y_offset+target_h, x_offset:x_offset+target_w] = holo_anatomy
        
        # Header
        cv2.rectangle(frame, (0, 0), (w, 60), (10, 10, 30), -1)
        cv2.putText(frame, "HOLOGRAM SURGICAL TRAINING", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if self.state == AppState.MENU:
            self.draw_menu_ui(frame)
        elif self.state == AppState.SURGERY:
            self.draw_surgery_ui(frame)
        elif self.state == AppState.REPORT:
            self.draw_report_ui(frame)
        
        return frame
    
    def draw_menu_ui(self, frame):
        """Menu interface"""
        h, w = frame.shape[:2]
        cv2.putText(frame, "SELECT PROCEDURE (Pinch to Start)", (w//2 - 300, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        item_height = 80
        start_y = h // 2 - (len(self.menu_items) * item_height) // 2
        
        for i, item in enumerate(self.menu_items):
            y_pos = start_y + i * item_height
            cv2.rectangle(frame, (w//4, y_pos), (3*w//4, y_pos + 60), (0, 150, 150), -1)
            cv2.rectangle(frame, (w//4, y_pos), (3*w//4, y_pos + 60), (0, 255, 255), 2)
            cv2.putText(frame, item["name"], (w//4 + 20, y_pos + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def draw_surgery_ui(self, frame):
        """Surgery interface with live stats"""
        h, w = frame.shape[:2]
        step = self.procedure_script.get_current_step()
        
        if step:
            # Instruction box
            cv2.rectangle(frame, (10, h - 150), (w - 10, h - 10), (10, 20, 40), -1)
            cv2.rectangle(frame, (10, h - 150), (w - 10, h - 10), (0, 255, 255), 2)
            
            cv2.putText(frame, f"STEP {step.step_id}: {step.description}",
                       (25, h - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if step.target_organ:
                cv2.putText(frame, f"TARGET: {step.target_organ.upper()}",
                           (25, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.putText(frame, f"TOOL: {self.active_tool.upper()}",
                       (25, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Live performance stats
        if self.performance_monitor:
            stats = self.performance_monitor.get_live_stats()
            
            cv2.putText(frame, f"Time: {stats['time']}s", (w - 250, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Accuracy: {stats['accuracy']}%", (w - 250, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Mistakes: {stats['mistakes']}", (w - 250, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Feedback
        cv2.putText(frame, self.current_feedback, (20, h - 165),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def draw_report_ui(self, frame):
        """Final report screen"""
        h, w = frame.shape[:2]
        
        if self.performance_monitor:
            report = self.performance_monitor.generate_report()
            
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (10, 20, 40), -1)
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 255), 3)
            
            cv2.putText(frame, "TRAINING COMPLETE", (w//2 - 180, h//4 + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            
            cv2.putText(frame, f"SCORE: {report['score']}/100", (w//2 - 120, h//2 - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            cv2.putText(frame, f"Time: {report['time']}s", (w//2 - 80, h//2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Accuracy: {report['accuracy']}%", (w//2 - 100, h//2 + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Mistakes: {report['mistakes']}", (w//2 - 100, h//2 + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.putText(frame, report['feedback'], (w//2 - 200, 3*h//4 - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, "Press 'Q' to exit", (w//2 - 100, 3*h//4 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run(self):
        """Main training loop"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera failed to open")
            return
        
        # Set high resolution for better detection
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.anatomy.camera.Zoom(1.5)
        
        print("🎯 Holographic Surgical Training System Starting...")
        print("Controls:")
        print("  - Show surgical instrument to camera for detection")
        print("  - Make pinch gesture (thumb + index) to interact")
        print("  - Z: Zoom In, X: Zoom Out")
        print("  - Q: Quit")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
            
            frame = cv2.flip(frame, 1)
            
            # Auto-rotate anatomy
            self.anatomy.rotate_all(0, 0.3, 0)
            
            # Focus camera on target organ
            target_organ = self.get_required_organ()
            if target_organ:
                self.anatomy.focus_on_organ(target_organ)
            else:
                self.anatomy.reset_view()
                self.anatomy.camera.Zoom(1.2)
            
            # Render 3D anatomy
            anatomy_image = self.anatomy.render_to_image()
            
            # Process hand + instrument detection
            frame = self.process_hand_and_instrument(frame)
            
            # Create holographic display
            display = self.draw_ui(frame, anatomy_image)
            
            # Show output
            cv2.imshow('Holographic Surgical Training', display)
            
            # Optional: Show hologram-ready output in separate window
            if anatomy_image is not None:
                holo_output = self.hologram_renderer.generate_hologram_output(anatomy_image, frame)
                cv2.imshow('Hologram Projection Output', holo_output)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('z'):
                self.anatomy.camera.Zoom(1.1)
            elif key == ord('x'):
                self.anatomy.camera.Zoom(0.9)
            elif key == ord('r') and self.state == AppState.REPORT:
                self.state = AppState.MENU
        
        cap.release()
        cv2.destroyAllWindows()


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
        for organ_data in self.anatomy.models.values():
            actor = organ_data['actor']
            actor.GetProperty().SetAmbient(0.1)
            actor.GetProperty().SetDiffuse(0.6)


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


if __name__ == "__main__":
    trainer = HologramSurgicalTrainer()
    trainer.run()
