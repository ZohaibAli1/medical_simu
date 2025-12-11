"""
Simplified Hologram Surgical Trainer
====================================

A holographic surgical training system that combines computer vision,
hand tracking, and 3D visualization for medical education.

Mathematical Concepts Implemented:
---------------------------------
1. **Linear Algebra**:
   - Vector operations for distance calculations
   - Euclidean norm: ||v|| = √(Σ vᵢ²)
   - 2D distance: d = √[(x₂-x₁)² + (y₂-y₁)²]

2. **Color Space Transformations**:
   - RGB to HSV conversion for instrument detection
   - HSV thresholding for color segmentation

3. **Geometry**:
   - Coordinate mapping for hand-to-organ interaction
   - Proximity detection using Euclidean distance

4. **Image Processing**:
   - Holographic scanline effect using matrix operations
   - Canvas composition with alpha blending

Algorithms:
----------
- Color-based instrument detection using HSV thresholding
- Real-time hand tracking with MediaPipe
- Pinch gesture recognition via fingertip distance
- Organ proximity detection with distance thresholds

Author: Medical Simulation Project Team
Style: PEP-8 Compliant
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Tuple, Dict, Optional, Any

class SimpleInstrumentDetector:
    """
    Color-based surgical instrument detection using HSV color space.
    
    Mathematical Basis:
    ------------------
    Uses HSV (Hue, Saturation, Value) color space for robust color detection.
    The HSV transformation separates color information (Hue) from intensity
    (Value), making detection more reliable under varying lighting.
    
    HSV Thresholding Algorithm:
        mask = (H_min ≤ H ≤ H_max) AND (S_min ≤ S ≤ S_max) AND (V_min ≤ V ≤ V_max)
    
    Color Mappings:
        - Red (H: 0-10°, 170-180°) → Scalpel
        - Blue (H: 100-130°) → Forceps
        - Green (H: 40-80°) → Cautery
    
    Attributes:
        current_tool: Currently detected instrument name
    """
    
    def __init__(self) -> None:
        """Initialize the instrument detector."""
        self.current_tool: str = "none"
    
    def detect_by_color(self, frame: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Detect surgical tools using color thresholding in HSV space.
        
        Mathematical Algorithm:
        ----------------------
        1. Convert RGB → HSV using the transformation:
           H = arctan2(√3(G-B), 2R-G-B) / 2π × 360°
           S = (max(R,G,B) - min(R,G,B)) / max(R,G,B)
           V = max(R,G,B)
        
        2. Apply thresholding to create binary masks:
           mask(x,y) = 1 if pixel within HSV range, else 0
        
        3. Count pixels: n = Σ mask(x,y)
        
        4. Classify based on maximum pixel count > threshold (500)
        
        Args:
            frame: Input BGR image as numpy array of shape (H, W, 3)
        
        Returns:
            Tuple of (detected_tool_name, annotated_frame)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        
        # Define HSV color ranges for instrument detection
        # RED marker = Scalpel (Hue wraps around 0°/180°)
        red_lower1 = np.array([0, 100, 100])    # H: 0-10°
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 100, 100])  # H: 170-180°
        red_upper2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) + \
                   cv2.inRange(hsv, red_lower2, red_upper2)
        
        # BLUE marker = Forceps (H: 100-130°)
        blue_lower = np.array([100, 100, 100])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # GREEN marker = Cautery (H: 40-80°)
        green_lower = np.array([40, 100, 100])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Count pixels for each color (sum of binary mask)
        red_pixels = np.sum(red_mask > 0)
        blue_pixels = np.sum(blue_mask > 0)
        green_pixels = np.sum(green_mask > 0)
        
        # Classification threshold: minimum 500 pixels required
        DETECTION_THRESHOLD = 500
        
        detected = "none"
        if red_pixels > DETECTION_THRESHOLD:
            detected = "scalpel"
            cv2.putText(frame, "SCALPEL (Red Marker)", (20, h - 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif blue_pixels > DETECTION_THRESHOLD:
            detected = "forceps"
            cv2.putText(frame, "FORCEPS (Blue Marker)", (20, h - 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        elif green_pixels > DETECTION_THRESHOLD:
            detected = "cautery"
            cv2.putText(frame, "CAUTERY (Green Marker)", (20, h - 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No tool detected - Show colored object", 
                       (20, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (128, 128, 128), 2)
        
        self.current_tool = detected
        return detected, frame


class Simple3DOrganDisplay:
    """
    Simplified 3D organ visualization with proximity detection.
    
    Mathematical Concepts:
    ---------------------
    1. **Coordinate System**:
       - Uses a 2D canvas representing 3D organ positions
       - Origin at top-left, x increases right, y increases down
    
    2. **Euclidean Distance for Proximity**:
       For hand position H = (hx, hy) and organ position O = (ox, oy):
       
           d(H, O) = √[(hx - ox)² + (hy - oy)²]
       
       Using numpy.hypot for numerical stability:
           np.hypot(a, b) = √(a² + b²)
    
    3. **Visual Effects**:
       - Scanline effect: row[i] = row[i] × 0.85 (intensity reduction)
       - Glow effect: Uses additive brightness for highlighted organs
    
    Attributes:
        organs: Dictionary mapping organ names to properties
        highlighted_organ: Currently highlighted organ (if any)
        canvas_width: Width of the display canvas in pixels
        canvas_height: Height of the display canvas in pixels
    """
    
    def __init__(self) -> None:
        """Initialize organ display with predefined organ positions."""
        # Organ properties: position (x, y), color (B, G, R), size, label
        self.organs: Dict[str, Dict[str, Any]] = {
            'heart': {'pos': (400, 250), 'color': (50, 50, 200), 'size': 60, 'label': 'HEART'},
            'liver': {'pos': (550, 300), 'color': (50, 100, 150), 'size': 70, 'label': 'LIVER'},
            'stomach': {'pos': (250, 300), 'color': (100, 120, 180), 'size': 55, 'label': 'STOMACH'},
            'lungs': {'pos': (400, 180), 'color': (150, 100, 100), 'size': 65, 'label': 'LUNGS'},
            'brain': {'pos': (400, 100), 'color': (180, 150, 200), 'size': 50, 'label': 'BRAIN'},
            'kidneys': {'pos': (500, 380), 'color': (80, 70, 120), 'size': 40, 'label': 'KIDNEYS'},
            'appendix': {'pos': (300, 400), 'color': (60, 200, 60), 'size': 35, 'label': 'APPENDIX'},
            'intestines': {'pos': (350, 350), 'color': (120, 140, 160), 'size': 45, 'label': 'INTESTINES'}
        }
        self.highlighted_organ: Optional[str] = None
        self.canvas_width: int = 800
        self.canvas_height: int = 600
    
    def create_canvas(self) -> np.ndarray:
        """
        Create holographic display canvas with all organs rendered.
        
        Mathematical Operations:
        -----------------------
        1. **Matrix Initialization**:
           canvas = zeros((height, width, 3), dtype=uint8)
           Creates H×W×3 matrix of unsigned 8-bit integers
        
        2. **Grid Lines** (Holographic Effect):
           For lines at intervals of 40 pixels:
           canvas[y, :] = [0, 50, 50]  (cyan grid)
        
        3. **Circle Rendering**:
           For each pixel (x, y), if (x-cx)² + (y-cy)² ≤ r²,
           then pixel is inside the circle.
        
        4. **Scanline Effect** (intensity reduction):
           canvas[i:i+2, :] = canvas[i:i+2, :] × 0.85
           This multiplies every even row by 0.85 to create 
           horizontal scanline effect.
        
        Returns:
            numpy array of shape (600, 800, 3) containing the rendered canvas
        """
        # Black background
        canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        
        # Add grid overlay (holographic effect)
        for y in range(0, self.canvas_height, 40):
            cv2.line(canvas, (0, y), (self.canvas_width, y), (0, 50, 50), 1)
        for x in range(0, self.canvas_width, 40):
            cv2.line(canvas, (x, 0), (x, self.canvas_height), (0, 50, 50), 1)
        
        # Draw all organs
        for organ_name, organ_data in self.organs.items():
            pos = organ_data['pos']
            color = organ_data['color']
            size = organ_data['size']
            label = organ_data['label']
            
            # Highlight effect if selected
            if self.highlighted_organ == organ_name:
                # Glow effect
                glow_color = (0, 255, 255)  # Cyan glow
                cv2.circle(canvas, pos, size + 15, glow_color, 3)
                cv2.circle(canvas, pos, size + 10, glow_color, 2)
                
                # Brighter organ
                bright_color = tuple(min(255, c + 100) for c in color)
                cv2.circle(canvas, pos, size, bright_color, -1)
                
                # Pulsing label
                cv2.putText(canvas, f">>> {label} <<<", (pos[0] - 80, pos[1] - size - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                # Normal organ + outline
                cv2.circle(canvas, pos, size, color, -1)
                cv2.circle(canvas, pos, size, (0, 150, 150), 2)
                
                # Label
                cv2.putText(canvas, label, (pos[0] - 40, pos[1] - size - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Add holographic scanlines
        for i in range(0, self.canvas_height, 4):
            canvas[i:i+2, :] = canvas[i:i+2, :] * 0.85
        
        return canvas
    
    def check_proximity(
        self, 
        hand_pos: Optional[Tuple[int, int]]
    ) -> Tuple[Optional[str], float]:
        """
        Check if hand is near any organ using Euclidean distance.
        
        Mathematical Algorithm:
        ----------------------
        For each organ O with position (ox, oy):
            1. Calculate Euclidean distance:
               d = √[(hx - ox)² + (hy - oy)²]
               
            2. Track minimum distance:
               if d < min_distance:
                   min_distance = d
                   closest_organ = O
        
        Proximity threshold: 80 pixels
        
        Time Complexity: O(n) where n = number of organs
        
        Args:
            hand_pos: Hand position as (x, y) tuple, or None if no hand detected
        
        Returns:
            Tuple of (closest_organ_name or None, distance)
        """
        if hand_pos is None:
            self.highlighted_organ = None
            return None, 999
        
        closest_organ: Optional[str] = None
        min_distance: float = 999
        
        hx, hy = hand_pos
        
        # Linear search through all organs for minimum distance
        for organ_name, organ_data in self.organs.items():
            ox, oy = organ_data['pos']
            # Euclidean distance using numpy.hypot for numerical stability
            distance = np.hypot(hx - ox, hy - oy)
            
            if distance < min_distance:
                min_distance = distance
                closest_organ = organ_name
        
        # Proximity threshold for highlighting
        PROXIMITY_THRESHOLD = 80
        if min_distance < PROXIMITY_THRESHOLD:
            self.highlighted_organ = closest_organ
        else:
            self.highlighted_organ = None
        
        return closest_organ, min_distance


class SimplifiedHologramTrainer:
    """Simplified hologram surgical trainer - guaranteed to work"""
    
    def __init__(self):
        # Hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Instrument detection
        self.instrument_detector = SimpleInstrumentDetector()
        
        # 3D Organ display
        self.organ_display = Simple3DOrganDisplay()
        
        # Training state
        self.current_procedure = None
        self.current_step = 0
        self.is_pinching = False
        self.last_action_time = 0
        self.score = 0
        self.mistakes = 0
        self.start_time = time.time()
        
        # Procedures
        self.procedures = {
            'appendectomy': {
                'name': 'Appendectomy',
                'steps': [
                    {'description': 'Make incision', 'tool': 'scalpel', 'organ': 'appendix'},
                    {'description': 'Identify appendix', 'tool': 'forceps', 'organ': 'appendix'},
                    {'description': 'Ligate blood supply', 'tool': 'cautery', 'organ': 'appendix'},
                    {'description': 'Remove appendix', 'tool': 'forceps', 'organ': 'appendix'},
                    {'description': 'Close incision', 'tool': 'scalpel', 'organ': 'stomach'}
                ]
            },
            'cholecystectomy': {
                'name': 'Gallbladder Removal',
                'steps': [
                    {'description': 'Expose liver', 'tool': 'forceps', 'organ': 'liver'},
                    {'description': 'Identify cystic duct', 'tool': 'forceps', 'organ': 'liver'},
                    {'description': 'Clip cystic duct', 'tool': 'cautery', 'organ': 'liver'},
                    {'description': 'Remove gallbladder', 'tool': 'forceps', 'organ': 'liver'}
                ]
            }
        }
        
        self.hand_position = None
        self.feedback = "Welcome! Select procedure by pressing key (1 or 2)"
    
    def process_hand_tracking(self, frame):
        """Process hand tracking and return hand position"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        hand_pos_2d = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand skeleton
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2),
                    self.mp_draw.DrawingSpec(color=(0, 200, 200), thickness=2)
                )
                
                # Get finger positions
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                
                # Map to organ canvas coordinates
                hand_pos_2d = (ix, int(iy * 0.8))  # Adjust mapping
                
                # Pinch detection
                pinch_dist = np.hypot(ix - tx, iy - ty)
                self.is_pinching = pinch_dist < 40
                
                # Visual feedback
                color = (0, 255, 0) if self.is_pinching else (255, 255, 0)
                cv2.circle(frame, (ix, iy), 20, color, 3)
                
                if self.is_pinching:
                    cv2.putText(frame, "PINCH!", (ix + 30, iy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        self.hand_position = hand_pos_2d
        return frame
    
    def handle_surgical_action(self, organ_name, distance, tool):
        """Handle surgical action on organ"""
        if self.current_procedure is None:
            return
        
        if self.current_step >= len(self.current_procedure['steps']):
            self.feedback = "✅ PROCEDURE COMPLETE!"
            return
        
        current_step_data = self.current_procedure['steps'][self.current_step]
        required_tool = current_step_data['tool']
        required_organ = current_step_data['organ']
        
        # Check if correct
        if tool == required_tool and organ_name == required_organ and distance < 80:
            self.feedback = f"✅ CORRECT! Step {self.current_step + 1} complete"
            self.score += 20
            self.current_step += 1
        else:
            self.feedback = f"❌ WRONG! Need {required_tool} on {required_organ}"
            self.mistakes += 1
    
    def draw_ui(self, camera_frame, organ_canvas):
        """Combine everything into final display"""
        h, w = camera_frame.shape[:2]
        
        # Create large display (1400x700)
        display = np.zeros((700, 1400, 3), dtype=np.uint8)
        
        # Left: Camera feed (resized)
        camera_resized = cv2.resize(camera_frame, (600, 450))
        display[50:500, 50:650] = camera_resized
        
        # Right: Organ Display (resize to fit)
        organ_resized = cv2.resize(organ_canvas, (600, 600))
        display[50:650, 750:1350] = organ_resized
        
        # Top bar
        cv2.rectangle(display, (0, 0), (1400, 50), (20, 20, 40), -1)
        cv2.putText(display, "🔮 HOLOGRAM SURGICAL TRAINING CENTER", (50, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Bottom instructions
        cv2.rectangle(display, (0, 650), (1400, 700), (20, 20, 40), -1)
        
        if self.current_procedure is None:
            cv2.putText(display, "Press '1' for Appendectomy  |  Press '2' for Cholecystectomy  |  'Q' to Quit",
                       (50, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Show current step
            if self.current_step < len(self.current_procedure['steps']):
                step_info = self.current_procedure['steps'][self.current_step]
                text = f"Step {self.current_step + 1}/{len(self.current_procedure['steps'])}: {step_info['description']} | Tool: {step_info['tool']} | Target: {step_info['organ']}"
                cv2.putText(display, text, (50, 680),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(display, f"✅ COMPLETE! Score: {self.score} | Mistakes: {self.mistakes} | Press 'R' to restart",
                           (50, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Feedback box
        cv2.rectangle(display, (50, 510), (650, 560), (30, 30, 50), -1)
        cv2.rectangle(display, (50, 510), (650, 560), (0, 200, 200), 2)
        cv2.putText(display, self.feedback[:60], (60, 540),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Stats panel
        elapsed = int(time.time() - self.start_time)
        cv2.rectangle(display, (50, 570), (650, 640), (30, 30, 50), -1)
        cv2.putText(display, f"Time: {elapsed}s  |  Score: {self.score}  |  Mistakes: {self.mistakes}",
                   (60, 605), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display
    
    def run(self):
        """Main training loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Camera tidak buka!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("🔮 Hologram Surgical Training System")
        print("=" * 50)
        print("📌 Show RED object = Scalpel")
        print("📌 Show BLUE object = Forceps")  
        print("📌 Show GREEN object = Cautery")
        print("📌 Make PINCH gesture (thumb+index) on organ")
        print("📌 Press 1 = Appendectomy, 2 = Cholecystectomy")
        print("📌 Press Q = Quit, R = Restart")
        print("=" * 50)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect instrument
            detected_tool, frame = self.instrument_detector.detect_by_color(frame)
            
            # Track hand
            frame = self.process_hand_tracking(frame)
            
            # Check organ proximity
            closest_organ, distance = self.organ_display.check_proximity(self.hand_position)
            
            # Handle pinch action
            if self.is_pinching and closest_organ:
                current_time = time.time()
                if current_time - self.last_action_time > 1.0:
                    self.handle_surgical_action(closest_organ, distance, detected_tool)
                    self.last_action_time = current_time
            
            # Create organ canvas
            organ_canvas = self.organ_display.create_canvas()
            
            # Draw hand cursor on organ canvas
            if self.hand_position:
                hx, hy = self.hand_position
                if 0 <= hx < 800 and 0 <= hy < 600:
                    cv2.circle(organ_canvas, (hx, hy), 15, (0, 255, 255), 2)
                    cv2.circle(organ_canvas, (hx, hy), 5, (0, 255, 255), -1)
            
            # Create final display
            display = self.draw_ui(frame, organ_canvas)
            
            cv2.imshow('Hologram Surgical Training', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.current_procedure = self.procedures['appendectomy']
                self.current_step = 0
                self.score = 0
                self.mistakes = 0
                self.start_time = time.time()
                self.feedback = f"Starting {self.current_procedure['name']}..."
            elif key == ord('2'):
                self.current_procedure = self.procedures['cholecystectomy']
                self.current_step = 0
                self.score = 0
                self.mistakes = 0
                self.start_time = time.time()
                self.feedback = f"Starting {self.current_procedure['name']}..."
            elif key == ord('r'):
                self.current_procedure = None
                self.current_step = 0
                self.score = 0
                self.mistakes = 0
                self.feedback = "Select procedure (1 or 2)"
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    trainer = SimplifiedHologramTrainer()
    trainer.run()
