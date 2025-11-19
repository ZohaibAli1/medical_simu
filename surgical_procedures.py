import numpy as np
import cv2
import math


class SurgicalProcedures:
    def __init__(self):
        self.procedures = {
            'incision': self.make_incision,
            'suture': self.apply_suture,
            'cauterize': self.cauterize,
            'extract': self.extract_organ
        }

    def render_tool(self, frame, tool_name, tip_position):
        """Renders surgical tool overlaid on hand"""
        x, y = tip_position
        color = (200, 200, 200)

        if tool_name == 'scalpel':
            cv2.line(frame, (x, y), (x + 40, y + 40), (100, 100, 100), 4)
            cv2.line(frame, (x, y), (x - 20, y + 20), (255, 255, 255), 2)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        elif tool_name == 'forceps' or tool_name == 'extract':
            cv2.line(frame, (x, y), (x + 30, y + 50), color, 3)
            cv2.line(frame, (x + 10, y), (x + 40, y + 50), color, 3)
        elif tool_name == 'cauterizer':
            cv2.line(frame, (x, y), (x + 50, y + 50), (50, 50, 50), 5)
            cv2.circle(frame, (x, y), 5, (0, 165, 255), -1)
        elif tool_name == 'needle_driver' or tool_name == 'suture':
            cv2.line(frame, (x, y), (x + 40, y + 40), color, 4)
            cv2.ellipse(frame, (x - 10, y), (15, 15), 0, 180, 360, (200, 200, 200), 2)
        else:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), 2)
        return frame

    def render_guidance(self, frame, step_type, current_hand_pos, target_area=None):
        """
        TEACHING MODE: Draws guides on where to move the hand.
        """
        h, w = frame.shape[:2]

        # Define a fixed target area for demo purposes
        start_pt = (w // 2 - 50, h // 2 + 50)
        end_pt = (w // 2 + 50, h // 2 + 80)

        instruction_text = ""

        if step_type == 'incision':
            # 1. Draw the Cut Line (Dotted Line)
            self.draw_dotted_line(frame, start_pt, end_pt, (255, 255, 255))

            # 2. Draw Start and End Markers
            cv2.circle(frame, start_pt, 8, (0, 255, 0), 2)  # Green Start
            cv2.putText(frame, "START", (start_pt[0] - 20, start_pt[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0), 1)
            cv2.circle(frame, end_pt, 8, (0, 0, 255), 2)  # Red End

            # 3. Calculate distance
            dist_to_start = math.hypot(current_hand_pos[0] - start_pt[0], current_hand_pos[1] - start_pt[1])
            dist_to_end = math.hypot(current_hand_pos[0] - end_pt[0], current_hand_pos[1] - end_pt[1])

            # 4. Logic for Guidance
            if dist_to_start < 30:
                instruction_text = "PRESS & DRAG TO END POINT"
                cv2.arrowedLine(frame, current_hand_pos, end_pt, (0, 255, 0), 3)
            elif dist_to_end < 30:
                instruction_text = "INCISION COMPLETE - RELEASE"
            else:
                instruction_text = "MOVE TO GREEN START POINT"
                cv2.arrowedLine(frame, current_hand_pos, start_pt, (0, 255, 255), 2)

        elif step_type == 'suture':
            points = [(w // 2, h // 2), (w // 2 + 20, h // 2 + 10), (w // 2 + 40, h // 2)]
            for pt in points:
                cv2.circle(frame, pt, 3, (255, 0, 0), -1)
            instruction_text = "STITCH THROUGH BLUE POINTS"

        elif step_type == 'identify_organ':
            cv2.rectangle(frame, (w // 2 - 60, h // 2 + 20), (w // 2 + 60, h // 2 + 120), (0, 255, 255), 2)
            instruction_text = "LOCATE AND POINT AT APPENDIX"

        # Draw instructions near hand
        if instruction_text:
            cv2.putText(frame, instruction_text, (current_hand_pos[0] + 20, current_hand_pos[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def draw_dotted_line(self, frame, p1, p2, color, thickness=2, gap=10):
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        pts = int(dist / gap)
        for i in range(pts):
            if i % 2 == 0:
                t1 = i / pts
                t2 = (i + 1) / pts
                x1 = int(p1[0] + (p2[0] - p1[0]) * t1)
                y1 = int(p1[1] + (p2[1] - p1[1]) * t1)
                x2 = int(p1[0] + (p2[0] - p1[0]) * t2)
                y2 = int(p1[1] + (p2[1] - p1[1]) * t2)
                cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

    def make_incision(self, frame, start_pos, end_pos, depth=1):
        color = (200, 0, 0)
        thickness = max(1, 3 + depth)
        cv2.line(frame, start_pos, end_pos, color, thickness)
        return frame

    def apply_suture(self, frame, points):
        return frame

    def cauterize(self, frame, pos):
        cv2.circle(frame, pos, 15, (50, 50, 50), -1)
        return frame

    def extract_organ(self, frame):
        return frame