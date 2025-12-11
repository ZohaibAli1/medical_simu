"""
Pyramid Hologram Output Generator
=================================

Creates a 4-view display for DIY smartphone pyramid holograms.

Mathematical Concepts Implemented:
---------------------------------
1. **Rotation Matrices (Linear Algebra)**:
   The 90° and 180° rotations are implemented via matrix operations:
   
   90° Clockwise:     R = |  0  1 |    (x', y') = (y, -x)
                          | -1  0 |
   
   180° Rotation:      R = | -1  0 |    (x', y') = (-x, -y)
                          |  0 -1 |
   
   90° Counter-CW:    R = |  0 -1 |    (x', y') = (-y, x)
                          |  1  0 |

2. **Image Resizing (Interpolation)**:
   cv2.resize uses bilinear interpolation by default:
   f(x, y) = (1-t)(1-u)f₀₀ + t(1-u)f₁₀ + (1-t)uf₀₁ + tuf₁₁
   Where t, u are fractional pixel positions.

3. **Canvas Geometry**:
   The 4-view layout forms a cross pattern for pyramid projection:
   
       [  TOP  ]
   [LEFT][    ][RIGHT]
       [BOTTOM]

4. **Matrix Operations**:
   Output canvas: zeros((H×2, W×2, 3), dtype=uint8)
   Creates a 2D array of shape (800, 800, 3) for RGBA display.

Author: Medical Simulation Project Team
Style: PEP-8 Compliant
"""

import cv2
import numpy as np
from typing import Tuple


def create_pyramid_hologram_output(organ_canvas: np.ndarray) -> np.ndarray:
    """
    Create 4-sided pyramid hologram display.
    
    Mathematical Algorithm:
    ----------------------
    1. Create 4 rotated views using rotation matrices:
       - Top view:    R(0°)   - Original
       - Right view:  R(90°)  - Clockwise rotation
       - Bottom view: R(180°) - Half rotation
       - Left view:   R(270°) - Counter-clockwise
    
    2. Resize each view to pyramid_size:
       Using bilinear interpolation: (H, W) → (400, 400)
    
    3. Arrange in cross pattern on output canvas:
       Output size: (800, 800, 3)
       
       Layout coordinates:
       - Top:    [0:400, 200:600]
       - Left:   [200:600, 0:400]
       - Right:  [200:600, 400:800]
       - Bottom: [400:800, 200:600]
    
    Args:
        organ_canvas: Input image as numpy array (H, W, 3)
    
    Returns:
        4-view pyramid display as numpy array (800, 800, 3)
    
    Note:
        For use with DIY smartphone pyramid hologram projector.
    """
    h, w = organ_canvas.shape[:2]
    
    # Create 4 rotated views using rotation matrix transformations
    # R(0°) - Identity (no rotation)
    view_top = organ_canvas.copy()
    
    # R(90°) clockwise: [x, y] → [y, W-1-x]
    view_right = cv2.rotate(organ_canvas, cv2.ROTATE_90_CLOCKWISE)
    
    # R(180°): [x, y] → [W-1-x, H-1-y]
    view_bottom = cv2.rotate(organ_canvas, cv2.ROTATE_180)
    
    # R(270°) = R(-90°): [x, y] → [H-1-y, x]
    view_left = cv2.rotate(organ_canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Resize to fit pyramid (bilinear interpolation)
    pyramid_size: Tuple[int, int] = (400, 400)
    view_top = cv2.resize(view_top, pyramid_size)
    view_right = cv2.resize(view_right, pyramid_size)
    view_bottom = cv2.resize(view_bottom, pyramid_size)
    view_left = cv2.resize(view_left, pyramid_size)
    
    # Create output canvas: 2x pyramid size
    # Matrix shape: (H×2, W×2, channels) = (800, 800, 3)
    output = np.zeros(
        (pyramid_size[1] * 2, pyramid_size[0] * 2, 3), 
        dtype=np.uint8
    )
    
    # Calculate center offsets for cross pattern layout
    center_y = pyramid_size[1] // 2  # 200
    center_x = pyramid_size[0] // 2  # 200
    
    # Place views in cross pattern using array slicing
    # Top: rows [0:400], cols [200:600]
    output[0:pyramid_size[1], center_x:center_x+pyramid_size[0]] = view_top
    
    # Left: rows [200:600], cols [0:400]
    output[center_y:center_y+pyramid_size[1], 0:pyramid_size[0]] = view_left
    
    # Right: rows [200:600], cols [400:800]
    output[center_y:center_y+pyramid_size[1], 
           pyramid_size[0]*2-pyramid_size[0]:pyramid_size[0]*2] = view_right
    
    # Bottom: rows [400:800], cols [200:600]
    output[pyramid_size[1]:pyramid_size[1]*2, 
           center_x:center_x+pyramid_size[0]] = view_bottom
    
    # Add text overlays for user guidance
    cv2.putText(
        output, "PYRAMID HOLOGRAM OUTPUT", (250, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
    )
    cv2.putText(
        output, "Place pyramid on center of screen", (200, 780),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2
    )
    
    return output


# Add to simplified_hologram_trainer.py in the run() method:
# 
# Inside the main loop, after creating organ_canvas:
#
#    # Create pyramid hologram output
#    pyramid_output = create_pyramid_hologram_output(organ_canvas)
#    cv2.imshow('Pyramid Hologram (For Smartphone)', pyramid_output)
#    cv2.resizeWindow('Pyramid Hologram (For Smartphone)', 800, 800)


if __name__ == "__main__":
    # Demo
    # Create sample organ canvas
    sample = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Draw sample organ (heart)
    cv2.circle(sample, (400, 300), 80, (50, 50, 200), -1)
    cv2.circle(sample, (400, 300), 80, (0, 200, 200), 3)
    cv2.putText(sample, "HEART", (350, 310),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Add grid
    for y in range(0, 600, 40):
        cv2.line(sample, (0, y), (800, y), (0, 50, 50), 1)
    for x in range(0, 800, 40):
        cv2.line(sample, (x, 0), (x, 600), (0, 50, 50), 1)
    
    # Create pyramid output
    pyramid = create_pyramid_hologram_output(sample)
    
    cv2.imshow('Pyramid Hologram Output', pyramid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
