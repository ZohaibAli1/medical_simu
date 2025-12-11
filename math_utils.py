"""
Mathematical Utilities Module for Medical Simulation Project
=============================================================

This module provides a comprehensive collection of mathematical functions
and algorithms used throughout the holographic surgical training system.

Mathematical Concepts Implemented:
- Linear Algebra: Vectors, Matrices, Norms, Transformations
- Probability: Gaussian Distribution
- Statistics: Mean, Variance, Weighted Scoring
- Calculus: Derivatives (implicit in motion calculations)
- Geometry: 2D/3D Transformations, Rotations

Author: Medical Simulation Project Team
"""

import numpy as np
from typing import Tuple, List, Optional, Union
import math


# =============================================================================
# LINEAR ALGEBRA OPERATIONS
# =============================================================================

def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two points.
    
    Mathematical Definition:
    -----------------------
    For two points P₁ = (x₁, y₁, z₁) and P₂ = (x₂, y₂, z₂):
    
        d(P₁, P₂) = √[(x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²]
    
    This is equivalent to the L2 norm of the difference vector:
        d = ||P₂ - P₁||₂
    
    Args:
        point1: First point as numpy array [x, y, z] or [x, y]
        point2: Second point as numpy array [x, y, z] or [x, y]
    
    Returns:
        Euclidean distance as a float
    
    Example:
        >>> p1 = np.array([0, 0, 0])
        >>> p2 = np.array([3, 4, 0])
        >>> euclidean_distance(p1, p2)
        5.0
    """
    return np.linalg.norm(point2 - point1)


def euclidean_distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate the Euclidean distance in 2D using the Pythagorean theorem.
    
    Mathematical Definition:
    -----------------------
    d = √[(x₂-x₁)² + (y₂-y₁)²]
    
    This is the hypotenuse of a right triangle formed by the horizontal
    and vertical distances between the two points.
    
    Args:
        x1, y1: Coordinates of first point
        x2, y2: Coordinates of second point
    
    Returns:
        Distance as a float
    """
    return np.hypot(x2 - x1, y2 - y1)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Mathematical Definition:
    -----------------------
    For a vector v = (v₁, v₂, ..., vₙ):
    
        v̂ = v / ||v||
    
    Where ||v|| is the L2 norm (Euclidean length):
        ||v|| = √(v₁² + v₂² + ... + vₙ²)
    
    The resulting unit vector has magnitude 1 and preserves direction.
    
    Args:
        vector: Input vector as numpy array
    
    Returns:
        Normalized unit vector
    
    Raises:
        ValueError: If vector has zero magnitude
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize zero vector")
    return vector / norm


def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the dot product (inner product) of two vectors.
    
    Mathematical Definition:
    -----------------------
    For vectors a = (a₁, a₂, ..., aₙ) and b = (b₁, b₂, ..., bₙ):
    
        a · b = Σᵢ aᵢbᵢ = a₁b₁ + a₂b₂ + ... + aₙbₙ
    
    Geometric interpretation:
        a · b = ||a|| ||b|| cos(θ)
    
    Where θ is the angle between the vectors.
    
    Properties:
    - If a · b = 0, vectors are perpendicular (orthogonal)
    - If a · b > 0, angle is acute
    - If a · b < 0, angle is obtuse
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        Scalar dot product value
    """
    return np.dot(v1, v2)


def vector_magnitude(vector: np.ndarray) -> float:
    """
    Calculate the magnitude (L2 norm) of a vector.
    
    Mathematical Definition:
    -----------------------
    For vector v = (v₁, v₂, ..., vₙ):
    
        ||v||₂ = √(v₁² + v₂² + ... + vₙ²)
    
    This represents the "length" of the vector in Euclidean space.
    
    Args:
        vector: Input vector as numpy array
    
    Returns:
        Magnitude as a float
    """
    return np.linalg.norm(vector)


# =============================================================================
# MATRIX OPERATIONS
# =============================================================================

def rotation_matrix_2d(angle_degrees: float) -> np.ndarray:
    """
    Create a 2D rotation matrix for the given angle.
    
    Mathematical Definition:
    -----------------------
    For rotation by angle θ (counterclockwise):
    
        R(θ) = | cos(θ)  -sin(θ) |
               | sin(θ)   cos(θ) |
    
    To rotate a point P = (x, y):
        P' = R(θ) × P
    
    Properties:
    - det(R) = 1 (rotation preserves area)
    - R⁻¹ = Rᵀ (orthogonal matrix)
    - R(-θ) = R(θ)⁻¹ (inverse rotation)
    
    Args:
        angle_degrees: Rotation angle in degrees
    
    Returns:
        2x2 rotation matrix as numpy array
    """
    theta = np.radians(angle_degrees)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    return np.array([
        [cos_t, -sin_t],
        [sin_t, cos_t]
    ])


def rotation_matrix_3d(axis: str, angle_degrees: float) -> np.ndarray:
    """
    Create a 3D rotation matrix around a specified axis.
    
    Mathematical Definition:
    -----------------------
    Rotation around X-axis by angle θ:
        Rx(θ) = | 1    0       0    |
                | 0  cos(θ) -sin(θ) |
                | 0  sin(θ)  cos(θ) |
    
    Rotation around Y-axis by angle θ:
        Ry(θ) = |  cos(θ)  0  sin(θ) |
                |    0     1    0    |
                | -sin(θ)  0  cos(θ) |
    
    Rotation around Z-axis by angle θ:
        Rz(θ) = | cos(θ) -sin(θ)  0 |
                | sin(θ)  cos(θ)  0 |
                |   0       0     1 |
    
    Args:
        axis: 'x', 'y', or 'z'
        angle_degrees: Rotation angle in degrees
    
    Returns:
        3x3 rotation matrix as numpy array
    """
    theta = np.radians(angle_degrees)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    if axis.lower() == 'x':
        return np.array([
            [1, 0, 0],
            [0, cos_t, -sin_t],
            [0, sin_t, cos_t]
        ])
    elif axis.lower() == 'y':
        return np.array([
            [cos_t, 0, sin_t],
            [0, 1, 0],
            [-sin_t, 0, cos_t]
        ])
    elif axis.lower() == 'z':
        return np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")


def matrix_determinant(matrix: np.ndarray) -> float:
    """
    Calculate the determinant of a square matrix.
    
    Mathematical Definition:
    -----------------------
    For a 2x2 matrix:
        A = | a  b |
            | c  d |
        det(A) = ad - bc
    
    For a 3x3 matrix, use Sarrus' rule or cofactor expansion.
    
    Properties:
    - det(AB) = det(A) × det(B)
    - det(Aᵀ) = det(A)
    - det(A⁻¹) = 1/det(A)
    - If det(A) = 0, matrix is singular (non-invertible)
    
    Args:
        matrix: Square matrix as numpy array
    
    Returns:
        Determinant as a float
    """
    return np.linalg.det(matrix)


def matrix_trace(matrix: np.ndarray) -> float:
    """
    Calculate the trace of a square matrix.
    
    Mathematical Definition:
    -----------------------
    The trace is the sum of diagonal elements:
    
        tr(A) = Σᵢ aᵢᵢ = a₁₁ + a₂₂ + ... + aₙₙ
    
    Properties:
    - tr(A + B) = tr(A) + tr(B)  (linearity)
    - tr(AB) = tr(BA)  (cyclic property)
    - tr(Aᵀ) = tr(A)
    
    Args:
        matrix: Square matrix as numpy array
    
    Returns:
        Trace as a float
    """
    return np.trace(matrix)


# =============================================================================
# COORDINATE TRANSFORMATIONS
# =============================================================================

def map_2d_to_3d(
    hand_x: float, 
    hand_y: float, 
    frame_width: int, 
    frame_height: int,
    hand_z: float = 0,
    anatomy_bounds: float = 200
) -> np.ndarray:
    """
    Map 2D screen coordinates to 3D anatomical space.
    
    Mathematical Definition:
    -----------------------
    Given screen coordinates (x, y) and image dimensions (W, H):
    
    1. Normalize to [0, 1]:
        x_norm = x / W
        y_norm = y / H
    
    2. Transform to centered range [-1, 1]:
        x_centered = 2 × x_norm - 1
        y_centered = 2 × y_norm - 1
    
    3. Scale to anatomy bounds:
        X_3d = x_centered × (bounds / 2)
        Y_3d = y_centered × (bounds / 2)
        Z_3d = base_z + hand_z × scale
    
    This is a simplified perspective projection inverse.
    
    Args:
        hand_x: X coordinate on screen
        hand_y: Y coordinate on screen
        frame_width: Width of the video frame
        frame_height: Height of the video frame
        hand_z: Estimated depth from hand tracking
        anatomy_bounds: Size of the 3D anatomy space
    
    Returns:
        3D position as numpy array [x, y, z]
    """
    # Normalize to [0, 1]
    norm_x = hand_x / frame_width
    norm_y = hand_y / frame_height
    
    # Transform to 3D coordinates centered at origin
    x_3d = (norm_x - 0.5) * anatomy_bounds
    y_3d = (norm_y - 0.5) * anatomy_bounds
    z_3d = 100 + hand_z * 100
    
    return np.array([x_3d, y_3d, z_3d])


def project_3d_to_2d(
    pos_3d: np.ndarray, 
    frame_width: int, 
    frame_height: int
) -> Tuple[int, int]:
    """
    Project 3D coordinates back to 2D screen space.
    
    Mathematical Definition:
    -----------------------
    Simplified orthographic projection (ignores perspective):
    
    1. Normalize 3D position to [-1, 1]:
        x_norm = X_3d / 100  (assuming bounds of [-100, 100])
        y_norm = Y_3d / 100
    
    2. Transform to screen coordinates:
        x_screen = (x_norm + 1) / 2 × W
        y_screen = (y_norm + 1) / 2 × H
    
    Note: Full perspective projection would use:
        x_screen = (f × X) / Z + cx
        y_screen = (f × Y) / Z + cy
    Where f is focal length and (cx, cy) is principal point.
    
    Args:
        pos_3d: 3D position as numpy array [x, y, z]
        frame_width: Width of the output frame
        frame_height: Height of the output frame
    
    Returns:
        Tuple of (x, y) screen coordinates
    """
    # Normalize 3D position to screen space
    x_norm = (pos_3d[0] + 100) / 200
    y_norm = (pos_3d[1] + 100) / 200
    
    # Convert to pixel coordinates
    x_2d = int(x_norm * frame_width)
    y_2d = int(y_norm * frame_height)
    
    # Clamp to valid range
    x_2d = max(0, min(frame_width - 1, x_2d))
    y_2d = max(0, min(frame_height - 1, y_2d))
    
    return (x_2d, y_2d)


# =============================================================================
# PROBABILITY & GAUSSIAN DISTRIBUTION
# =============================================================================

def gaussian_1d(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Calculate the 1D Gaussian (Normal) probability density.
    
    Mathematical Definition:
    -----------------------
    The Gaussian/Normal distribution PDF:
    
        f(x) = (1 / (σ√(2π))) × exp(-(x-μ)² / (2σ²))
    
    Where:
    - μ (mu) is the mean (center of the distribution)
    - σ (sigma) is the standard deviation (spread)
    - σ² is the variance
    
    Properties:
    - Symmetric about the mean
    - 68.27% of values within ±1σ
    - 95.45% of values within ±2σ
    - 99.73% of values within ±3σ
    
    Args:
        x: Input value
        mu: Mean of the distribution
        sigma: Standard deviation
    
    Returns:
        Probability density at x
    """
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coefficient * np.exp(exponent)


def gaussian_kernel_2d(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel for image convolution.
    
    Mathematical Definition:
    -----------------------
    The 2D Gaussian function:
    
        G(x, y) = (1 / (2πσ²)) × exp(-(x² + y²) / (2σ²))
    
    This creates a "bell-shaped" surface used for:
    - Image blurring/smoothing
    - Noise reduction
    - Edge detection (Difference of Gaussians)
    
    The kernel is normalized so all values sum to 1,
    preserving image brightness.
    
    Args:
        size: Kernel size (should be odd number)
        sigma: Standard deviation
    
    Returns:
        2D Gaussian kernel as numpy array
    
    Note:
        OpenCV's cv2.GaussianBlur() uses this kernel internally.
    """
    x = np.arange(size) - (size - 1) / 2
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.sum()


# =============================================================================
# STATISTICS
# =============================================================================

def calculate_mean(values: List[float]) -> float:
    """
    Calculate the arithmetic mean (average).
    
    Mathematical Definition:
    -----------------------
    For a set of n values {x₁, x₂, ..., xₙ}:
    
        μ = (1/n) × Σᵢ xᵢ = (x₁ + x₂ + ... + xₙ) / n
    
    Properties:
    - Sensitive to outliers
    - Minimizes sum of squared deviations
    
    Args:
        values: List of numerical values
    
    Returns:
        Mean value
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_variance(values: List[float], sample: bool = True) -> float:
    """
    Calculate variance of a dataset.
    
    Mathematical Definition:
    -----------------------
    Population variance:
        σ² = (1/N) × Σᵢ (xᵢ - μ)²
    
    Sample variance (unbiased):
        s² = (1/(n-1)) × Σᵢ (xᵢ - x̄)²
    
    The sample variance uses (n-1) for Bessel's correction,
    providing an unbiased estimate of population variance.
    
    Args:
        values: List of numerical values
        sample: If True, calculate sample variance (n-1 denominator)
    
    Returns:
        Variance value
    """
    if len(values) < 2:
        return 0.0
    
    mean = calculate_mean(values)
    squared_diffs = [(x - mean) ** 2 for x in values]
    
    if sample:
        return sum(squared_diffs) / (len(values) - 1)
    else:
        return sum(squared_diffs) / len(values)


def calculate_standard_deviation(values: List[float], sample: bool = True) -> float:
    """
    Calculate standard deviation of a dataset.
    
    Mathematical Definition:
    -----------------------
    Standard deviation is the square root of variance:
    
        σ = √(variance)
    
    It measures the average distance of data points from the mean,
    in the same units as the original data.
    
    Args:
        values: List of numerical values
        sample: If True, calculate sample standard deviation
    
    Returns:
        Standard deviation value
    """
    return np.sqrt(calculate_variance(values, sample))


def calculate_accuracy(correct: int, total: int) -> float:
    """
    Calculate accuracy percentage.
    
    Mathematical Definition:
    -----------------------
    Accuracy = (Number of Correct Actions / Total Actions) × 100
    
        Accuracy% = (correct / total) × 100
    
    This is used in the assessment module to evaluate
    surgical procedure performance.
    
    Args:
        correct: Number of correct actions
        total: Total number of actions
    
    Returns:
        Accuracy as a percentage (0-100)
    """
    if total == 0:
        return 0.0
    return (correct / total) * 100


def weighted_score(
    base_score: float,
    accuracy_weight: float,
    time_penalty: float,
    mistake_penalty: float,
    mistakes: int,
    time_seconds: float
) -> float:
    """
    Calculate weighted performance score.
    
    Mathematical Definition:
    -----------------------
    The scoring algorithm used in the assessment module:
    
        Score = base × accuracy_weight 
                + max(0, 100 - mistakes × mistake_penalty) × (1 - accuracy_weight)
                - (time / 600) × time_penalty
    
    Components:
    - Base performance from accuracy
    - Penalty for each mistake
    - Time-based penalty (10 points per 10 minutes)
    
    The score is clamped to [0, 100].
    
    Args:
        base_score: Initial score (typically 100)
        accuracy_weight: Weight for accuracy component (0-1)
        time_penalty: Points deducted per time unit
        mistake_penalty: Points deducted per mistake
        mistakes: Number of mistakes made
        time_seconds: Total time taken in seconds
    
    Returns:
        Final weighted score (0-100)
    """
    accuracy_component = base_score * accuracy_weight
    mistake_component = max(0, 100 - mistakes * mistake_penalty) * (1 - accuracy_weight)
    time_component = (time_seconds / 600) * time_penalty
    
    final_score = accuracy_component + mistake_component - time_component
    return max(0, min(100, final_score))


# =============================================================================
# GEOMETRIC CALCULATIONS
# =============================================================================

def linear_interpolation(
    t: float, 
    p1: Tuple[float, float], 
    p2: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Perform linear interpolation between two points.
    
    Mathematical Definition:
    -----------------------
    For parameter t ∈ [0, 1] and points P₁, P₂:
    
        P(t) = (1 - t) × P₁ + t × P₂
    
    Or equivalently:
        P(t) = P₁ + t × (P₂ - P₁)
    
    This is used for:
    - Drawing dotted lines
    - Smooth transitions
    - Path interpolation
    
    Args:
        t: Interpolation parameter (0 = at p1, 1 = at p2)
        p1: Start point as (x, y) tuple
        p2: End point as (x, y) tuple
    
    Returns:
        Interpolated point as (x, y) tuple
    """
    x = p1[0] + t * (p2[0] - p1[0])
    y = p1[1] + t * (p2[1] - p1[1])
    return (x, y)


def point_in_rectangle(
    point: Tuple[float, float],
    rect_min: Tuple[float, float],
    rect_max: Tuple[float, float]
) -> bool:
    """
    Check if a point lies within a rectangle.
    
    Mathematical Definition:
    -----------------------
    Point P = (x, y) is inside rectangle if:
    
        x_min ≤ x ≤ x_max  AND  y_min ≤ y ≤ y_max
    
    This is used for:
    - Menu item hover detection
    - Tool selection
    - Boundary checking
    
    Args:
        point: Point coordinates (x, y)
        rect_min: Bottom-left corner (x_min, y_min)
        rect_max: Top-right corner (x_max, y_max)
    
    Returns:
        True if point is inside rectangle
    """
    return (rect_min[0] <= point[0] <= rect_max[0] and 
            rect_min[1] <= point[1] <= rect_max[1])


def distance_point_to_line(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float]
) -> float:
    """
    Calculate shortest distance from a point to a line segment.
    
    Mathematical Definition:
    -----------------------
    Using the vector projection method:
    
    1. Line vector: L = line_end - line_start
    2. Point vector: P = point - line_start
    3. Projection parameter: t = (P · L) / (L · L)
    4. Clamp t to [0, 1] for segment
    5. Closest point: C = line_start + t × L
    6. Distance: d = ||point - C||
    
    Args:
        point: The point (x, y)
        line_start: Start of line segment (x1, y1)
        line_end: End of line segment (x2, y2)
    
    Returns:
        Perpendicular distance to line segment
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Line vector
    dx = x2 - x1
    dy = y2 - y1
    
    # Handle degenerate case (line is a point)
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    
    # Calculate projection parameter
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    
    # Find closest point on line
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Return distance to closest point
    return math.hypot(px - closest_x, py - closest_y)


# =============================================================================
# HSV COLOR SPACE (Linear Algebra Application)
# =============================================================================

def rgb_to_hsv_explanation() -> str:
    """
    Returns explanation of RGB to HSV conversion.
    
    This is used in the surgical instrument detection
    for color-based recognition of tools.
    
    Mathematical Definition:
    -----------------------
    Given RGB values normalized to [0, 1]:
    
    Let: M = max(R, G, B)
         m = min(R, G, B)
         C = M - m  (Chroma)
    
    Value (V):
        V = M
    
    Saturation (S):
        S = C / V  if V ≠ 0, else S = 0
    
    Hue (H):
        H = 0° + 60° × (G-B)/C   if M = R
        H = 120° + 60° × (B-R)/C if M = G
        H = 240° + 60° × (R-G)/C if M = B
    
    HSV is preferred for color detection because:
    - Hue represents color type (0-360°)
    - Saturation represents color intensity
    - Value represents brightness
    
    This separation makes it easier to detect specific
    colors under varying lighting conditions.
    """
    return """
    HSV Color Space Transformation
    ==============================
    - H (Hue): Color type, 0-180° in OpenCV
    - S (Saturation): Color intensity, 0-255
    - V (Value): Brightness, 0-255
    
    Used for detecting surgical tools by color:
    - Red marker → Scalpel
    - Blue marker → Forceps  
    - Green marker → Cautery
    """


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Mathematical Utilities Module - Test Suite")
    print("=" * 60)
    
    # Test Linear Algebra
    print("\n--- Linear Algebra Tests ---")
    p1 = np.array([0, 0, 0])
    p2 = np.array([3, 4, 0])
    print(f"Euclidean distance between {p1} and {p2}: {euclidean_distance(p1, p2)}")
    
    v = np.array([3, 4, 0])
    print(f"Magnitude of {v}: {vector_magnitude(v)}")
    print(f"Normalized: {normalize_vector(v)}")
    
    # Test Matrix Operations
    print("\n--- Matrix Operations Tests ---")
    R = rotation_matrix_2d(90)
    print(f"2D Rotation Matrix (90°):\n{R}")
    print(f"Determinant: {matrix_determinant(R)}")
    print(f"Trace: {matrix_trace(R)}")
    
    # Test Gaussian
    print("\n--- Probability Tests ---")
    print(f"Gaussian(x=0, μ=0, σ=1): {gaussian_1d(0, 0, 1):.4f}")
    print(f"Gaussian(x=1, μ=0, σ=1): {gaussian_1d(1, 0, 1):.4f}")
    
    # Test Statistics
    print("\n--- Statistics Tests ---")
    data = [85, 90, 78, 92, 88]
    print(f"Data: {data}")
    print(f"Mean: {calculate_mean(data):.2f}")
    print(f"Variance: {calculate_variance(data):.2f}")
    print(f"Standard Deviation: {calculate_standard_deviation(data):.2f}")
    print(f"Accuracy (4/5): {calculate_accuracy(4, 5):.1f}%")
    
    # Test Coordinate Transformation
    print("\n--- Coordinate Transformation Tests ---")
    pos_3d = map_2d_to_3d(640, 360, 1280, 720)
    print(f"2D (640, 360) → 3D: {pos_3d}")
    pos_2d = project_3d_to_2d(pos_3d, 1280, 720)
    print(f"3D {pos_3d} → 2D: {pos_2d}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
