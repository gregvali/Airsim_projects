import numpy as np
import cv2
from collections import deque

class RobustGateDetector:
    """
    Robust gate detection with:
    - Temporal filtering (tracks gates across frames)
    - Optical flow for motion robustness
    - Configurable color thresholds
    - Noise rejection
    """
    
    def __init__(self, buffer_size=5):
        self.gate_colors = {
            'red': {
                'lower': np.array([0, 100, 100]),
                'upper': np.array([10, 255, 255]),
            },
            'green': {
                'lower': np.array([40, 100, 100]),
                'upper': np.array([80, 255, 255]),
            },
            'blue': {
                'lower': np.array([100, 100, 100]),
                'upper': np.array([130, 255, 255]),
            },
        }
        
        # Temporal filtering
        self.detection_history = deque(maxlen=buffer_size)
        self.gate_center_history = deque(maxlen=buffer_size)
        self.gate_area_history = deque(maxlen=buffer_size)
        
        # Optical flow tracking
        self.prev_frame = None
        self.prev_gray = None
        
        # Detection confidence
        self.min_contour_area = 200
        self.min_detection_confidence = 0.6
        
    def detect_gate_center(self, image):
        """
        Detect colored gate in image with temporal filtering.
        
        Returns:
            gate_center: (x, y) of gate center or None
            gate_area: Area of detected gate or 0
            confidence: Confidence score (0-1)
        """
        if image is None or len(image) == 0:
            self.detection_history.append(False)
            return None, 0, 0.0
        
        h, w = image.shape[:2]
        
        # Convert to HSV for robust color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        best_contour = None
        best_area = 0
        best_color = None
        
        # Search all color gates
        for color_name, thresholds in self.gate_colors.items():
            mask = cv2.inRange(hsv, thresholds['lower'], thresholds['upper'])
            
            # Morphological operations to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by size
                if area < self.min_contour_area:
                    continue
                
                # Prefer larger, more prominent gates
                if area > best_area:
                    best_area = area
                    best_contour = contour
                    best_color = color_name
        
        # Extract gate center if found
        gate_detected = False
        gate_center = None
        
        if best_contour is not None:
            M = cv2.moments(best_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                gate_center = np.array([cx, cy])
                gate_detected = True
        
        # Calculate confidence based on detection quality
        confidence = self._calculate_confidence(gate_area=best_area, image_shape=(h, w))
        
        # Temporal filtering - smooth detections across frames
        self.detection_history.append(gate_detected)
        if gate_detected:
            self.gate_center_history.append(gate_center)
            self.gate_area_history.append(best_area)
        
        # Use consensus from history
        filtered_gate_center, filtered_area = self._apply_temporal_filter()
        
        # Reduce confidence if consensus is weak
        if len(self.detection_history) > 0:
            detection_consensus = sum(self.detection_history) / len(self.detection_history)
            confidence *= detection_consensus
        
        return filtered_gate_center, filtered_area, confidence
    
    def _calculate_confidence(self, gate_area, image_shape):
        """Calculate detection confidence based on gate prominence"""
        h, w = image_shape
        image_area = h * w
        
        # Confidence from gate size (should be 5-50% of image)
        relative_size = gate_area / image_area
        size_confidence = 1.0 - np.clip(abs(relative_size - 0.15) / 0.15, 0, 1)
        
        return np.clip(size_confidence, 0, 1)
    
    def _apply_temporal_filter(self):
        """Apply temporal filtering to smooth gate detections"""
        if len(self.gate_center_history) == 0:
            return None, 0
        
        # Median filtering for robustness to outliers
        centers = np.array(self.gate_center_history)
        areas = np.array(self.gate_area_history)
        
        median_center = np.median(centers, axis=0).astype(int)
        median_area = np.median(areas)
        
        return median_center, median_area
    
    def get_desired_direction(self, gate_center, image_shape, confidence):
        """
        Calculate desired movement direction to gate.
        
        Returns:
            direction: Normalized (dx, dy) or None if no gate
            search_mode: Boolean, True if searching for gate
        """
        if gate_center is None or confidence < self.min_detection_confidence:
            return None, True  # Search mode
        
        h, w = image_shape
        image_center_x = w / 2
        image_center_y = h / 2
        
        # Deviation from center (normalized, -1 to +1)
        dx = (gate_center[0] - image_center_x) / image_center_x
        dy = (gate_center[1] - image_center_y) / image_center_y
        
        # Clamp to reasonable steering ranges
        dx = np.clip(dx, -1, 1)
        dy = np.clip(dy, -1, 1)
        
        return np.array([dx, dy]), False
    
    def calculate_optical_flow(self, image):
        """
        Calculate optical flow for motion-based corrections.
        
        Returns:
            flow_magnitude: Overall motion magnitude (0-1)
            flow_direction: Primary motion direction
        """
        if image is None:
            return 0.0, None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_frame = image
            return 0.0, None
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, n8=True, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Calculate flow magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Average magnitude (normalized)
        mean_magnitude = np.mean(magnitude) / 10.0  # Normalize by expected max
        mean_magnitude = np.clip(mean_magnitude, 0, 1)
        
        # Average angle
        mean_angle = np.mean(angle) if angle.size > 0 else 0
        
        self.prev_gray = gray
        self.prev_frame = image
        
        return mean_magnitude, mean_angle
