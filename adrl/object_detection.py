import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

class YOLOObjectDetector:
    """
    Real-time object detection using YOLOv8.
    
    Detects standard objects (person, car, dog, etc.) from drone FPV camera.
    Can be fine-tuned on custom dataset for specific objects like racing gates.
    """
    
    def __init__(self, model_size='n', confidence_threshold=0.5):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
                       nano is fastest, xlarge is most accurate
            confidence_threshold: 0.0-1.0, minimum confidence to report detection
        """
        # Load pretrained model (auto-downloads on first run)
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.confidence_threshold = confidence_threshold
        
        # Detection history for temporal filtering
        self.detection_history = deque(maxlen=5)
        
        # Class names (COCO dataset)
        self.class_names = self.model.names
        
    def detect(self, image, classes_of_interest=None):
        """
        Run object detection on image.
        
        Args:
            image: BGR numpy array from camera
            classes_of_interest: List of class names to filter (e.g., ['person', 'car'])
                                If None, returns all detections
        
        Returns:
            detections: List of dicts with:
                - 'class': class name
                - 'confidence': confidence score (0-1)
                - 'bbox': (x1, y1, x2, y2) pixel coordinates
                - 'center': (cx, cy) center of bounding box
                - 'area': pixel area of bbox
        """
        if image is None or len(image) == 0:
            return []
        
        # Run inference
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        
        for result in results:
            # Extract boxes
            boxes = result.boxes
            
            for box in boxes:
                # Get class ID and name
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]
                
                # Filter by class if specified
                if classes_of_interest is not None:
                    if class_name not in classes_of_interest:
                        continue
                
                # Get confidence
                confidence = float(box.conf[0])
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate center and area
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                detection = {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'center': (cx, cy),
                    'area': area,
                    'width': width,
                    'height': height,
                }
                
                detections.append(detection)
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def draw_detections(self, image, detections, line_thickness=2):
        """
        Draw bounding boxes on image for visualization.
        
        Args:
            image: BGR numpy array
            detections: List from detect()
            line_thickness: Thickness of bbox lines
        
        Returns:
            annotated_image: Image with drawn boxes
        """
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx, cy = det['center']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)
            
            # Draw center point
            cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)
            
            # Draw label
            label = f"{det['class']} {det['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 4),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated
    
    def get_largest_detection(self, detections, class_name=None):
        """
        Get the largest/most prominent detection.
        
        Args:
            detections: List from detect()
            class_name: Filter by specific class, or None for largest of any
        
        Returns:
            largest_detection: Single detection dict, or None if no detections
        """
        if not detections:
            return None
        
        if class_name is not None:
            detections = [d for d in detections if d['class'] == class_name]
        
        if not detections:
            return None
        
        return max(detections, key=lambda x: x['area'])
    
    def get_detections_by_class(self, detections):
        """
        Group detections by class name.
        
        Args:
            detections: List from detect()
        
        Returns:
            dict: {class_name: [detections...]}
        """
        grouped = {}
        for det in detections:
            class_name = det['class']
            if class_name not in grouped:
                grouped[class_name] = []
            grouped[class_name].append(det)
        
        return grouped


class CustomGateDetector:
    """
    Simpler detector for racing gates using color + shape detection.
    Useful if you want to avoid YOLO overhead or need very specific gate detection.
    
    Can detect gates by:
    1. Color (HSV thresholding)
    2. Shape (contour properties)
    3. Size (relative to image)
    """
    
    def __init__(self):
        # Define gate colors (HSV ranges)
        self.gate_colors = {
            'red': {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
            'green': {'lower': np.array([40, 100, 100]), 'upper': np.array([80, 255, 255])},
            'blue': {'lower': np.array([100, 100, 100]), 'upper': np.array([130, 255, 255])},
            'yellow': {'lower': np.array([20, 100, 100]), 'upper': np.array([40, 255, 255])},
        }
        
        self.min_contour_area = 500
        self.max_contour_area = 500000
        
    def detect_gates(self, image, colors=None):
        """
        Detect colored gates in image.
        
        Args:
            image: BGR numpy array
            colors: List of color names to detect, or None for all
        
        Returns:
            detections: List of dicts with 'color', 'center', 'area', 'bbox'
        """
        if image is None or len(image) == 0:
            return []
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detections = []
        
        if colors is None:
            colors = list(self.gate_colors.keys())
        
        for color_name in colors:
            if color_name not in self.gate_colors:
                continue
            
            thresholds = self.gate_colors[color_name]
            mask = cv2.inRange(hsv, thresholds['lower'], thresholds['upper'])
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by size
                if area < self.min_contour_area or area > self.max_contour_area:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w // 2, y + h // 2
                
                # Calculate circularity (how circular is the shape)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)
                
                detection = {
                    'color': color_name,
                    'center': (cx, cy),
                    'area': area,
                    'bbox': (x, y, x + w, y + h),
                    'width': w,
                    'height': h,
                    'circularity': circularity,
                }
                
                detections.append(detection)
        
        # Sort by area (largest first)
        detections.sort(key=lambda x: x['area'], reverse=True)
        
        return detections
    
    def draw_detections(self, image, detections):
        """Draw detected gates on image"""
        annotated = image.copy()
        
        color_map = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
        }
        
        for det in detections:
            cx, cy = det['center']
            x1, y1, x2, y2 = det['bbox']
            color = color_map.get(det['color'], (255, 255, 255))
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw center
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 255), -1)
            
            # Draw label
            label = f"{det['color']} {det['area']}"
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
