"""
Example: Real-time object detection on AirSim drone FPV camera stream.

This script shows how to:
1. Capture video from the drone's FPV camera
2. Run object detection (YOLO or color-based)
3. Visualize detections
4. Use detections for drone control
"""

import time
import airsimdroneracinglab as airsim
import cv2
import numpy as np
from object_detection import YOLOObjectDetector, CustomGateDetector

class DroneVisionDemo:
    """Demo showing object detection from drone FPV camera"""
    
    def __init__(self):
        self.client = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
        self.client.confirmConnection()
        
        # Choose detection method:
        # Option 1: YOLO (accurate, slower, requires ultralytics)
        try:
            self.detector = YOLOObjectDetector(model_size='n', confidence_threshold=0.5)
            self.detection_method = 'yolo'
            print("Using YOLO object detection")
        except:
            # Option 2: Color-based (fast, specific to colored gates)
            self.detector = CustomGateDetector()
            self.detection_method = 'color'
            print("Using color-based gate detection (YOLO not available)")
        
        # Output video writer (optional, for recording detections)
        self.out_video = None
        
    def get_fpv_frame(self):
        """Capture frame from front FPV camera"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])
            
            if responses and len(responses) > 0:
                response = responses[0]
                img_1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
                img = img_1d.reshape(response.height, response.width, 3)
                # Convert RGB to BGR for OpenCV
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error capturing frame: {e}")
        
        return None
    
    def run_detection_demo(self, duration=30.0, display=False):
        """
        Run object detection on FPV camera stream.
        
        Args:
            duration: How long to run (seconds)
            display: If True, saves visualization to file
        """
        print(f"\nRunning object detection for {duration} seconds...")
        print("Classes to detect (YOLO):")
        if hasattr(self.detector, 'class_names'):
            print(f"  {list(self.detector.class_names.values())[:10]}... (80 total)")
        print()
        
        start_time = time.time()
        frame_count = 0
        detections_total = {}
        
        while time.time() - start_time < duration:
            frame_count += 1
            
            # Get camera frame
            frame = self.get_fpv_frame()
            if frame is None:
                continue
            
            # Run detection
            if self.detection_method == 'yolo':
                detections = self.detector.detect(frame)
            else:  # color-based
                detections = self.detector.detect_gates(frame)
            
            # Draw detections
            if self.detection_method == 'yolo':
                annotated = self.detector.draw_detections(frame, detections)
            else:
                annotated = self.detector.draw_detections(frame, detections)
            
            # Add frame info
            cv2.putText(annotated, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated, f"Detections: {len(detections)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Log detections
            if detections:
                for det in detections[:3]:  # Show top 3
                    if self.detection_method == 'yolo':
                        print(f"  [{frame_count:04d}] {det['class']}: "
                              f"conf={det['confidence']:.2f}, area={det['area']}")
                        det_key = det['class']
                    else:
                        print(f"  [{frame_count:04d}] {det['color']}: "
                              f"area={det['area']}, circularity={det['circularity']:.2f}")
                        det_key = det['color']
                    
                    detections_total[det_key] = detections_total.get(det_key, 0) + 1
            
            # Save frame if display enabled
            if display:
                if self.out_video is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.out_video = cv2.VideoWriter(
                        '/mnt/user-data/outputs/detection_demo.mp4',
                        fourcc, 30.0, (w, h)
                    )
                
                self.out_video.write(annotated)
            
            time.sleep(0.01)
        
        # Print summary
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Detection Demo Complete")
        print(f"{'='*60}")
        print(f"Duration: {elapsed:.1f} seconds")
        print(f"Frames processed: {frame_count}")
        print(f"Frame rate: {frame_count/elapsed:.1f} Hz")
        print(f"\nDetections found:")
        for obj_class, count in sorted(detections_total.items(), 
                                       key=lambda x: x[1], reverse=True):
            print(f"  {obj_class}: {count} frames")
        
        if self.out_video:
            self.out_video.release()
            print(f"\nVideo saved to: /mnt/user-data/outputs/detection_demo.mp4")

def example_1_color_detection():
    """Example 1: Detect colored racing gates"""
    print("\n" + "="*60)
    print("Example 1: Color-Based Gate Detection")
    print("="*60)
    print("Looks for red, green, blue, and yellow colored gates")
    print("Fast, works offline, but less general-purpose\n")
    
    detector = CustomGateDetector()
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    # Take a single frame and detect
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])
    
    if responses:
        response = responses[0]
        img_1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img = img_1d.reshape(response.height, response.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Detect gates
        gates = detector.detect_gates(img)
        print(f"Found {len(gates)} gates:")
        for gate in gates:
            print(f"  {gate['color']}: area={gate['area']}, circularity={gate['circularity']:.2f}")
        
        # Draw and save
        annotated = detector.draw_detections(img, gates)
        cv2.imwrite('/mnt/user-data/outputs/color_detection_example.jpg', annotated)
        print(f"\nVisualization saved: /mnt/user-data/outputs/color_detection_example.jpg")

def example_2_yolo_detection():
    """Example 2: YOLO general object detection"""
    print("\n" + "="*60)
    print("Example 2: YOLO Object Detection")
    print("="*60)
    print("Detects 80 different object classes (people, cars, etc.)")
    print("More general, but slower (requires ultralytics library)\n")
    
    try:
        from object_detection import YOLOObjectDetector
        detector = YOLOObjectDetector(model_size='n')
        
        client = airsim.MultirotorClient()
        client.confirmConnection()
        
        # Take a single frame and detect
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])
        
        if responses:
            response = responses[0]
            img_1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            img = img_1d.reshape(response.height, response.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Detect objects
            detections = detector.detect(img)
            print(f"Found {len(detections)} objects:")
            for det in detections[:10]:
                print(f"  {det['class']}: confidence={det['confidence']:.2f}")
            
            # Draw and save
            annotated = detector.draw_detections(img, detections)
            cv2.imwrite('/mnt/user-data/outputs/yolo_detection_example.jpg', annotated)
            print(f"\nVisualization saved: /mnt/user-data/outputs/yolo_detection_example.jpg")
    
    except ImportError:
        print("YOLO not available. Install with: pip install ultralytics")

def example_3_streaming_detection():
    """Example 3: Real-time streaming detection"""
    print("\n" + "="*60)
    print("Example 3: Real-Time Streaming Detection")
    print("="*60)
    print("Continuous detection on drone video stream\n")
    
    demo = DroneVisionDemo()
    demo.run_detection_demo(duration=10.0, display=True)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
    else:
        example_num = "1"
    
    try:
        if example_num == "1":
            example_1_color_detection()
        elif example_num == "2":
            example_2_yolo_detection()
        elif example_num == "3":
            example_3_streaming_detection()
        else:
            print("Usage: python object_detection_demo.py [1|2|3]")
            print("  1 = Color-based gate detection")
            print("  2 = YOLO general object detection")
            print("  3 = Real-time streaming detection")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
