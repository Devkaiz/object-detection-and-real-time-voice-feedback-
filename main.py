"""
Smart Navigation System for Blind People
Uses YOLOv5 + Google TTS (more reliable audio)
"""

import cv2
import torch
import numpy as np
import time
import warnings
from gtts import gTTS
import pygame
import os
from io import BytesIO

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class NavigationSystem:
    def __init__(self):
        # Initialize YOLOv5 model
        print("Loading YOLOv5 model...")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.conf = 0.45
        
        # Initialize pygame for audio playback
        print("Initializing audio system...")
        pygame.mixer.init()
        
        # Camera setup
        self.cap = None
        self.is_running = False
        self.frame_width = 640
        self.frame_height = 480
        
        # Timing
        self.last_guidance_time = 0
        self.guidance_interval = 3.0
        self.is_speaking = False
        
        # Test audio
        print("\nTesting audio...")
        self.speak_gtts("Navigation system initializing", test=True)
        print("Audio test complete!\n")
        
        # Path planning
        self.num_zones = 5
        
    def speak_gtts(self, text, test=False):
        """Text-to-speech using Google TTS - non-blocking"""
        if self.is_speaking and not test:
            return
            
        try:
            self.is_speaking = True
            print(f"[SPEAKING] {text}")
            
            # Generate speech in background thread
            def _speak():
                try:
                    tts = gTTS(text=text, lang='en', slow=False)
                    fp = BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    
                    pygame.mixer.music.load(fp)
                    pygame.mixer.music.play()
                    
                    # Wait for audio to finish
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    
                    print("[SPEECH COMPLETE]")
                except Exception as e:
                    print(f"[SPEECH ERROR] {e}")
                finally:
                    self.is_speaking = False
            
            # Run speech in separate thread if not test
            if test:
                _speak()
            else:
                from threading import Thread
                Thread(target=_speak, daemon=True).start()
            
        except Exception as e:
            print(f"[SPEECH ERROR] {e}")
            self.is_speaking = False
    
    def estimate_distance(self, bbox_height, object_class):
        """Estimate distance based on bounding box height"""
        typical_heights = {
            'person': 1.7, 'car': 1.5, 'chair': 0.9, 'dog': 0.6,
            'bicycle': 1.0, 'motorcycle': 1.2, 'bottle': 0.25,
        }
        
        real_height = typical_heights.get(object_class, 1.0)
        focal_length = 700
        
        if bbox_height > 0:
            distance = (real_height * focal_length) / bbox_height
            return max(0.3, min(distance, 15))
        return 5.0
    
    def get_position(self, bbox_center_x):
        """Determine object position"""
        if bbox_center_x < self.frame_width * 0.35:
            return "on your left"
        elif bbox_center_x > self.frame_width * 0.65:
            return "on your right"
        else:
            return "ahead"
    
    def plan_path(self, detections):
        """Analyze zones and suggest walking direction"""
        zone_width = self.frame_width / self.num_zones
        zone_scores = [100] * self.num_zones
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center_x = (x1 + x2) / 2
            zone_idx = int(center_x / zone_width)
            zone_idx = max(0, min(zone_idx, self.num_zones - 1))
            
            distance = det['distance']
            if distance < 1.0:
                penalty = 80
            elif distance < 2.0:
                penalty = 50
            elif distance < 3.5:
                penalty = 25
            else:
                penalty = 10
            
            zone_scores[zone_idx] = max(0, zone_scores[zone_idx] - penalty)
            
            if zone_idx > 0:
                zone_scores[zone_idx - 1] = max(0, zone_scores[zone_idx - 1] - penalty * 0.5)
            if zone_idx < self.num_zones - 1:
                zone_scores[zone_idx + 1] = max(0, zone_scores[zone_idx + 1] - penalty * 0.5)
        
        center_zone = self.num_zones // 2
        best_zone = zone_scores.index(max(zone_scores))
        best_score = zone_scores[best_zone]
        
        if best_score < 30:
            direction = "Stop! Too many obstacles"
        elif best_zone < center_zone - 1:
            direction = "Move left"
        elif best_zone > center_zone + 1:
            direction = "Move right"
        elif best_zone == center_zone:
            direction = "Continue forward"
        else:
            direction = "Path is good"
        
        return direction, best_score, zone_scores
    
    def generate_guidance(self, detections, path_direction, safety_score):
        """Generate audio guidance - simplified, no distances"""
        if len(detections) == 0:
            return "Path clear"
        
        detections.sort(key=lambda x: x['distance'])
        closest = detections[0]
        
        # Critical warning for very close objects
        if closest['distance'] < 1.0:
            return f"Stop! {closest['class']} very close {closest['position']}"
        
        # Simple guidance: object + direction
        guidance = f"{closest['class']} {closest['position']}"
        
        # Add path direction
        if path_direction != "Path is good" and path_direction != "Continue forward":
            guidance += f". {path_direction}"
        
        return guidance
    
    def draw_path_overlay(self, frame, zone_scores):
        """Draw path planning overlay"""
        zone_width = self.frame_width / self.num_zones
        overlay = frame.copy()
        
        for i, score in enumerate(zone_scores):
            x1 = int(i * zone_width)
            x2 = int((i + 1) * zone_width)
            
            if score > 70:
                color = (0, 255, 0)
            elif score > 40:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            
            cv2.rectangle(overlay, (x1, self.frame_height - 60), 
                         (x2, self.frame_height), color, -1)
            cv2.putText(overlay, f"{score}", (x1 + 10, self.frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    def process_frame(self, frame):
        """Process frame with YOLO"""
        results = self.model(frame)
        detections = []
        
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            class_name = self.model.names[int(cls)]
            
            bbox_height = y2 - y1
            bbox_center_x = (x1 + x2) / 2
            
            distance = self.estimate_distance(bbox_height, class_name)
            position = self.get_position(bbox_center_x)
            
            detections.append({
                'class': class_name,
                'bbox': (x1, y1, x2, y2),
                'distance': distance,
                'position': position
            })
            
            if distance < 1.5:
                color = (0, 0, 255)
            elif distance < 3.0:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {distance:.1f}m"
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame, detections
    
    def run(self):
        """Main loop"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        self.is_running = True
        self.speak_gtts("Navigation system activated")
        
        print("\n" + "="*50)
        print("    SMART NAVIGATION SYSTEM")
        print("="*50)
        print("Q - Quit | S - Toggle speech | C - Toggle camera")
        print("="*50 + "\n")
        
        speech_enabled = True
        show_camera = True
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                processed_frame, detections = self.process_frame(frame)
                path_direction, safety_score, zone_scores = self.plan_path(detections)
                processed_frame = self.draw_path_overlay(processed_frame, zone_scores)
                
                # Audio guidance
                current_time = time.time()
                if speech_enabled and not self.is_speaking:
                    if (current_time - self.last_guidance_time) > self.guidance_interval:
                        guidance = self.generate_guidance(detections, path_direction, safety_score)
                        print(f"[{time.strftime('%H:%M:%S')}] {guidance}")
                        self.speak_gtts(guidance)
                        self.last_guidance_time = current_time
                
                # Display
                if show_camera:
                    cv2.putText(processed_frame, f"Objects: {len(detections)}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"Safety: {safety_score}%", 
                               (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"{path_direction}", 
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow('Navigation System', processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    speech_enabled = not speech_enabled
                    status = "enabled" if speech_enabled else "disabled"
                    print(f"\nSpeech {status}")
                    self.speak_gtts(f"Speech {status}")
                elif key == ord('c'):
                    show_camera = not show_camera
                    if not show_camera:
                        cv2.destroyAllWindows()
        
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.speak_gtts("Navigation system deactivated")
        pygame.mixer.quit()
        print("\nSystem stopped\n")


def main():
    print("\n" + "="*50)
    print("  SMART NAVIGATION SYSTEM")
    print("  Using Google Text-to-Speech")
    print("="*50 + "\n")
    
    nav_system = NavigationSystem()
    nav_system.run()


if __name__ == "__main__":
    main()