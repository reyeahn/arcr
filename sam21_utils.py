"""
SAM 2.1 Multi-Object Tracking Utilities

This module provides utility functions for SAM 2.1 webcam implementation,
focusing on visualization, tracking, and performance monitoring.
"""

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import logging

logger = logging.getLogger("sam21-utils")

class FPSCounter:
    """FPS counter with rolling average"""
    
    def __init__(self, window_size=30):
        """Initialize FPS counter with specified window size"""
        self.frame_times = deque(maxlen=window_size)
        self.last_frame_time = None
    
    def start_frame(self):
        """Start timing a new frame"""
        self.last_frame_time = time.time()
    
    def end_frame(self):
        """End timing the current frame and update metrics"""
        if self.last_frame_time is None:
            return None
        
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.frame_times.append(frame_time)
        return current_time
    
    def get_fps(self):
        """Get current FPS based on rolling average"""
        if not self.frame_times:
            return 0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    
    def get_frame_time_ms(self):
        """Get average frame processing time in milliseconds"""
        if not self.frame_times:
            return 0
        
        return sum(self.frame_times) / len(self.frame_times) * 1000

class ObjectTracker:
    """Utility class for tracking multiple objects with SAM 2.1"""
    
    def __init__(self):
        """Initialize object tracker"""
        self.objects = {}  # Dictionary to store tracked objects by ID
        self.next_id = 0   # Counter for assigning unique IDs
        self.color_map = plt.cm.get_cmap('tab20', 20)  # Color map for visualization
    
    def add_object(self, mask, prompt_type, prompt):
        """Add a new object to tracking"""
        obj_id = self.next_id
        self.objects[obj_id] = {
            "mask": mask,
            "prompt_type": prompt_type,
            "prompt": prompt,
            "first_seen": time.time(),
            "last_updated": time.time(),
            "frames_visible": 1,
            "color": self._get_color(obj_id)
        }
        self.next_id += 1
        return obj_id
    
    def update_object(self, obj_id, mask):
        """Update an existing object's mask and metadata"""
        if obj_id in self.objects:
            self.objects[obj_id]["mask"] = mask
            self.objects[obj_id]["last_updated"] = time.time()
            self.objects[obj_id]["frames_visible"] += 1
            return True
        return False
    
    def remove_object(self, obj_id):
        """Remove an object from tracking"""
        if obj_id in self.objects:
            del self.objects[obj_id]
            return True
        return False
    
    def get_object_count(self):
        """Get the number of tracked objects"""
        return len(self.objects)
    
    def get_object_ids(self):
        """Get list of all tracked object IDs"""
        return list(self.objects.keys())
    
    def get_object_data(self, obj_id):
        """Get data for a specific object"""
        return self.objects.get(obj_id)
    
    def get_all_masks(self):
        """Get all masks with their corresponding object IDs"""
        masks = []
        ids = []
        for obj_id, obj_data in self.objects.items():
            masks.append(obj_data["mask"])
            ids.append(obj_id)
        return masks, ids
    
    def clear_all(self):
        """Clear all tracked objects"""
        self.objects.clear()
    
    def reset_ids(self):
        """Reset the ID counter"""
        self.next_id = 0
    
    def _get_color(self, idx):
        """Generate a unique color for visualization"""
        color_rgb = self.color_map(idx % 20)[:3]  # Get RGB values
        return [int(c * 255) for c in color_rgb]  # Convert to 0-255 range

class VisualizationHelper:
    """Helper class for visualization and UI elements"""
    
    @staticmethod
    def draw_points(frame, points, labels):
        """Draw prompt points on the frame"""
        for i, point in enumerate(points):
            color = (0, 255, 0) if labels[i] == 1 else (0, 0, 255)
            cv2.circle(frame, (int(point[0]), int(point[1])), 5, color, -1)
        return frame
    
    @staticmethod
    def draw_box(frame, box):
        """Draw bounding box on the frame"""
        if len(box) == 4:
            cv2.rectangle(frame, 
                        (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), 
                        (255, 255, 0), 2)
        return frame
    
    @staticmethod
    def draw_masks(frame, masks, colors):
        """Draw masks with unique colors and transparency"""
        if not masks or not colors:
            return frame
        
        # Create a copy for overlay
        overlay = frame.copy()
        mask_overlay = np.zeros_like(frame)
        
        for i, mask in enumerate(masks):
            # Create colored mask
            color = colors[i]
            colored_mask = np.zeros_like(frame)
            colored_mask[mask] = color
            
            # Add to mask overlay with alpha blending
            alpha = 0.5  # Transparency
            mask_overlay = cv2.addWeighted(mask_overlay, 1, colored_mask, alpha, 0)
        
        # Combine frame with mask overlay
        result = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)
        return result
    
    @staticmethod
    def draw_object_info(frame, masks, obj_ids, objects_data):
        """Draw object IDs and information on the frame"""
        for i, mask in enumerate(masks):
            obj_id = obj_ids[i]
            obj_data = objects_data.get(obj_id)
            
            if obj_data and np.any(mask):
                # Find mask center
                y_indices, x_indices = np.where(mask)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    center_x, center_y = int(np.mean(x_indices)), int(np.mean(y_indices))
                    
                    # Draw object ID
                    color = obj_data["color"]
                    cv2.putText(frame, f"ID: {obj_id}", (center_x, center_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw additional info if desired
                    frames_visible = obj_data.get("frames_visible", 0)
                    cv2.putText(frame, f"Frames: {frames_visible}", (center_x, center_y + 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    @staticmethod
    def draw_interface(frame, status):
        """Draw interface elements (status, controls, etc.)"""
        # Draw status information
        cv2.putText(frame, f"Mode: {status.get('mode', 'N/A')}", 
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Status: {status.get('tracking_status', 'N/A')}", 
                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Objects: {status.get('object_count', 0)}", 
                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw FPS if available
        if status.get('show_fps', True) and 'fps' in status:
            cv2.putText(frame, f"FPS: {status['fps']:.1f}", 
                      (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw controls
        controls = [
            "P: Point Mode",
            "B: Box Mode",
            "Space: Track/Pause",
            "C: Clear All",
            "F: Toggle FPS",
            "Q: Quit"
        ]
        for i, control in enumerate(controls):
            y = frame.shape[0] - 20 * (len(controls) - i)
            cv2.putText(frame, control, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame

class FrameProcessor:
    """Helper class for frame pre/post-processing"""
    
    @staticmethod
    def resize_frame(frame, width=None, height=None):
        """Resize frame while maintaining aspect ratio"""
        if width is None and height is None:
            return frame
        
        h, w = frame.shape[:2]
        
        if width is None:
            # Calculate width to maintain aspect ratio
            aspect = w / h
            width = int(height * aspect)
        elif height is None:
            # Calculate height to maintain aspect ratio
            aspect = h / w
            height = int(width * aspect)
        
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def normalize_frame(frame):
        """Normalize frame for better segmentation performance"""
        # Convert to RGB (SAM expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Ensure frame is in float32 format
        if frame_rgb.dtype != np.float32:
            frame_rgb = frame_rgb.astype(np.float32) / 255.0
        
        return frame_rgb
    
    @staticmethod
    def denormalize_frame(frame):
        """Convert normalized frame back to display format"""
        if frame.dtype == np.float32:
            frame = (frame * 255.0).astype(np.uint8)
        
        # Convert back to BGR for OpenCV display
        if frame.shape[-1] == 3:  # Check if it's a color image
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        return frame
    
    @staticmethod
    def apply_preprocessing(frame, clahe=False, denoise=False):
        """Apply preprocessing to improve segmentation quality"""
        if clahe:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
            l, a, b = cv2.split(lab)
            
            clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe_obj.apply(l)
            
            lab = cv2.merge((l, a, b))
            frame = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        
        if denoise:
            # Apply denoising
            frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        
        return frame

def mask_iou(mask1, mask2):
    """
    Calculate IoU (Intersection over Union) between two binary masks
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
    
    Returns:
        IoU score between 0 and 1
    """
    if not np.any(mask1) or not np.any(mask2):
        return 0.0
    
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

def save_tracking_results(frame, masks, obj_ids, output_dir, frame_idx, include_masks=True):
    """
    Save tracking results to disk
    
    Args:
        frame: Original frame
        masks: List of binary masks
        obj_ids: List of object IDs corresponding to masks
        output_dir: Directory to save results
        frame_idx: Frame index/number
        include_masks: Whether to save individual mask files
    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the original frame
    frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
    cv2.imwrite(frame_path, frame)
    
    # Save visualization with all masks
    if masks and obj_ids:
        # Create visualization helper to draw masks
        viz_helper = VisualizationHelper()
        
        # Get colors for each object
        colors = []
        for obj_id in obj_ids:
            color_palette = plt.cm.get_cmap('tab20', 20)
            color_rgb = color_palette(obj_id % 20)[:3]
            colors.append([int(c * 255) for c in color_rgb])
        
        # Draw masks on frame
        vis_frame = viz_helper.draw_masks(frame.copy(), masks, colors)
        vis_path = os.path.join(output_dir, f"vis_{frame_idx:06d}.jpg")
        cv2.imwrite(vis_path, vis_frame)
        
        # Save individual masks if requested
        if include_masks:
            masks_dir = os.path.join(output_dir, "masks")
            os.makedirs(masks_dir, exist_ok=True)
            
            for i, mask in enumerate(masks):
                obj_id = obj_ids[i]
                mask_path = os.path.join(masks_dir, f"frame_{frame_idx:06d}_obj_{obj_id:03d}.png")
                
                # Convert boolean mask to uint8 (0 or 255)
                mask_img = np.zeros(mask.shape, dtype=np.uint8)
                mask_img[mask] = 255
                
                # Save mask
                cv2.imwrite(mask_path, mask_img)
    
    # Save metadata
    if masks:
        import json
        metadata = {
            "frame_idx": frame_idx,
            "timestamp": time.time(),
            "objects": [
                {
                    "id": obj_id,
                    "area": int(np.sum(mask)),
                    "center_x": int(np.mean(np.where(mask)[1])) if np.any(mask) else -1,
                    "center_y": int(np.mean(np.where(mask)[0])) if np.any(mask) else -1
                }
                for mask, obj_id in zip(masks, obj_ids)
            ]
        }
        
        metadata_path = os.path.join(output_dir, f"meta_{frame_idx:06d}.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

def create_prompt_from_detection(detection, prompt_type="point"):
    """
    Create a prompt for SAM 2.1 from a detection (e.g., from an object detector)
    
    Args:
        detection: Dictionary with detection information (bbox, confidence, etc.)
        prompt_type: Type of prompt to create ("point" or "box")
    
    Returns:
        Dictionary with prompt information for SAM 2.1
    """
    import torch
    
    if prompt_type == "point":
        # Use center of bounding box as point prompt
        x1, y1, x2, y2 = detection["bbox"]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return {
            "point_coords": torch.Tensor([[center_x, center_y]]),
            "point_labels": torch.Tensor([1])  # Positive label
        }
    
    elif prompt_type == "box":
        # Use bounding box as box prompt
        x1, y1, x2, y2 = detection["bbox"]
        
        return {
            "boxes": torch.Tensor([[x1, y1, x2, y2]])
        }
    
    else:
        raise ValueError(f"Unsupported prompt type: {prompt_type}")