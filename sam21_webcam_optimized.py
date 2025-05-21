"""
Usage:
python sam21_webcam_optimized.py --use_vos_optimized
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import cv2
import torch
import matplotlib
import threading
import queue
from collections import deque

# Use a non-interactive backend to avoid warnings
matplotlib.use('Agg')
import matplotlib.colors as mcolors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("sam21-webcam-optimized")

# Check for required dependencies
try:
    import torch
    logger.info(f"PyTorch version: {torch.__version__}")
    if torch.__version__ < "2.5.1":
        logger.warning("PyTorch version lower than 2.5.1. Full model compilation may not work properly.")
except ImportError:
    logger.error("PyTorch not found. Please install PyTorch 2.5.1 or higher.")
    sys.exit(1)

# Parse arguments
parser = argparse.ArgumentParser(description="SAM 2.1 Webcam Multi-Object Tracking (Optimized)")
parser.add_argument("--checkpoint", type=str, default="./checkpoints/sam2.1_hiera_small.pt",
                    help="Path to SAM 2.1 checkpoint")
parser.add_argument("--model_config", type=str, default="configs/sam2.1/sam2.1_hiera_s.yaml",
                    help="Path to model configuration file")
parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                    help="Device to run inference on")
parser.add_argument("--webcam_id", type=int, default=0, 
                    help="Camera device ID")
parser.add_argument("--width", type=int, default=640,
                    help="Width of the display frame")
parser.add_argument("--height", type=int, default=480,
                    help="Height of the display frame")
parser.add_argument("--use_vos_optimized", action="store_true",
                    help="Use VOS optimized camera predictor with model compilation")
parser.add_argument("--precision", type=str, choices=["float32", "bfloat16", "float16"], default="bfloat16",
                    help="Precision for model inference")
parser.add_argument("--use_quantized", action="store_true",
                    help="Use quantized model for faster inference")
parser.add_argument("--max_queue_size", type=int, default=5,
                    help="Maximum size for frame queues")
parser.add_argument("--record", action="store_true",
                    help="Record the tracking session with mask overlays")
parser.add_argument("--output_dir", type=str, default="output/recordings",
                    help="Directory to save recordings")
parser.add_argument("--vis_scale", type=float, default=1.0,
                    help="Scale factor for visualization (0.5 = half resolution, faster)")
args = parser.parse_args()

# Global variables for interaction
selected_points = []
selected_labels = []
current_bbox = []
is_drawing_box = False
prompt_mode = "point"  # 'point' or 'box'
next_object_id = 1
is_tracking = False
show_fps = True
fps_history = []
if_initialized = False
exit_flag = False
is_recording = False
recording_started = False
recording_frames = []

# Queues for thread communication
frame_queue = queue.Queue(maxsize=args.max_queue_size)
result_queue = queue.Queue(maxsize=args.max_queue_size)
prompt_queue = queue.Queue()

# Pre-define colors for visualization
COLORS = [
    (31, 119, 180),   # Blue
    (255, 127, 14),   # Orange
    (44, 160, 44),    # Green
    (214, 39, 40),    # Red
    (148, 103, 189),  # Purple
    (140, 86, 75),    # Brown
    (227, 119, 194),  # Pink
    (127, 127, 127),  # Gray
    (188, 189, 34),   # Olive
    (23, 190, 207),   # Cyan
    (174, 199, 232),  # Light blue
    (255, 187, 120),  # Light orange
    (152, 223, 138),  # Light green
    (255, 152, 150),  # Light red
    (197, 176, 213),  # Light purple
    (196, 156, 148),  # Light brown
    (247, 182, 210),  # Light pink
    (199, 199, 199),  # Light gray
    (219, 219, 141),  # Light olive
    (158, 218, 229),  # Light cyan
]

def get_color(idx):
    """Generate a unique color for the given object ID"""
    return COLORS[idx % len(COLORS)]

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for interactive prompting"""
    global selected_points, selected_labels, current_bbox, prompt_mode, is_drawing_box
    
    # Ensure coordinates are within bounds
    x = max(0, min(x, args.width - 1))
    y = max(0, min(y, args.height - 1))
    
    if prompt_mode == "point":
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add positive point with left click
            selected_points.append([x, y])
            selected_labels.append(1)  # positive label
            logger.info(f"Added positive point at ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Add negative point with right click
            selected_points.append([x, y])
            selected_labels.append(0)  # negative label
            logger.info(f"Added negative point at ({x}, {y})")
            
    elif prompt_mode == "box":
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start box with left click
            current_bbox = [x, y]
            is_drawing_box = True
            logger.info(f"Starting box at ({x}, {y})")
            
        elif event == cv2.EVENT_MOUSEMOVE and is_drawing_box:
            # Update box while dragging
            if len(current_bbox) == 2:
                current_bbox.append(x)
                current_bbox.append(y)
            else:
                current_bbox[2] = x
                current_bbox[3] = y
            
        elif event == cv2.EVENT_LBUTTONUP and is_drawing_box:
            # Complete box with left release
            if len(current_bbox) == 2:
                current_bbox.append(x)
                current_bbox.append(y)
            else:
                current_bbox[2] = x
                current_bbox[3] = y
            is_drawing_box = False
            
            # Ensure the box has a minimum size
            min_size = 5
            if abs(current_bbox[2] - current_bbox[0]) < min_size:
                current_bbox[2] = current_bbox[0] + min_size
            if abs(current_bbox[3] - current_bbox[1]) < min_size:
                current_bbox[3] = current_bbox[1] + min_size
                
            logger.info(f"Completed box from ({current_bbox[0]}, {current_bbox[1]}) to ({current_bbox[2]}, {current_bbox[3]})")

def check_and_create_directories():
    """Create necessary directories for output"""
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/frames", exist_ok=True)
    if args.record:
        os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Created output directories")

def load_sam_model():
    """Load the SAM 2.1 model and build the camera predictor"""
    try:
        # Import SAM 2.1 modules
        sys.path.append("./sam2")  # Adjust if the repo is located elsewhere
        from sam2.build_sam import build_sam2_camera_predictor
        
        # Determine checkpoint path based on quantization
        checkpoint_path = args.checkpoint
        if args.use_quantized:
            # Try to find a quantized version of the checkpoint
            base_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
            quantized_path = os.path.join(os.path.dirname(args.checkpoint), f"{base_name}_quantized.pt")
            if os.path.exists(quantized_path):
                checkpoint_path = quantized_path
                logger.info(f"Using quantized model: {checkpoint_path}")
            else:
                logger.warning(f"Quantized model not found at {quantized_path}. Using standard model.")
        
        logger.info(f"Loading SAM 2.1 model from {checkpoint_path}")
        logger.info(f"Using device: {args.device}")
        logger.info(f"Using precision: {args.precision}")
        
        # Build camera predictor (using VOS optimized version if requested)
        logger.info("Building SAM 2.1 camera predictor")
        predictor = build_sam2_camera_predictor(
            args.model_config, 
            checkpoint_path,
            device=args.device,
            vos_optimized=args.use_vos_optimized
        )
        
        # Convert model to specified precision
        if args.precision == "bfloat16":
            predictor = predictor.to(dtype=torch.bfloat16)
        elif args.precision == "float16":
            predictor = predictor.to(dtype=torch.float16)
        
        # Add diagnostic logging
        logger.info(f"Model device: {next(predictor.parameters()).device}")
        logger.info(f"Model dtype: {next(predictor.parameters()).dtype}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
        return predictor
    except Exception as e:
        logger.error(f"Failed to load SAM 2.1 model: {e}")
        sys.exit(1)

def setup_webcam():
    """Initialize the webcam capture"""
    logger.info(f"Setting up webcam with ID {args.webcam_id}")
    cap = cv2.VideoCapture(args.webcam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    if not cap.isOpened():
        logger.error(f"Failed to open webcam with ID {args.webcam_id}")
        sys.exit(1)
        
    logger.info(f"Webcam initialized with resolution {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cap

def reset_prompts():
    """Reset all prompts and selection state"""
    global selected_points, selected_labels, current_bbox, is_drawing_box
    selected_points = []
    selected_labels = []
    current_bbox = []
    is_drawing_box = False

def prepare_points_prompt(points, labels):
    """Prepare point prompts for SAM 2.1"""
    if not points:
        return None, None
    
    points_np = np.array(points, dtype=np.float32)
    labels_np = np.array(labels, np.int32)
    return points_np, labels_np

def prepare_box_prompt(box):
    """Prepare box prompt for SAM 2.1"""
    if len(box) != 4:
        return None
    
    # Convert to [x0, y0, x1, y1] format
    x0, y0 = min(box[0], box[2]), min(box[1], box[3])
    x1, y1 = max(box[0], box[2]), max(box[1], box[3])
    
    # Ensure the box has a minimum size
    min_size = 5
    if x1 - x0 < min_size:
        x1 = x0 + min_size
    if y1 - y0 < min_size:
        y1 = y0 + min_size
    
    # Ensure the box is within image bounds
    x0 = max(0, x0)
    y0 = max(0, y0)
    
    box_np = np.array([x0, y0, x1, y1], dtype=np.float32)
    return box_np

def draw_interface(frame, fps=None):
    """Draw interface elements on the frame"""
    # Draw current points
    for i, point in enumerate(selected_points):
        color = (0, 255, 0) if selected_labels[i] == 1 else (0, 0, 255)
        cv2.circle(frame, tuple(point), 5, color, -1)
    
    # Draw current box
    if len(current_bbox) >= 4:
        cv2.rectangle(frame, 
                     (current_bbox[0], current_bbox[1]), 
                     (current_bbox[2], current_bbox[3]), 
                     (255, 255, 0), 2)
    elif len(current_bbox) == 2 and is_drawing_box:
        # Draw line while dragging
        cv2.circle(frame, (current_bbox[0], current_bbox[1]), 5, (255, 255, 0), -1)
    
    # Show prompt mode
    text = f"Mode: {prompt_mode.upper()}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show tracking status
    status = "TRACKING" if is_tracking else "PAUSED"
    cv2.putText(frame, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show FPS if available
    if fps is not None and show_fps:
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show recording status if enabled
    if args.record:
        rec_status = "REC ●" if is_recording else "REC ○"
        cv2.putText(frame, rec_status, (frame.shape[1] - 120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if is_recording else (200, 200, 200), 2)
    
    # Show controls
    controls = [
        "P: Point Mode",
        "B: Box Mode",
        "Space: Track/Pause",
        "C: Clear All",
        "F: Toggle FPS",
        "R: Toggle Recording" if args.record else "",
        "Q: Quit"
    ]
    controls = [c for c in controls if c]  # Remove empty strings
    
    for i, control in enumerate(controls):
        y = frame.shape[0] - 20 * (len(controls) - i)
        cv2.putText(frame, control, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

def create_overlay_mask(frame, masks, object_ids):
    """Create a visualization mask for overlaying on the frame"""
    if masks is None or object_ids is None or len(object_ids) == 0:
        return None
    
    h, w = frame.shape[:2]
    mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Create a combined mask for faster processing
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    id_map = np.zeros((h, w), dtype=np.uint8)
    
    # Process each mask
    for i, obj_id in enumerate(object_ids):
        try:
            # Get the mask
            if isinstance(masks, torch.Tensor):
                # Process a tensor
                if masks.dim() > 3:  # Batch of masks
                    mask_tensor = masks[i, 0]
                else:
                    mask_tensor = masks[i]
                # Convert to numpy and threshold
                mask_np = mask_tensor.detach().cpu().numpy()
            else:
                # Already numpy array
                mask_np = masks[i]
            
            # Threshold the mask
            binary_mask = mask_np > 0
            
            # Ensure correct shape
            if binary_mask.shape != (h, w):
                temp_mask = np.zeros((h, w), dtype=bool)
                # Handle different mask shapes gracefully
                if len(binary_mask.shape) > 0:
                    # Resize to frame dimensions
                    resize_h, resize_w = binary_mask.shape[-2:]
                    if resize_h > 0 and resize_w > 0:
                        mask_resized = cv2.resize(
                            binary_mask.astype(np.uint8), 
                            (w, h),
                            interpolation=cv2.INTER_NEAREST
                        )
                        temp_mask = mask_resized > 0
                binary_mask = temp_mask
            
            # Update combined mask and id map
            combined_mask[binary_mask] = 1
            id_map[binary_mask] = obj_id % 255  # Limit to 255 for uint8
            
            # Find center for label (only for the first 5 objects to save time)
            if i < 5 and np.any(binary_mask):
                y_indices, x_indices = np.where(binary_mask)
                if len(y_indices) > 0 and len(x_indices) > 0:  # Make sure we have points
                    center_x = int(np.mean(x_indices))
                    center_y = int(np.mean(y_indices))
                    cv2.putText(mask_overlay, f"ID: {obj_id}", (center_x, center_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
        except Exception as e:
            logger.debug(f"Error processing mask for object {obj_id}: {e}")
            continue
    
    # Apply colors to the combined mask all at once
    mask_indices = np.where(combined_mask > 0)
    for i in range(len(mask_indices[0])):
        y, x = mask_indices[0][i], mask_indices[1][i]
        obj_id = id_map[y, x]
        color = get_color(obj_id)
        mask_overlay[y, x] = color
    
    return mask_overlay

def overlay_masks(frame, masks, object_ids):
    """Overlay mask visualization on frame"""
    if masks is None or object_ids is None or len(object_ids) == 0:
        return frame
    
    # Scale down for visualization if requested
    h, w = frame.shape[:2]
    vis_scale = args.vis_scale
    
    if vis_scale < 1.0:
        # Scale down frame for faster processing
        small_h, small_w = int(h * vis_scale), int(w * vis_scale)
        small_frame = cv2.resize(frame, (small_w, small_h))
        
        # Create mask overlay at smaller resolution
        mask_overlay = create_overlay_mask(small_frame, masks, object_ids)
        if mask_overlay is None:
            return frame
            
        # Blend with scaled frame
        alpha = 0.5
        result_small = cv2.addWeighted(small_frame, 1.0, mask_overlay, alpha, 0)
        
        # Scale back up to original size
        result = cv2.resize(result_small, (w, h))
        return result
    else:
        # Create mask overlay at original resolution
        mask_overlay = create_overlay_mask(frame, masks, object_ids)
        if mask_overlay is None:
            return frame
        
        # Blend with original frame
        alpha = 0.5
        return cv2.addWeighted(frame, 1.0, mask_overlay, alpha, 0)

def calculate_fps(start_time, end_time):
    """Calculate frames per second"""
    time_diff = end_time - start_time
    # Prevent division by zero
    if time_diff <= 0:
        return np.mean(fps_history) if fps_history else 0.0
        
    fps = 1.0 / time_diff
    fps_history.append(fps)
    
    # Keep only the last 30 FPS values for a smoother average
    if len(fps_history) > 30:
        fps_history.pop(0)
    
    return np.mean(fps_history)

def setup_video_writer(frame):
    """Set up video writer for recording"""
    h, w = frame.shape[:2]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(args.output_dir, f"tracking_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, 20.0, (w, h)), output_path

def capture_thread_function(cap):
    """Thread function for capturing frames from webcam"""
    global exit_flag
    
    logger.info("Starting capture thread")
    
    # Add timing diagnostics
    capture_times = []
    queue_times = []
    
    while not exit_flag:
        try:
            # Measure capture time
            capture_start = time.time()
            ret, frame = cap.read()
            capture_time = time.time() - capture_start
            capture_times.append(capture_time)
            
            if not ret:
                logger.error("Failed to capture frame from webcam")
                time.sleep(0.1)  # Short delay to prevent CPU spinning
                continue
                
            # Measure queue processing time
            queue_start = time.time()
            # If queue is full, remove oldest frame to make room
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            frame_queue.put((frame, time.time()), block=False)
            queue_time = time.time() - queue_start
            queue_times.append(queue_time)
            
            # Log timing stats every 100 frames
            if len(capture_times) % 100 == 0:
                avg_capture = np.mean(capture_times[-100:]) * 1000
                avg_queue = np.mean(queue_times[-100:]) * 1000
                logger.info(f"Capture thread timing (ms): Capture={avg_capture:.1f}, Queue={avg_queue:.1f}")
                # Keep the lists from growing too large
                if len(capture_times) > 1000:
                    capture_times = capture_times[-500:]
                    queue_times = queue_times[-500:]
                
        except Exception as e:
            logger.error(f"Error in capture thread: {e}")
            time.sleep(0.1)  # Short delay to prevent CPU spinning
    
    logger.info("Capture thread terminated")

def inference_thread_function(predictor):
    """Thread function for running inference"""
    global exit_flag, if_initialized, is_tracking, next_object_id
    
    logger.info("Starting inference thread")
    
    # Set up autocast for precision
    precision_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16
    }[args.precision]
    
    obj_ids = []
    masks = None
    
    # Add timing diagnostics
    inference_times = []
    tracking_times = []
    
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=precision_dtype):
        while not exit_flag:
            try:
                # Check for user prompts first (higher priority)
                try:
                    prompt_type, prompt_data = prompt_queue.get_nowait()
                    
                    if prompt_type == "reset":
                        predictor.reset_state()
                        if_initialized = False
                        next_object_id = 1
                        obj_ids = []
                        masks = None
                        logger.info("Reset predictor state")
                        
                    elif if_initialized:
                        if prompt_type == "points":
                            points_np, labels_np = prompt_data
                            start_time = time.time()
                            with torch.amp.autocast(device_type="cuda", dtype=precision_dtype):
                                _, obj_ids, masks = predictor.add_new_points(
                                    frame_idx=0,
                                    obj_id=next_object_id,
                                    points=points_np,
                                    labels=labels_np
                                )
                            inference_time = time.time() - start_time
                            inference_times.append(inference_time)
                            logger.info(f"Point inference time: {inference_time*1000:.1f}ms")
                            next_object_id += 1
                            logger.info(f"Added new object with ID {next_object_id-1} using point prompt")
                            
                        elif prompt_type == "box":
                            bbox_np = prompt_data
                            start_time = time.time()
                            with torch.amp.autocast(device_type="cuda", dtype=precision_dtype):
                                _, obj_ids, masks = predictor.add_new_prompt(
                                    frame_idx=0,
                                    obj_id=next_object_id,
                                    bbox=bbox_np
                                )
                            inference_time = time.time() - start_time
                            inference_times.append(inference_time)
                            logger.info(f"Box inference time: {inference_time*1000:.1f}ms")
                            next_object_id += 1
                            logger.info(f"Added new object with ID {next_object_id-1} using box prompt")
                    
                except queue.Empty:
                    pass
                
                # Process a frame if available
                try:
                    frame, timestamp = frame_queue.get(timeout=0.01)
                    
                    # Initialize with first frame if needed
                    if not if_initialized:
                        logger.info("Initializing with first frame")
                        predictor.load_first_frame(frame)
                        if_initialized = True
                        result_queue.put((frame, None, None, timestamp), block=False)
                        continue
                    
                    # Track if enabled
                    if is_tracking and if_initialized and len(obj_ids) > 0:
                        try:
                            start_time = time.time()
                            with torch.amp.autocast(device_type="cuda", dtype=precision_dtype):
                                new_obj_ids, new_masks = predictor.track(frame)
                                if new_obj_ids is not None and new_masks is not None:
                                    obj_ids = new_obj_ids
                                    masks = new_masks
                            tracking_time = time.time() - start_time
                            tracking_times.append(tracking_time)
                            
                            # Log timing stats every 30 frames
                            if len(tracking_times) % 30 == 0:
                                avg_tracking = np.mean(tracking_times[-30:]) * 1000
                                avg_inference = np.mean(inference_times[-30:]) * 1000 if inference_times else 0
                                logger.info(f"Timing stats - Tracking: {avg_tracking:.1f}ms, Inference: {avg_inference:.1f}ms")
                                
                        except Exception as track_err:
                            logger.error(f"Error during tracking: {str(track_err)}")
                            # Don't update masks on error, keep using previous ones
                            time.sleep(0.001)  # Short delay to prevent CPU spinning
                    
                    # Put results in queue
                    if result_queue.full():
                        try:
                            result_queue.get_nowait()
                        except queue.Empty:
                            pass
                            
                    result_queue.put((frame, obj_ids, masks, timestamp), block=False)
                    
                except queue.Empty:
                    time.sleep(0.001)  # Short delay to prevent CPU spinning
                    
            except Exception as e:
                import traceback
                logger.error(f"Error in inference thread: {e}\n{traceback.format_exc()}")
                time.sleep(0.1)  # Short delay to prevent CPU spinning
    
    logger.info("Inference thread terminated")

def display_thread_function():
    """Thread function for displaying results"""
    global exit_flag, selected_points, selected_labels, current_bbox, prompt_mode
    global is_drawing_box, is_tracking, show_fps, is_recording, recording_started
    
    logger.info("Starting display thread")
    
    # Create window and set mouse callback
    window_name = "SAM 2.1 Webcam Tracking (Optimized)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    video_writer = None
    output_path = None
    
    # Add timing diagnostics for display thread
    display_times = []
    overlay_times = []
    render_times = []
    total_pipeline_times = []
    last_frame_time = time.time()
    
    while not exit_flag:
        try:
            # Get processed frame with masks
            try:
                pipeline_start = time.time()
                frame, obj_ids, masks, timestamp = result_queue.get(timeout=0.1)
                
                # Process the frame for display
                processed_frame = frame.copy()
                
                # Only overlay masks if we have valid data
                overlay_start = time.time()
                if masks is not None and obj_ids is not None and len(obj_ids) > 0:
                    processed_frame = overlay_masks(processed_frame, masks, obj_ids)
                overlay_time = time.time() - overlay_start
                overlay_times.append(overlay_time)
                
                # Calculate FPS
                current_time = time.time()
                try:
                    # Calculate two types of FPS:
                    # 1. Model processing FPS (based on tracking time)
                    model_fps = calculate_fps(timestamp, current_time) if timestamp else None
                    
                    # 2. End-to-end pipeline FPS (includes all overhead)
                    pipeline_fps = 1.0 / (current_time - last_frame_time) if last_frame_time else None
                    last_frame_time = current_time
                    
                    # Use pipeline FPS for display as it's the actual frame rate user sees
                    fps = pipeline_fps
                    
                    # Record total pipeline time
                    pipeline_time = current_time - pipeline_start
                    total_pipeline_times.append(pipeline_time)
                    
                    # Log timing stats every 30 frames
                    if len(total_pipeline_times) % 30 == 0:
                        avg_pipeline = np.mean(total_pipeline_times[-30:]) * 1000
                        avg_overlay = np.mean(overlay_times[-30:]) * 1000
                        avg_render = np.mean(render_times[-30:]) * 1000 if render_times else 0
                        logger.info(f"Display thread timing (ms): Pipeline={avg_pipeline:.1f}, Overlay={avg_overlay:.1f}, Render={avg_render:.1f}")
                        
                except Exception as fps_err:
                    logger.error(f"Error calculating FPS: {fps_err}")
                    fps = None
                
                # Draw interface elements
                render_start = time.time()
                final_frame = draw_interface(processed_frame, fps)
                render_time = time.time() - render_start
                render_times.append(render_time)
                
                # Handle recording if enabled
                if args.record and is_recording:
                    if not recording_started:
                        video_writer, output_path = setup_video_writer(final_frame)
                        recording_started = True
                        logger.info(f"Started recording to {output_path}")
                    
                    if video_writer is not None:
                        video_writer.write(final_frame)
                
                # Display frame
                cv2.imshow(window_name, final_frame)
                
                # Track display time
                display_time = time.time() - pipeline_start
                display_times.append(display_time)
                
            except queue.Empty:
                continue
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quitting application")
                exit_flag = True
                break
            elif key == ord('p'):
                prompt_mode = "point"
                logger.info("Switched to point prompt mode")
            elif key == ord('b'):
                prompt_mode = "box"
                logger.info("Switched to box prompt mode")
            elif key == ord('c'):
                reset_prompts()
                prompt_queue.put(("reset", None))
                logger.info("Cleared all objects and prompts")
            elif key == ord(' '):
                is_tracking = not is_tracking
                logger.info(f"Tracking {'resumed' if is_tracking else 'paused'}")
            elif key == ord('f'):
                show_fps = not show_fps
                logger.info(f"FPS display {'enabled' if show_fps else 'disabled'}")
            elif key == ord('r') and args.record:
                is_recording = not is_recording
                if not is_recording and recording_started:
                    # Stop recording
                    if video_writer is not None:
                        video_writer.release()
                        logger.info(f"Saved recording to {output_path}")
                    recording_started = False
                logger.info(f"Recording {'started' if is_recording else 'stopped'}")
            
            # Process user inputs (points or box)
            if selected_points and prompt_mode == "point":
                points_np, labels_np = prepare_points_prompt(selected_points, selected_labels)
                prompt_queue.put(("points", (points_np, labels_np)))
                reset_prompts()
            
            elif len(current_bbox) == 4 and prompt_mode == "box" and not is_drawing_box:
                bbox_np = prepare_box_prompt(current_bbox)
                prompt_queue.put(("box", bbox_np))
                reset_prompts()
                
        except Exception as e:
            import traceback
            logger.error(f"Error in display thread: {e}\n{traceback.format_exc()}")
            time.sleep(0.1)  # Short delay to prevent CPU spinning
    
    # Clean up
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    logger.info("Display thread terminated")

def main():
    """Main application entry point"""
    global exit_flag
    
    logger.info("Starting SAM 2.1 Webcam Multi-Object Tracking (Optimized)")
    check_and_create_directories()
    
    # Load model
    predictor = load_sam_model()
    
    # Setup webcam
    cap = setup_webcam()
    
    try:
        # Create threads
        capture_thread = threading.Thread(target=capture_thread_function, args=(cap,))
        inference_thread = threading.Thread(target=inference_thread_function, args=(predictor,))
        display_thread = threading.Thread(target=display_thread_function)
        
        # Start threads
        capture_thread.start()
        inference_thread.start()
        display_thread.start()
        
        # Wait for threads to finish
        capture_thread.join()
        inference_thread.join()
        display_thread.join()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        exit_flag = True
    except Exception as e:
        logger.error(f"Error in main thread: {e}")
    finally:
        # Ensure we set exit flag to terminate threads
        exit_flag = True
        
        # Release resources
        cap.release()
        logger.info("Application terminated")

if __name__ == "__main__":
    main() 
