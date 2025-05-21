"""
Utility script for quantizing SAM 2.1 models to INT8 or FP16 precision
for improved memory usage and inference speed.

Usage:
python sam21_quantize.py --checkpoint ./checkpoints/sam2.1_hiera_small.pt --output ./checkpoints/sam2.1_hiera_small_int8.pt --precision int8
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("sam21-quantize")

def parse_args():
    parser = argparse.ArgumentParser(description="SAM 2.1 Model Quantization")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to SAM 2.1 checkpoint")
    parser.add_argument("--model_config", type=str, default="configs/sam2.1/sam2.1_hiera_s.yaml",
                        help="Path to model configuration file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for quantized model")
    parser.add_argument("--precision", type=str, choices=["int8", "fp16"], default="int8",
                        help="Quantization precision")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for quantization")
    parser.add_argument("--calibration_images", type=str, default=None,
                        help="Path to folder with calibration images (for INT8)")
    parser.add_argument("--num_calibration_images", type=int, default=100,
                        help="Number of calibration images to use")
    
    return parser.parse_args()

def load_model(config_file, ckpt_path, device):
    """Load the SAM 2.1 model"""
    try:
        # Import SAM 2.1 modules
        sys.path.append("./sam2")
        from sam2.build_sam import build_sam2
        
        logger.info(f"Loading SAM 2.1 model from {ckpt_path}")
        model = build_sam2(config_file, ckpt_path, device=device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load SAM 2.1 model: {e}")
        sys.exit(1)

def load_calibration_images(path, num_images=100):
    """Load calibration images for INT8 quantization"""
    import cv2
    import glob
    from pathlib import Path
    
    if path is None:
        logger.error("Calibration images path is required for INT8 quantization")
        sys.exit(1)
    
    image_paths = list(glob.glob(os.path.join(path, "*.jpg"))) + \
                  list(glob.glob(os.path.join(path, "*.jpeg"))) + \
                  list(glob.glob(os.path.join(path, "*.png")))
    
    if len(image_paths) == 0:
        logger.error(f"No images found in {path}")
        sys.exit(1)
    
    logger.info(f"Found {len(image_paths)} images, using {min(num_images, len(image_paths))} for calibration")
    
    images = []
    for img_path in image_paths[:num_images]:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    
    return images

def quantize_to_int8(model, calibration_images, device):
    """Quantize model to INT8 using static quantization"""
    logger.info("Quantizing model to INT8...")
    
    # Define quantization configuration
    qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    
    # Prepare the model for quantization
    model_prepared = prepare_fx(model, qconfig_mapping)
    
    # Calibrate with sample data
    logger.info("Calibrating with sample images...")
    with torch.no_grad():
        for i, img in enumerate(calibration_images):
            if i % 10 == 0:
                logger.info(f"Calibrating with image {i+1}/{len(calibration_images)}")
            
            # Preprocess image for the model
            from sam2.utils.transforms import ResizeLongestSide
            transform = ResizeLongestSide(model.image_encoder.img_size)
            input_image = transform.apply_image(img)
            input_image_torch = torch.as_tensor(input_image, device=device)
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
            
            # Run model forward pass for calibration
            with torch.no_grad():
                model_prepared.get_image_embeddings(input_image_torch)
    
    # Convert the model to quantized version
    model_quantized = convert_fx(model_prepared)
    
    return model_quantized

def quantize_to_fp16(model):
    """Convert model to FP16 precision"""
    logger.info("Converting model to FP16...")
    
    # Convert model parameters to half precision
    model_fp16 = model.half()
    
    return model_fp16

def save_quantized_model(model, output_path):
    """Save the quantized model"""
    logger.info(f"Saving quantized model to {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the model
    torch.save({"model": model.state_dict()}, output_path)
    
    logger.info("Quantized model saved successfully")

def main():
    args = parse_args()
    
    # Load the model
    model = load_model(args.model_config, args.checkpoint, args.device)
    
    # Quantize based on selected precision
    if args.precision == "int8":
        # Load calibration images for INT8 quantization
        calibration_images = load_calibration_images(args.calibration_images, args.num_calibration_images)
        quantized_model = quantize_to_int8(model, calibration_images, args.device)
    else:  # fp16
        quantized_model = quantize_to_fp16(model)
    
    # Save the quantized model
    save_quantized_model(quantized_model, args.output)
    
    # Report memory savings
    original_size = os.path.getsize(args.checkpoint) / (1024 * 1024)  # MB
    quantized_size = os.path.getsize(args.output) / (1024 * 1024)  # MB
    savings = (1 - quantized_size / original_size) * 100
    
    logger.info(f"Original model size: {original_size:.2f} MB")
    logger.info(f"Quantized model size: {quantized_size:.2f} MB")
    logger.info(f"Memory savings: {savings:.2f}%")

if __name__ == "__main__":
    main() 