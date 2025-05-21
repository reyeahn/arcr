# SAM 2.1 Webcam Multi-Object Tracking

A real-time interactive application for segmenting and tracking objects in webcam video using Meta AI's SAM 2.1 (Segment Anything Model 2.1).

## Overview

This project demonstrates the capabilities of SAM 2.1 in a real-time webcam environment, allowing interactive segmentation and tracking for multiple objects simultaneously.

## New Features in Optimized Version

- **Multithreaded Processing**: Separate threads for capture, inference, and rendering
- **Model Quantization**: INT8/FP16 quantization for reduced memory usage and faster inference
- **Adaptive Frame Skipping**: Automatically skip frames to maintain responsiveness
- **Memory Management**: Periodic CUDA cache clearing to prevent memory leaks
- **Multiple Precision Options**: Choose between float32, bfloat16, or float16 for inference

## Requirements

- Python 3.10 or higher
- PyTorch 2.5.1 or higher
- CUDA-capable GPU (recommended)
- Webcam

### Dependencies

See `pypi_requirements.txt` for a complete list of dependencies.

## Installation

1. Clone the repository:

   git clone https://github.com/yujie-tao/ARCreativity-ryan.git
   
   cd sam2

2. Create conda environment:

   conda create -n realtime_sam python=3.11

   conda activate realtime_sam


3. Install the package:

   pip install -e .


4. Install additional requirements:

   pip install -r pypi_requirements.txt


5. Install torch with cuda:

   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126


6. Download the model checkpoints:

   cd checkpoints

   ./download_ckpts.sh
   
   cd ..


## Usage

### Optimized Version

Run the optimized webcam application with:

python sam21_webcam_optimized.py --use_vos_optimized --use_quantized --precision bfloat16 --device cuda:0


### Model Quantization

Quantize a model for improved performance:


python sam21_quantize.py --checkpoint ./checkpoints/sam2.1_hiera_small.pt --output ./checkpoints/sam2.1_hiera_small_int8.pt --precision int8 --calibration_images ./path/to/images


### Command-line Arguments

- `--checkpoint`: Path to SAM 2.1 checkpoint
- `--model_config`: Path to model configuration file
- `--device`: Device to run inference on
- `--webcam_id`: Camera device ID
- `--width`: Width of the display frame
- `--height`: Height of the display frame
- `--use_vos_optimized`: Use VOS optimized camera predictor
- `--precision`: Precision for model inference (float32, bfloat16, float16)
- `--use_quantized`: Use quantized model


### Controls

- **P**: Switch to point prompt mode
- **B**: Switch to box prompt mode
- **Space**: Toggle tracking on/off
- **C**: Clear all objects and prompts
- **Q**: Quit application

### Interaction Guide

1. **Point Mode**:
   - Left-click to add positive points (include in mask)
   - Right-click to add negative points (exclude from mask)

2. **Box Mode**:
   - Click and drag to draw a bounding box around an object

3. **Tracking**:
   - Press Space to start/pause tracking
   - Objects are assigned unique IDs and colors

## Technical Details

### Architecture

1. **SAM 2.1 Model**: The core segmentation model from Meta AI
2. **Camera Predictor**: Specialized predictor for real-time webcam input
3. **Visualization Layer**: Handles UI elements and mask visualization
4. **Input Processing**: Manages user interaction via mouse and keyboard

### Performance Optimization

- **Multithreading**: Separate threads for capture, inference, and rendering
- **Model Quantization**: INT8/FP16 quantization for reduced memory usage
- **VOS Optimization**: Uses `torch.compile` for the entire model pipeline
- **Memory Management**: Efficient handling of multiple objects
- **FPS Monitoring**: Real-time performance tracking with rolling average

### Implementation Notes

- The application uses OpenCV for webcam capture and display
- Torch inference mode and autocast are used for optimal GPU performance
- Mask visualization uses alpha blending for better visibility
- Object tracking maintains state across frames
