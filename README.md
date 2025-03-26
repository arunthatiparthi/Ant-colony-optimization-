# Artificial Bee Colony (ABC) Steganography using Least Significant Bit (LSB)

## Overview
This project implements **steganography** using the **Least Significant Bit (LSB) method**, enhanced with **Artificial Bee Colony (ABC) optimization** for pixel selection. The algorithm embeds a secret message inside an image while ensuring minimal distortion, making detection difficult.

## Features
✅ **LSB-Based Steganography** – Hides message bits in image pixel values.  
✅ **ABC Optimization** – Selects pixels efficiently for minimal distortion.  
✅ **Image Quality Analysis** – Compares original and stego images using **SSIM, histogram similarity, and pixel difference**.  
✅ **Stego Image Extraction** – Recovers the hidden message accurately.  
✅ **Matplotlib Visualization** – Displays original and stego images side by side.  

## Dependencies
To run this project, install the required dependencies:
```sh
pip install numpy opencv-python scikit-image matplotlib
```

## Usage

### 1️⃣ Embed a Secret Message
1. Place an image (e.g., `image.jpg`) in the project directory.
2. Run the script and enter a message to hide:
```sh
python abc_steganography.py
```
3. The script will generate `stego_image.png` with the hidden message.

### 2️⃣ Extract a Secret Message
The script automatically extracts the message from the **stego image** and prints it:
```sh
 Extracted Message: [Your hidden text]
```

### 3️⃣ Image Quality Comparison
The script computes:
- **Pixel Difference** – Number of changed pixels.
- **SSIM Score** – Structural similarity index (1 = identical, 0 = completely different).
- **Histogram Similarity** – Measures how closely histograms match.

## Algorithm Details
1. **Convert message to binary**.
2. **Apply ABC Optimization** to select optimal pixels.
3. **Modify LSB** of selected pixels.
4. **Save the stego image**.
5. **Extract and reconstruct the message** from the stego image.

## Results & Analysis
The system ensures:
- **Minimal perceptual changes** to the image.
- **High extraction accuracy** using ABC-selected pixels.
- **Stego image remains visually unchanged**.

🚀 **Now you can securely hide messages in images with optimized pixel selection!**

