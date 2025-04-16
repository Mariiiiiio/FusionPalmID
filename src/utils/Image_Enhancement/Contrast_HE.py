import cv2
import numpy as np
import os
from skimage import exposure
import matplotlib.pyplot as plt
from pathlib import Path

def create_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

def traditional_he(img):
    """Apply traditional Histogram Equalization"""
    return cv2.equalizeHist(img)

def adaptive_he(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply Adaptive Histogram Equalization"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def bi_histogram_equalization(img):
    """Apply Bi-Histogram Equalization (BHE)
    Splits the histogram at mean intensity and equalizes both parts separately"""
    mean = np.mean(img)
    low = img[img < mean]
    high = img[img >= mean]
    
    if len(low) > 0:
        low_eq = (low - np.min(low)) * (mean - 1) / (np.max(low) - np.min(low) + 1e-8)
    else:
        low_eq = low
        
    if len(high) > 0:
        high_eq = mean + (high - mean) * (255 - mean) / (np.max(high) - mean + 1e-8)
    else:
        high_eq = high
    
    result = np.zeros_like(img, dtype=np.uint8)
    result[img < mean] = low_eq.astype(np.uint8)
    result[img >= mean] = high_eq.astype(np.uint8)
    
    return result

def dualistic_sub_image_he(img):
    """Apply Dualistic Sub-Image Histogram Equalization (DSIHE)
    Splits the histogram at median intensity and equalizes both parts separately"""
    median = np.median(img)
    low = img[img < median]
    high = img[img >= median]
    
    if len(low) > 0:
        low_eq = (low - np.min(low)) * (median - 1) / (np.max(low) - np.min(low) + 1e-8)
    else:
        low_eq = low
        
    if len(high) > 0:
        high_eq = median + (high - median) * (255 - median) / (np.max(high) - median + 1e-8)
    else:
        high_eq = high
    
    result = np.zeros_like(img, dtype=np.uint8)
    result[img < median] = low_eq.astype(np.uint8)
    result[img >= median] = high_eq.astype(np.uint8)
    
    return result

def gamma_correction(img, gamma=1.5):
    """Apply Gamma Correction"""
    # Normalize to 0-1 range
    normalized = img / 255.0
    # Apply gamma correction
    corrected = np.power(normalized, gamma)
    # Scale back to 0-255 range
    return (corrected * 255).astype(np.uint8)

def log_transform(img):
    """Apply Logarithmic Transform"""
    # Avoid log(0) by adding 1
    return np.uint8(255 * np.log1p(img / 255.0))

def enhance_image(input_path, output_dir):
    """Apply various contrast enhancement techniques to an image and save results"""
    # Read image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image {input_path}")
        return
    
    # Get filename without extension
    filename = Path(input_path).stem
    
    # Apply enhancement techniques
    he_img = traditional_he(img)
    ahe_img = adaptive_he(img)
    bhe_img = bi_histogram_equalization(img)
    dsihe_img = dualistic_sub_image_he(img)
    gamma_img = gamma_correction(img)
    log_img = log_transform(img)
    
    # Save enhanced images
    cv2.imwrite(os.path.join(output_dir, f"{filename}_original.png"), img)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_HE.png"), he_img)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_AHE.png"), ahe_img)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_BHE.png"), bhe_img)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_DSIHE.png"), dsihe_img)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_Gamma.png"), gamma_img)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_Log.png"), log_img)
    
    # Create comparison visualization
    plt.figure(figsize=(15, 10))
    
    # Set the spacing between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Create subplots with larger font size for titles
    plt.subplot(2, 4, 1), plt.imshow(img, cmap='gray'), plt.title('Original', fontsize=25)
    plt.subplot(2, 4, 2), plt.imshow(he_img, cmap='gray'), plt.title('HE', fontsize=25)
    plt.subplot(2, 4, 3), plt.imshow(ahe_img, cmap='gray'), plt.title('AHE', fontsize=25)
    plt.subplot(2, 4, 4), plt.imshow(bhe_img, cmap='gray'), plt.title('BHE', fontsize=25)
    plt.subplot(2, 4, 5), plt.imshow(dsihe_img, cmap='gray'), plt.title('DSIHE', fontsize=25)
    plt.subplot(2, 4, 6), plt.imshow(gamma_img, cmap='gray'), plt.title('Gamma', fontsize=25)
    plt.subplot(2, 4, 7), plt.imshow(log_img, cmap='gray'), plt.title('Log', fontsize=25)
    
    plt.tight_layout()
    os.makedirs('./output_img_comparison', exist_ok=True)
    plt.savefig(os.path.join('./output_img_comparison', f"{filename}_comparison.png"))
    plt.close()
    
    print(f"Processed: {input_path}")

def process_directory(input_dir, output_dir):
    """Process all images in the input directory"""
    create_output_dir(output_dir)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(input_dir).glob(f'*{ext}')))
        image_files.extend(list(Path(input_dir).glob(f'*{ext.upper()}')))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for img_path in image_files:
        enhance_image(str(img_path), output_dir)
    
    print(f"All images processed. Results saved to {output_dir}")

def calculate_metrics(original, enhanced):
    """Calculate image quality metrics"""
    # PSNR (Peak Signal-to-Noise Ratio)
    psnr = cv2.PSNR(original, enhanced)
    
    # SSIM (Structural Similarity Index)
    ssim = exposure.compare_ssim(original, enhanced)
    
    # CNR (Contrast-to-Noise Ratio) - simplified version
    foreground = np.mean(enhanced)
    background = np.std(enhanced)
    cnr = abs(foreground - background) / (background + 1e-8)
    
    return {
        'PSNR': psnr,
        'SSIM': ssim,
        'CNR': cnr
    }

if __name__ == "__main__":
    input_dir = "train_img"
    output_dir = "output_img"
    
    process_directory(input_dir, output_dir)
