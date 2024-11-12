import cv2
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from scipy import ndimage
from numba import jit
import concurrent.futures

def preprocess_images(ortho_path, dsm_path, scale_factor=0.3):
    """Load and preprocess images with efficient memory handling"""
    # Load orthoimage using GDAL for better memory management
    ortho = gdal.Open(ortho_path)
    bands = [
        cv2.resize(ortho.GetRasterBand(i+1).ReadAsArray(), (0, 0), 
                  fx=scale_factor, fy=scale_factor) 
        for i in range(3)
    ]
    ortho_image_resized = np.dstack(bands)
    
    # Load and preprocess DSM
    dsm = gdal.Open(dsm_path)
    dsm_data = dsm.ReadAsArray()
    dsm_data_resized = cv2.resize(dsm_data, 
                                 (ortho_image_resized.shape[1], 
                                  ortho_image_resized.shape[0]))
    
    # Clean up
    ortho = None
    dsm = None
    
    return ortho_image_resized, dsm_data_resized

@jit(nopython=True)
def calculate_slope_metrics(gradient_x, gradient_y):
    """Calculate slope magnitude and direction with Numba optimization"""
    slope_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    slope_direction = np.arctan2(gradient_y, gradient_x)
    return slope_magnitude, slope_direction

def analyze_terrain(dsm_data, window_size=5):
    """Analyze terrain features focusing on irregular mounds"""
    # Calculate gradients using Sobel with larger kernel for better noise handling
    gradient_y = cv2.Sobel(dsm_data, cv2.CV_64F, 0, 1, ksize=window_size)
    gradient_x = cv2.Sobel(dsm_data, cv2.CV_64F, 1, 0, ksize=window_size)
    
    # Calculate slope metrics using optimized function
    slope_magnitude, slope_direction = calculate_slope_metrics(gradient_x, gradient_y)
    
    # Calculate slope variability (indicates irregular surfaces)
    slope_variance = ndimage.uniform_filter(
        (slope_magnitude - ndimage.uniform_filter(slope_magnitude, window_size))**2,
        window_size
    )
    
    return slope_magnitude, slope_direction, slope_variance

def detect_mound_regions(slope_magnitude, slope_variance, dsm_data):
    """Detect regions likely to be waste mounds based on slope characteristics"""
    # Normalize metrics
    slope_magnitude_norm = cv2.normalize(slope_magnitude, None, 0, 1, cv2.NORM_MINMAX)
    slope_variance_norm = cv2.normalize(slope_variance, None, 0, 1, cv2.NORM_MINMAX)
    
    # Calculate local height variation
    height_variation = ndimage.gaussian_filter(dsm_data, sigma=3)
    height_variation = np.abs(dsm_data - height_variation)
    height_variation_norm = cv2.normalize(height_variation, None, 0, 1, cv2.NORM_MINMAX)
    
    # Combine metrics with weights favoring steep, irregular slopes
    mound_probability = (0.5 * slope_magnitude_norm + 
                        0.3 * slope_variance_norm +
                        0.2 * height_variation_norm)
    
    # Threshold to create binary mask
    mound_mask = (mound_probability > np.percentile(mound_probability, 75)).astype(np.uint8) * 255
    
    return mound_mask, mound_probability

def refine_mound_boundaries(mound_mask, min_area_ratio=0.01):
    """Refine mound boundaries with minimal morphological operations"""
    # Simple noise removal
    kernel = np.ones((5,5), np.uint8)
    cleaned_mask = cv2.morphologyEx(mound_mask, cv2.MORPH_OPEN, kernel)
    
    # Find and filter contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)
    
    total_area = cleaned_mask.shape[0] * cleaned_mask.shape[1]
    min_area = total_area * min_area_ratio
    
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Smooth contour with Douglas-Peucker algorithm
            epsilon = 0.003 * cv2.arcLength(contour, True)
            smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
            filtered_contours.append(smoothed_contour)
    
    return filtered_contours

def extract_waste_boundaries(ortho_image, dsm_data):
    """Main function to extract waste dump boundaries"""
    # Analyze terrain features
    slope_magnitude, slope_direction, slope_variance = analyze_terrain(dsm_data)
    
    # Detect potential mound regions
    mound_mask, mound_probability = detect_mound_regions(
        slope_magnitude, slope_variance, dsm_data
    )
    
    # Refine and extract boundaries
    contours = refine_mound_boundaries(mound_mask)
    
    return contours, mound_probability, mound_mask

def visualize_results(ortho_image, dsm_data, contours, mound_probability, mound_mask):
    """Visualize results with improved color schemes"""
    # Create result image with contours
    result_image = ortho_image.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
    
    # Create DSM visualization
    dsm_normalized = cv2.normalize(dsm_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dsm_colored = cv2.applyColorMap(dsm_normalized, cv2.COLORMAP_JET)
    dsm_overlay = cv2.addWeighted(ortho_image, 0.7, dsm_colored, 0.3, 0)
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Original image with detected boundaries
    axes[0,0].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title('Detected Waste Mounds')
    
    # DSM overlay
    axes[0,1].imshow(cv2.cvtColor(dsm_overlay, cv2.COLOR_BGR2RGB))
    axes[0,1].set_title('Elevation Model')
    
    # Probability map
    probability_plot = axes[1,0].imshow(mound_probability, cmap='magma')
    axes[1,0].set_title('Mound Probability Map')
    plt.colorbar(probability_plot, ax=axes[1,0])
    
    # Final mask
    axes[1,1].imshow(mound_mask, cmap='gray')
    axes[1,1].set_title('Mound Mask')
    
    plt.tight_layout()
    plt.show()

def main(ortho_path, dsm_path):
    # Load and preprocess images
    ortho_image, dsm_data = preprocess_images(ortho_path, dsm_path)
    
    # Extract boundaries
    contours, mound_probability, mound_mask = extract_waste_boundaries(
        ortho_image, dsm_data
    )
    
    # Visualize results
    visualize_results(ortho_image, dsm_data, contours, 
                     mound_probability, mound_mask)
    
    return contours

if __name__ == "__main__":
    ortho_path = r'C:\Users\shubh\Documents\Analytics\Waste Dumps\waste_area_ortho.tif'
    dsm_path = r'C:\Users\shubh\Documents\Analytics\Waste Dumps\waste_area_dsm.tif'
    contours = main(ortho_path, dsm_path)