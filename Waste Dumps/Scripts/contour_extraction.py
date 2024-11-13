import numpy as np
import matplotlib.pyplot as plt
import rasterio
import cv2
from skimage import morphology

def preprocess_dsm(dsm, lower_threshold=0.3, upper_threshold=0.8):
    """
    Preprocess DSM by normalizing and applying elevation-based thresholds to detect waste dumps.
    
    Parameters:
    - dsm: 2D numpy array representing the Digital Surface Model (DSM).
    - lower_threshold: Lower elevation threshold (scaled between 0 and 1).
    - upper_threshold: Upper elevation threshold (scaled between 0 and 1).
    
    Returns:
    - binary_dsm: Binary image of the DSM suitable for contour detection.
    """
    # Normalize DSM for thresholding
    dsm_normalized = (dsm - np.min(dsm)) / (np.max(dsm) - np.min(dsm))
    
    # Apply elevation thresholds
    binary_dsm = np.logical_and(dsm_normalized > lower_threshold, dsm_normalized < upper_threshold).astype(np.uint8) * 255
    return binary_dsm

def remove_small_objects(binary_image, min_size=500):
    """
    Remove small objects from a binary image using morphological operations.
    
    Parameters:
    - binary_image: Input binary image.
    - min_size: Minimum size of objects to retain.
    
    Returns:
    - cleaned_image: Binary image with small objects removed.
    """
    cleaned_image = morphology.remove_small_objects(binary_image.astype(bool), min_size=min_size).astype(np.uint8) * 255
    return cleaned_image

def extract_contours_opencv(dsm, lower_threshold=0.3, upper_threshold=0.8, contour_area_threshold=1000):
    """
    Extract closed contours from DSM using OpenCV's findContours function.
    
    Parameters:
    - dsm: 2D numpy array representing the Digital Surface Model (DSM).
    - lower_threshold: Lower threshold to filter out low elevations.
    - upper_threshold: Upper threshold to filter out high elevations (vegetation).
    - contour_area_threshold: Minimum area of contours to be included.
    
    Returns:
    - contours_list: A list of closed contours from the DSM.
    """
    print("Extracting contours from DSM using OpenCV with elevation filtering...")
    
    # Preprocess DSM to get a binary image within specified elevation range
    binary_dsm = preprocess_dsm(dsm, lower_threshold, upper_threshold)
    
    # Remove small objects to filter out noise and vegetation
    binary_dsm_cleaned = remove_small_objects(binary_dsm, min_size=contour_area_threshold)
    
    # Detect contours
    contours, _ = cv2.findContours(binary_dsm_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area
    contours_list = [contour for contour in contours if cv2.contourArea(contour) > contour_area_threshold]
    print(f"Extracted {len(contours_list)} contours after filtering by area.")
    
    return contours_list

def visualize_dsm_with_contours(dsm, contours, ortho=None):
    """Visualize DSM with extracted contours and optional orthophoto background."""
    plt.figure(figsize=(10, 8))
    
    # Use orthophoto as background if provided, otherwise display DSM
    if ortho is not None:
        ortho_resized = cv2.resize(ortho, (dsm.shape[1], dsm.shape[0]))
        plt.imshow(ortho_resized)
    else:
        plt.imshow(dsm, cmap='terrain')
    
    # Plot each contour individually
    for contour in contours:
        contour = contour.reshape(-1, 2)  # Reshape for plotting
        plt.plot(contour[:, 0], contour[:, 1], color='red', linewidth=1)
    
    plt.title("DSM with Filtered Contours for Waste Dumps")
    plt.colorbar()
    plt.show()

# Usage example
ortho_path = r"C:\Users\shubh\Documents\Analytics\Waste Dumps\Data\DDS_Ortho.tif"
dsm_path = r"C:\Users\shubh\Documents\Analytics\Waste Dumps\Data\waste_area_dsm.tif"

with rasterio.open(dsm_path) as src:
    dsm = src.read(1)

# Normalize DSM for visualization
dsm_normalized = (dsm - np.min(dsm)) / (np.max(dsm) - np.min(dsm))

# Load orthophoto if available
with rasterio.open(ortho_path) as src:
    ortho = src.read()
    ortho = np.transpose(ortho, (1, 2, 0))  # Convert to HWC format

# Extract closed contours with elevation filtering and noise removal
contours = extract_contours_opencv(dsm_normalized, lower_threshold=0.3, upper_threshold=0.8, contour_area_threshold=1000)

# Visualize DSM with filtered contours and orthophoto background
visualize_dsm_with_contours(dsm_normalized, contours, ortho)
