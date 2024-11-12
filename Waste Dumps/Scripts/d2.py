import cv2
import numpy as np
import rasterio
from skimage import filters, measure, morphology
import matplotlib.pyplot as plt

# Step 1: Load the DSM using rasterio, preprocess, and downscale for faster processing
def preprocess_dsm(dsm_path, scale_factor=0.3):
    with rasterio.open(dsm_path) as dsm:
        dsm_data = dsm.read(1).astype(np.float32)
    # Resize the DSM data
    dsm_data_resized = cv2.resize(dsm_data, (0, 0), fx=scale_factor, fy=scale_factor)
    return dsm_data_resized

# Step 2: Calculate the slope from DSM data
def calculate_slope(dsm_data):
    # Calculate the gradients in x and y directions
    sobelx = filters.sobel_v(dsm_data)
    sobely = filters.sobel_h(dsm_data)
    # Calculate the gradient magnitude (slope)
    slope = np.sqrt(sobelx**2 + sobely**2)
    return slope

# Step 3: Threshold the slope to identify steep regions (likely waste dump edges)
def threshold_slope(slope, slope_threshold=0.2):
    # Apply a threshold to isolate steep slopes
    slope_binary = (slope > slope_threshold).astype(np.uint8) * 255
    return slope_binary

# Step 4: Apply morphological operations to break up connected edges
def refine_slope_edges(slope_binary):
    # Use morphological closing to close gaps, then erode to separate close boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_edges = cv2.morphologyEx(slope_binary, cv2.MORPH_CLOSE, kernel)
    separated_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_ERODE, kernel, iterations=2)
    return separated_edges

# Step 5: Label connected components to identify individual waste dumps
def label_components(separated_edges):
    # Label each connected component
    num_labels, labels = cv2.connectedComponents(separated_edges)
    return num_labels, labels

# Step 6: Extract and display individual contours for each component
def extract_and_display_contours(dsm_data, labels, num_labels, original_shape, scale_factor):
    # Resize DSM for display purposes
    dsm_resized = cv2.resize(dsm_data, (original_shape[1], original_shape[0]))
    
    plt.figure(figsize=(15, 10))
    plt.subplot(121), plt.imshow(dsm_resized, cmap='terrain')
    plt.title('DSM Data (Resized)')
    
    plt.subplot(122), plt.imshow(dsm_resized, cmap='terrain')
    
    for label in range(1, num_labels):  # Skip the background (label 0)
        # Create a binary mask for each component
        component_mask = (labels == label).astype(np.uint8) * 255
        # Find contours for this component
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Plot each contour separately
        for cnt in contours:
            plt.plot(cnt[:, 0, 0], cnt[:, 0, 1], '-b', lw=2)
    
    plt.title('Separate Waste Dump Boundaries')
    plt.show()

# Main function to apply all steps in the workflow
def main(dsm_path, slope_threshold=0.2, scale_factor=0.3):
    # Step 1: Pre-process the DSM
    dsm_data_resized = preprocess_dsm(dsm_path, scale_factor)
    
    # Step 2: Calculate slope from DSM
    slope = calculate_slope(dsm_data_resized)
    
    # Step 3: Threshold the slope to find potential waste dump edges
    slope_binary = threshold_slope(slope, slope_threshold)
    
    # Step 4: Refine slope edges to break up connected contours
    refined_edges = refine_slope_edges(slope_binary)
    
    # Step 5: Label connected components to separate individual waste dumps
    num_labels, labels = label_components(refined_edges)
    
    # Step 6: Extract and display separate contours for each waste dump
    original_shape = dsm_data_resized.shape  # Get the original shape for resizing
    extract_and_display_contours(dsm_data_resized, labels, num_labels, original_shape, scale_factor)

# Run the main function
dsm_path = r'C:\Users\shubh\Documents\Analytics\Waste Dumps\waste_area_dsm.tif'  # Replace with the actual DSM path
main(dsm_path, slope_threshold=0.2, scale_factor=0.3)
