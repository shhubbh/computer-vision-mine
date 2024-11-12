import cv2
import numpy as np
from skimage import measure, segmentation
from skimage.segmentation import active_contour
import matplotlib.pyplot as plt
from osgeo import gdal

# Enable GDAL exceptions to avoid warnings
gdal.UseExceptions()

# Step 1: Load the high-resolution orthoimage and DSM, preprocess, and downscale for faster processing
def preprocess_images(ortho_path, dsm_path, scale_factor=0.3):
    # Load orthoimage and resize
    ortho_image = cv2.imread(ortho_path, cv2.IMREAD_COLOR)
    ortho_image_resized = cv2.resize(ortho_image, (0, 0), fx=scale_factor, fy=scale_factor)
    gray_image = cv2.cvtColor(ortho_image_resized, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.equalizeHist(gray_image)
    blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)

    # Load DSM and resize
    dsm = gdal.Open(dsm_path)
    dsm_data = dsm.ReadAsArray()
    dsm_data_resized = cv2.resize(dsm_data, (0, 0), fx=scale_factor, fy=scale_factor)

    return blurred_image, dsm_data_resized, ortho_image_resized

# Step 2: Generate a mask from the DSM to identify areas of interest
def create_dsm_mask(dsm_data, elevation_threshold):
    # Ensure DSM is of type uint8 for compatibility
    dsm_data = np.clip(dsm_data, 0, 255)  # Clip DSM values to 0-255 range
    dsm_data = dsm_data.astype(np.uint8)
    # Create a binary mask based on elevation threshold
    _, dsm_mask = cv2.threshold(dsm_data, elevation_threshold, 255, cv2.THRESH_BINARY)
    return dsm_mask

# Step 3: Apply Canny Edge Detection and combine with DSM mask
def apply_canny_with_mask(image, mask):
    edges = cv2.Canny(image, 30, 120)
    
    # Ensure mask has the same dimensions as edges and is of type uint8
    if mask.shape != edges.shape:
        mask = cv2.resize(mask, (edges.shape[1], edges.shape[0]))
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
    return masked_edges

# Step 4: Apply Watershed Segmentation using DSM mask
def apply_watershed_segmentation(image, edges, mask):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Resize and ensure compatibility between binary and mask
    if mask.shape != binary.shape:
        mask = cv2.resize(mask, (binary.shape[1], binary.shape[0]))
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    binary = cv2.bitwise_and(binary, binary, mask=mask)  # Apply DSM mask
    
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(binary, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(image_colored, markers)
    watershed_boundaries = np.zeros_like(markers, dtype=np.uint8)
    watershed_boundaries[markers == -1] = 255
    return watershed_boundaries

# Step 5: Refine with Active Contours
def apply_active_contours(gray_image, edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        init = np.squeeze(largest_contour)[::10]  # Downsample to speed up
        
        snake = active_contour(gray_image, init, alpha=0.015, beta=10, gamma=0.001, max_num_iter=250)
        return snake
    return None

# Step 6: Extract and Display Results
def extract_and_display_contours(ortho_image, edges, watershed_boundaries, snake_contour):
    plt.figure(figsize=(15, 10))
    plt.subplot(221), plt.imshow(cv2.cvtColor(ortho_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image (Resized)')
    
    plt.subplot(222), plt.imshow(edges, cmap='gray')
    plt.title('Canny Edges (with DSM Mask)')
    
    plt.subplot(223), plt.imshow(watershed_boundaries, cmap='gray')
    plt.title('Watershed Boundaries')
    
    plt.subplot(224), plt.imshow(cv2.cvtColor(ortho_image, cv2.COLOR_BGR2RGB))
    if snake_contour is not None:
        plt.plot(snake_contour[:, 1], snake_contour[:, 0], '-b', lw=2)
    plt.title('Contours with Active Contours')
    
    plt.show()

# Main function to apply all steps in the workflow
def main(ortho_path, dsm_path, elevation_threshold=30):
    # Step 1: Pre-process images and load DSM mask
    preprocessed_image, dsm_data, ortho_image_resized = preprocess_images(ortho_path, dsm_path)
    
    # Step 2: Create a DSM mask for areas with elevations above the threshold
    dsm_mask = create_dsm_mask(dsm_data, elevation_threshold)
    
    # Step 3: Apply Canny Edge Detection with DSM Mask
    edges = apply_canny_with_mask(preprocessed_image, dsm_mask)
    
    # Step 4: Apply Watershed Segmentation with DSM Mask
    watershed_boundaries = apply_watershed_segmentation(preprocessed_image, edges, dsm_mask)
    
    # Step 5: Refine with Active Contours (Snakes)
    snake_contour = apply_active_contours(preprocessed_image, edges)
    
    # Step 6: Extract Contours and Display Results
    extract_and_display_contours(ortho_image_resized, edges, watershed_boundaries, snake_contour)

# Run the main function
ortho_path = r'C:\Users\shubh\Documents\Analytics\Waste Dumps\waste_area_ortho.tif'
dsm_path = r'C:\Users\shubh\Documents\Analytics\Waste Dumps\waste_area_dsm.tif'
main(ortho_path, dsm_path)
