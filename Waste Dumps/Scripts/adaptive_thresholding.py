import numpy as np
import cv2
from scipy import ndimage
from skimage import filters, morphology, segmentation, measure
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from joblib import Parallel, delayed

def print_status(message):
    """Print status message with timestamp"""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

def load_and_preprocess(ortho_path, dsm_path):
    """Load and preprocess both orthophoto and DSM"""
    print_status("Loading input data...")
    
    # Read orthophoto
    with rasterio.open(ortho_path) as src:
        ortho = src.read()
        ortho = np.transpose(ortho, (1, 2, 0))  # Convert to HWC format
        print_status(f"Ortho shape: {ortho.shape}")
    
    # Read DSM
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1)  # Read first band
        print_status(f"DSM shape: {dsm.shape}")
        
    # Ensure DSM has same dimensions as ortho
    if dsm.shape != ortho.shape[:2]:
        print_status("Resizing DSM to match ortho dimensions...")
        dsm = cv2.resize(dsm, (ortho.shape[1], ortho.shape[0]))
    
    # Normalize DSM
    dsm_normalized = (dsm - np.min(dsm)) / (np.max(dsm) - np.min(dsm))
    
    print_status("Data loading complete")
    return ortho, dsm_normalized

def create_feature_stack(ortho, dsm):
    """Create a stack of features from both ortho and DSM"""
    print_status("Creating feature stack...")
    
    # Convert ortho to LAB color space
    lab = cv2.cvtColor(ortho, cv2.COLOR_RGB2LAB)
    print_status("Converted to LAB color space")
    
    # Calculate texture features from ortho with reduced kernel size for faster processing
    texture_kernel_size = 3
    texture = cv2.GaussianBlur(lab[:,:,0], (texture_kernel_size, texture_kernel_size), 0)
    print_status("Calculated texture features")
    
    # Downsample DSM for faster slope and height calculations
    dsm_downsampled = cv2.resize(dsm, (dsm.shape[1] // 2, dsm.shape[0] // 2))
    slope = filters.sobel(dsm_downsampled)
    slope = cv2.resize(slope, (dsm.shape[1], dsm.shape[0]))  # Resize back to original
    
    # Calculate local height variations on downsampled DSM
    height_var = ndimage.generic_filter(dsm_downsampled, np.var, size=3)
    height_var = cv2.resize(height_var, (dsm.shape[1], dsm.shape[0]))  # Resize back to original
    print_status("Calculated slope and height variations")
    
    # Ensure all features have the same dimensions
    features = [
        lab,
        np.expand_dims(texture, axis=2),
        np.expand_dims(dsm, axis=2),
        np.expand_dims(slope, axis=2),
        np.expand_dims(height_var, axis=2)
    ]
    
    # Stack all features
    feature_stack = np.concatenate(features, axis=2)
    print_status(f"Feature stack shape: {feature_stack.shape}")
    
    return feature_stack

def detect_dump_boundaries(feature_stack, dsm, min_area=1000, min_edge_strength=0.1):
    """Detect waste dump boundaries using multiple features"""
    print_status("Starting boundary detection...")
    
    # Create super pixels with higher segment count and lower compactness
    print_status("Generating superpixels...")
    n_segments = 300
    segments = segmentation.slic(feature_stack, n_segments=n_segments, compactness=5, sigma=1)
    
    # Parallel processing for mean elevation calculation
    print_status("Analyzing segment elevations...")
    segment_indices = np.arange(segments.max() + 1)
    segment_elevations = Parallel(n_jobs=-1)(delayed(ndimage.mean)(dsm, labels=segments, index=i) for i in segment_indices)
    
    # Edge detection with adaptive thresholding
    print_status("Computing elevation gradients and edges...")
    dsm_gradient = filters.sobel(dsm)
    edges = cv2.Canny((dsm_gradient * 255).astype(np.uint8), 20, 80)
    
    # Edge thresholding and morphology for cleanup
    print_status("Thresholding and cleaning up edges...")
    strong_edges = edges > (edges.max() * min_edge_strength)
    cleaned_edges = morphology.remove_small_objects(strong_edges, min_size=min_area)
    cleaned_edges = morphology.skeletonize(cleaned_edges)
    
    # Find and analyze contours
    print_status("Finding and analyzing contours...")
    contours = measure.find_contours(cleaned_edges, 0.5)
    filtered_contours = []
    
    for contour in tqdm(contours, desc="Processing contours"):
        # Convert to integer coordinates
        contour_int = np.round(contour).astype(np.int32)
        
        # Calculate area and filter
        area = cv2.contourArea(contour_int)
        if area >= min_area:
            filtered_contours.append(contour)
    
    print_status(f"Found {len(filtered_contours)} valid boundaries")
    return filtered_contours, cleaned_edges

def visualize_results(ortho, dsm, contours, edges):
    """Visualize the results with multiple views"""
    print_status("Generating visualizations...")
    
    plt.figure(figsize=(20, 10))
    
    # Original ortho
    plt.subplot(231)
    plt.imshow(ortho)
    plt.title('Original Orthophoto')
    plt.axis('off')
    
    # DSM
    plt.subplot(232)
    plt.imshow(dsm, cmap='terrain')
    plt.title('Digital Surface Model')
    plt.colorbar()
    plt.axis('off')
    
    # Detected edges
    plt.subplot(233)
    plt.imshow(edges, cmap='gray')
    plt.title('Detected Edges')
    plt.axis('off')
    
    # Overlay results on ortho
    plt.subplot(234)
    overlay = ortho.copy()
    for contour in contours:
        contour_int = np.round(contour).astype(np.int32)
        cv2.drawContours(overlay, [contour_int], -1, (255, 0, 0), 2)
    plt.imshow(overlay)
    plt.title('Detected Boundaries')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    print_status("Visualization complete")

def main(ortho_path, dsm_path):
    """Main function to run the waste dump boundary detection"""
    start_time = time.time()
    print_status("Starting waste dump boundary detection...")
    
    # Load and preprocess data
    ortho, dsm = load_and_preprocess(ortho_path, dsm_path)
    
    # Create feature stack
    feature_stack = create_feature_stack(ortho, dsm)
    
    # Detect boundaries
    contours, edges = detect_dump_boundaries(
        feature_stack, 
        dsm,
        min_area=1000,
        min_edge_strength=0.1
    )
    
    # Visualize results
    visualize_results(ortho, dsm, contours, edges)
    
    elapsed_time = time.time() - start_time
    print_status(f"Processing completed in {elapsed_time:.2f} seconds")
    
    return contours, edges

if __name__ == "__main__":
    ortho_path = r"C:\Users\shubh\Documents\Analytics\Waste Dumps\Data\DDS_Ortho.tif"
    dsm_path = r"C:\Users\shubh\Documents\Analytics\Waste Dumps\Data\waste_area_dsm.tif"
    contours, edges = main(ortho_path, dsm_path)
