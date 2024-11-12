import cv2
import numpy as np
import rasterio
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Step 1: Load and downsample images
def preprocess_images(ortho_path, dsm_path, scale_factor=0.15):
    with rasterio.open(ortho_path) as ortho:
        bands = [ortho.read(i+1) for i in range(3)]
        ortho_image = np.dstack(bands).astype(np.uint8)
    ortho_image_resized = cv2.resize(ortho_image, (0, 0), fx=scale_factor, fy=scale_factor)
    
    with rasterio.open(dsm_path) as dsm:
        dsm_data = dsm.read(1).astype(np.float32)
    dsm_data_resized = cv2.resize(dsm_data, (ortho_image_resized.shape[1], ortho_image_resized.shape[0]))
    
    return ortho_image_resized, dsm_data_resized

# Step 2: Create a feature set combining color and elevation information
def create_feature_set(ortho_patch, dsm_patch):
    color_features = ortho_patch.reshape((-1, 3))
    elevation_features = dsm_patch.reshape((-1, 1))
    features = np.hstack((color_features, elevation_features))
    return features

# Step 3: Apply DBSCAN to a patch of data
def apply_dbscan_to_patch(features, eps=10, min_samples=50):
    features_normalized = (features - features.mean(axis=0)) / features.std(axis=0)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(features_normalized)
    return labels

# Step 4: Sliding window approach for clustering
def sliding_window_clustering(ortho_image, dsm_data, window_size=200, step_size=100, eps=10, min_samples=50):
    height, width, _ = ortho_image.shape
    label_map = np.full((height, width), -1)  # Initialize with -1 (for noise)

    for y in range(0, height - window_size + 1, step_size):
        for x in range(0, width - window_size + 1, step_size):
            # Extract patch
            ortho_patch = ortho_image[y:y+window_size, x:x+window_size]
            dsm_patch = dsm_data[y:y+window_size, x:x+window_size]
            
            # Create features and apply DBSCAN to the patch
            features = create_feature_set(ortho_patch, dsm_patch)
            labels = apply_dbscan_to_patch(features, eps=eps, min_samples=min_samples)
            
            # Map patch labels back to the full image label map
            patch_label_map = labels.reshape((window_size, window_size))
            label_map[y:y+window_size, x:x+window_size] = np.where(
                label_map[y:y+window_size, x:x+window_size] == -1,
                patch_label_map,
                label_map[y:y+window_size, x:x+window_size]
            )
    
    return label_map

# Step 5: Extract contours from DBSCAN label map
def extract_contours_from_clusters(label_map):
    unique_labels = np.unique(label_map[label_map != -1])  # Exclude noise
    contours = []
    
    for label in unique_labels:
        dump_mask = (label_map == label).astype(np.uint8) * 255
        label_contours, _ = cv2.findContours(dump_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(label_contours)
    
    return contours, label_map

# Step 6: Visualize results
def visualize_results(ortho_image, dsm_data, contours, label_map):
    result_image = ortho_image.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes[0].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Detected Waste Dump Boundaries')
    
    dsm_normalized = cv2.normalize(dsm_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dsm_colored = cv2.applyColorMap(dsm_normalized, cv2.COLORMAP_JET)
    axes[1].imshow(cv2.cvtColor(dsm_colored, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Elevation Model')
    
    label_map_display = (label_map != -1).astype(np.uint8) * 255  # Display detected areas
    axes[2].imshow(label_map_display, cmap='gray')
    axes[2].set_title('DBSCAN Cluster Mask')
    
    plt.tight_layout()
    plt.show()

# Main function to run the analysis
def main(ortho_path, dsm_path, window_size=200, step_size=100, eps=10, min_samples=50):
    ortho_image, dsm_data = preprocess_images(ortho_path, dsm_path)
    label_map = sliding_window_clustering(ortho_image, dsm_data, window_size, step_size, eps, min_samples)
    contours, label_map = extract_contours_from_clusters(label_map)
    visualize_results(ortho_image, dsm_data, contours, label_map)

# Run the main function
ortho_path = r'C:\Users\shubh\Documents\Analytics\Waste Dumps\waste_area_ortho.tif'
dsm_path = r'C:\Users\shubh\Documents\Analytics\Waste Dumps\waste_area_dsm.tif'
main(ortho_path, dsm_path, window_size=200, step_size=100, eps=10, min_samples=50)
