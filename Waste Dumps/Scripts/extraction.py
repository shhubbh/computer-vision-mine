import numpy as np
import cv2
import rasterio
from PIL import Image
import matplotlib.pyplot as plt

# Step 1: Convert .tif file to .png image
tif_path = r'C:\Users\shubh\Documents\Analytics\Waste Dumps\Data\sobel5x5.tif'
with rasterio.open(tif_path) as src:
    image = src.read(1)  # Read the first band

# Normalize image values to 0-255 for PNG format
image_normalized = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
image_normalized = image_normalized.astype(np.uint8)

# Save as .png
png_path = r'C:\Users\shubh\Documents\Analytics\Waste Dumps\Data\image.png'
Image.fromarray(image_normalized).save(png_path)
print(f"Image saved as {png_path}")

# Step 2: Load the .png image
image_png = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)

# Step 3: Contrast Enhancement with Histogram Equalization
equalized_image = cv2.equalizeHist(image_png)

# Step 4: Apply Gaussian Blur for Smoothing
blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

# Step 5: Apply Otsu's Thresholding
_, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 6: Morphological Closing to Fill Gaps
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Step 7: Edge Detection on the Smoothed Image
edges = cv2.Canny(closed_image, 50, 150)

# Step 8: Contour Detection
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 9: Filter Contours by Area, Perimeter, and Aspect Ratio
max_area = max(cv2.contourArea(cnt) for cnt in contours)
max_perimeter = max(cv2.arcLength(cnt, closed=True) for cnt in contours)

upper_area_threshold = max_area * 0.85
upper_perimeter_threshold = max_perimeter * 0.85
min_area = 1500
min_perimeter = 450
aspect_ratio_threshold = 2  # Threshold for removing elongated contours

filtered_contours = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, closed=True)

    # Bounding rectangle to compute aspect ratio
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = max(w / h, h / w)  # Calculate the aspect ratio

    # Apply filtering based on area, perimeter, and aspect ratio
    if min_area < area < upper_area_threshold and min_perimeter < perimeter < upper_perimeter_threshold and aspect_ratio < aspect_ratio_threshold:
        filtered_contours.append(cnt)

# Step 10: Draw the Filtered Contours Directly (No Further Simplification)
output_image = np.zeros_like(image_png, dtype=np.uint8)
cv2.drawContours(output_image, filtered_contours, -1, 255, thickness=2)  # Increase thickness for visibility

# Step 11: Brightness Adjustment for Final Image
brightened_image = cv2.convertScaleAbs(output_image, alpha=3, beta=100)  # Further increase brightness

# Display the Result
plt.figure(figsize=(10, 10))
plt.imshow(brightened_image, cmap='gray')
plt.title("Enhanced Boundaries with Aspect Ratio Filtering for Thin Contours")
plt.axis('off')
plt.show()
