import cv2
from matplotlib import pyplot as plt
import numpy as np

# Load the image
img = cv2.imread(r"C:\Users\shubh\Documents\Analytics\Waste Dumps\Data\DDS_Ortho.png")

# Convert the image to grayscale (since edge detection is typically applied on grayscale images)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Auto Canny parameters
sigma = 1.0
median = np.median(gray_img)

# Apply automatic Canny edge detection using the computed median
lower = int(max(0, (1.0 - sigma) * median))
upper = int(min(255, (1.0 + sigma) * median))
auto_canny = cv2.Canny(gray_img, lower, upper)

# Plot the image using plt.imshow (Canny output is already grayscale)
plt.imshow(auto_canny, cmap='gray')
plt.axis('off')  # Hide axis if preferred
plt.show()
