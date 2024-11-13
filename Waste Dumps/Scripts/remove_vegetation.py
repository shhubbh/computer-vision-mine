import rasterio
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes

# Input and output file paths
input_tif = r"C:\Users\shubh\Documents\Analytics\Waste Dumps\Data\waste_area_ortho.tif"
dem_tif = r"C:\Users\shubh\Documents\Analytics\Waste Dumps\Data\waste_area_dsm.tif"
output_dem_tif = r"C:\Users\shubh\Documents\Analytics\Waste Dumps\Data\output_dem_excluded_vegetation.tif"

# Vegetation threshold for ExG (adjust as needed based on your data)
threshold = 10

# Step 1: Generate the vegetation mask from the orthophoto
with rasterio.open(input_tif) as src:
    # Read each RGB band (assuming band order is R=1, G=2, B=3)
    red = src.read(1).astype("float32")
    green = src.read(2).astype("float32")
    blue = src.read(3).astype("float32")
    
    # Calculate the Excess Green (ExG) index
    exg = 2 * green - red - blue

    # Threshold ExG to identify vegetation
    vegetation_mask = exg > threshold

    # Apply morphological closing to fill small gaps in the vegetation mask
    structure = np.ones((3, 3))  # 3x3 structuring element for morphological operation
    vegetation_mask = binary_closing(vegetation_mask, structure=structure)
    
    # Fill holes within the vegetation mask to make it more continuous
    vegetation_mask = binary_fill_holes(vegetation_mask).astype("uint8")

# Step 2: Resample the DEM to match the orthophoto resolution, apply the vegetation mask, and save
with rasterio.open(dem_tif) as dem_src:
    # Resample DEM to match the resolution of the vegetation mask (orthophoto)
    dem_resampled = dem_src.read(
        out_shape=(1, src.height, src.width),
        resampling=rasterio.enums.Resampling.bilinear
    )[0]

    # Set vegetation areas to NaN to exclude them from the DEM
    dem_excluded_vegetation = np.where(vegetation_mask, np.nan, dem_resampled).astype("float32")

    # Prepare metadata for the output DEM file
    output_meta = dem_src.meta.copy()
    output_meta.update({
        "dtype": "float32",            # Keep as float32 for DEM values
        "height": src.height,          # Set height to match orthophoto
        "width": src.width,            # Set width to match orthophoto
        "transform": src.transform,    # Update transform to match orthophoto
        "nodata": np.nan               # Set nodata value to NaN
    })

    # Write the modified DEM to a single output file
    with rasterio.open(output_dem_tif, "w", **output_meta) as dest:
        dest.write(dem_excluded_vegetation, 1)  # Write the modified DEM to the single band
