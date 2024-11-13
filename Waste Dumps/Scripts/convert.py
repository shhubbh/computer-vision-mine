from PIL import Image

# Load the .tif image
input_path = r"C:\Users\shubh\Documents\Analytics\Waste Dumps\Data\filled_vegetation.tif"
output_path = r"C:\Users\shubh\Documents\Analytics\Waste Dumps\Data\NI.png"

# Open the TIFF file
with Image.open(input_path) as img:
    # Convert to PNG and save with no compression
    img.save(output_path, format="PNG", compress_level=0)
