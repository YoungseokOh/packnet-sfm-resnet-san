from PIL import Image
import numpy as np

mask_path = "/workspace/packnet-sfm/ncdb-cls/synced_data/binary_mask.png"

try:
    mask_img = Image.open(mask_path).convert('L') # Convert to grayscale
    mask_np = np.array(mask_img)

    print(f"Mask file: {mask_path}")
    print(f"Mask shape: {mask_np.shape}")
    print(f"Mask data type: {mask_np.dtype}")
    print(f"Mask unique values: {np.unique(mask_np)}")
    print(f"Mask min value: {mask_np.min()}")
    print(f"Mask max value: {mask_np.max()}")

    # Check if it's truly binary (0 or 255 for 'L' mode)
    if np.all(np.isin(mask_np, [0, 255])):
        print("Mask appears to be binary (0 or 255).")
    else:
        print("Mask contains values other than 0 or 255.")

except FileNotFoundError:
    print(f"Error: Mask file not found at {mask_path}")
except Exception as e:
    print(f"An error occurred: {e}")
