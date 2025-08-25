from PIL import Image
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_image_dimensions.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"Image dimensions: {width}x{height}")
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
    except Exception as e:
        print(f"Error processing image: {e}")