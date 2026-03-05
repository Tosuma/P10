import numpy as np
from PIL import Image
import argparse

def create_binary_mask(image_path, output_path, threshold=128):
    # 1. Load the image and ensure it's in grayscale mode ('L')
    try:
        img = Image.open(image_path).convert('L')
    except FileNotFoundError as e:
        print(f"Error: Could not find image file. {e}")
        return None

    # 2. Convert the image into a NumPy array
    # This turns the image into a grid of numbers from 0 to 255
    pixel_data = np.array(img)

    # 3. Create the mask of 1s and 0s
    # This evaluates every pixel instantly: True (1) if > threshold, False (0) otherwise.
    binary_mask = (pixel_data > threshold).astype(np.uint8)

    # 4. Prepare the mask for saving
    # IMPORTANT: An image saved purely as 1s and 0s will look entirely pitch black 
    # to the human eye. To make the mask visible (white for 1, black for 0), 
    # we multiply the array by 255 before saving it as an image file.
    visible_mask = binary_mask * 255
    
    # 5. Convert back to a Pillow image and save
    output_image = Image.fromarray(visible_mask, mode='L')
    output_image.save(output_path)
    
    print(f"Success! Threshold mask saved to: {output_path}")
    
    # Return the actual array of 1s and 0s in case you need to use the raw data in your code
    return binary_mask

# --- How to use the function ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()


    input_file = args.input
    if args.output is not None:
        output_mask_file = args.output
    else:
        x = input_file.split('/')
        x[-1] = output_mask_file = "mask_" + x[-1]
        
        if not len(x) == 1:
            output_mask_file  = ""
            for part in x:
                output_mask_file += part + "/"
            output_mask_file = "masks/" + output_mask_file[0:len(output_mask_file)-1]

    
    print(output_mask_file)
    # You can adjust the threshold value (0-255) to be more or less sensitive
    mask_array = create_binary_mask(input_file, output_mask_file, threshold=50)
    
    # Just to prove it is an array of 1s and 0s
    if mask_array is not None:
        print(f"Unique values in the raw mask data: {np.unique(mask_array)}")
