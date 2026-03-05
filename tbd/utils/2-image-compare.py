from PIL import Image, ImageChops
import argparse

def compare_and_output_difference(image1_path, image2_path, output_path):
    # 1. Open the two images and ensure they are in the same color mode (e.g., RGB)
    try:
        img1 = Image.open(image1_path).convert('L')
        img2 = Image.open(image2_path).convert('L')
    except FileNotFoundError as e:
        print(f"Error: Could not find image file. {e}")
        return

    # 2. Check if the images are the exact same size
    if img1.size != img2.size:
        raise ValueError(f"Images must be the same size! Image 1 is {img1.size} and Image 2 is {img2.size}.")

    # 3. Calculate the pixel-by-pixel difference
    # ImageChops.difference calculates the absolute value of the pixel difference 
    # (e.g., |pixel1 - pixel2|) across all color channels.
    diff_image = ImageChops.difference(img1, img2)

    # 4. Save the resulting difference image
    diff_image.save(output_path)
    print(f"Success! Difference image saved to: {output_path}")

# --- How to use the function ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument("--input1", type=str, required=True)
    parser.add_argument("--input2",  type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    img1_file = args.input1
    img2_file = args.input2
    if args.output is not None:
        output_file = args.output
    else:
        x = img1_file.split('/')
        y = img2_file.split('/')
        x[-1] = output_file =  y[-1].split('.')[0] + x[-1] 
        
        if not len(x) == 1:
            output_file  = ""
            for part in x:
                output_file += part + "/"
            output_file = "diff/" + output_file[0:len(output_file)-1] 

    
    print(output_file)
    breakpoint()
    # Replace these strings with the paths to your actual images
    
    compare_and_output_difference(img1_file, img2_file, output_file)
