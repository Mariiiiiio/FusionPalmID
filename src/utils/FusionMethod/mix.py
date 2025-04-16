import os
import cv2

class ImageBlender:
    def __init__(self):
        self.visible_images = []
        self.nir_images = []

    def blend_images(self, output_folder="output_blended"):
        print("\nCreating blended images...")

        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Sort images by timestamp in filename
        self.visible_images.sort()
        self.nir_images.sort()

        # Validate image pairs
        if len(self.visible_images) != len(self.nir_images):
            print("Error: Number of visible and NIR images don't match!")
            return

        # Track success/failure counts
        processed_count = 0
        error_count = 0

        for vis_path, nir_path in zip(self.visible_images, self.nir_images):
            try:
                # Extract numbers from both paths
                vis_number = ''.join(filter(str.isdigit, os.path.basename(vis_path)))
                nir_number = ''.join(filter(str.isdigit, os.path.basename(nir_path)))
                
                # Check if numbers match
                if vis_number != nir_number:
                    print(f"Warning: Number mismatch - VIS: {vis_number}, NIR: {nir_number}")
                    error_count += 1
                    continue
                
                # Read both images  
                visible_img = cv2.imread(vis_path)
                nir_img = cv2.imread(nir_path)
                
                # Ensure both images exist and have the same size
                if visible_img is None or nir_img is None:
                    print(f"Error: Could not read images for timestamp {vis_number}")
                    continue
                
                if visible_img.shape != nir_img.shape:
                    nir_img = cv2.resize(nir_img, (visible_img.shape[1], visible_img.shape[0]))
                
                # Create blended image (original size)
                # The -25 is a brightness adjustment (gamma) that darkens the final image by 25 units
                # alpha=0.8 for NIR, beta=0.2 for visible, gamma=-25 to reduce overall brightness
                # gamma: positive values increase brightness, negative values decrease brightness
                blended = cv2.addWeighted(nir_img, 0.8, visible_img, 0.2, -20)  # Darkens image by 25 units
                
                # Create filename with the same number as input
                output_filename = f"blended_{vis_number}.jpg"
                
                # Save blended image
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, blended)
                
                processed_count += 1

            except Exception as e:
                print(f"Error processing image pair: {vis_path}, {nir_path}")
                error_count += 1

        # Clear the stored paths
        self.visible_images = []
        self.nir_images = []

        print(f"Blended images created successfully! Processed: {processed_count}, Errors: {error_count}")

    def add_visible_image(self, path):
        """Add a visible light image path to the processing queue."""
        if os.path.exists(path):
            self.visible_images.append(path)
        else:
            print(f"Warning: Visible image path does not exist: {path}")

    def add_nir_image(self, path):
        """Add a NIR image path to the processing queue."""
        if os.path.exists(path):
            self.nir_images.append(path)
        else:
            print(f"Warning: NIR image path does not exist: {path}")

# Main execution
if __name__ == "__main__":
    blender = ImageBlender()
    
    # Get all files from input directories
    for filename in os.listdir("input_visible"):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            blender.add_visible_image(os.path.join("input_visible", filename))
            
    for filename in os.listdir("input_NIR"):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            blender.add_nir_image(os.path.join("input_NIR", filename))
    
    # Process the images
    blender.blend_images()