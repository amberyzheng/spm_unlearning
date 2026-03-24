import os
import numpy as np
import argparse
import json
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_images_to_fid(fid_obj, directory, is_real):
    """Helper to walk through a directory and update the FID object."""
    print(f"Loading {'real' if is_real else 'fake'} images from: {directory}")
    found_files = False
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                found_files = True
                img = Image.open(os.path.join(root, file)).convert('RGB')
                # Convert to Tensor (C, H, W) and scale to uint8 range
                img_np = np.array(img)
                img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
                fid_obj.update(img_t, real=is_real)
    
    if not found_files:
        print(f"Warning: No images found in {directory}")

def compute_fid(real_path, fake_path, out_path):
    # Initialize FID (feature=2048 is standard for InceptionV3)
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Load images
    load_images_to_fid(fid, real_path, is_real=True)
    load_images_to_fid(fid, fake_path, is_real=False)

    # Compute score
    score = fid.compute().item()
    print(f"\nFinal FID Score: {score:.4f}")

    # Save to JSON
    output_file = os.path.join(out_path, 'fid_score.json')
    
    # Create directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({'fid_score': score, 'real_path': real_path, 'fake_path': fake_path}, f, indent=4)
    
    print(f"Results saved to: {output_file}")
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID score between two directories.")
    
    parser.add_argument("--real_path", type=str, required=True, 
                        help="Path to the folder containing real images.")
    parser.add_argument("--fake_path", type=str, required=True, 
                        help="Path to the folder containing generated/fake images.")
    parser.add_argument("--out_path", type=str, required=True, 
                        help="Directory where the fid_score.json will be saved.")

    args = parser.parse_args()

    compute_fid(args.real_path, args.fake_path, args.out_path)