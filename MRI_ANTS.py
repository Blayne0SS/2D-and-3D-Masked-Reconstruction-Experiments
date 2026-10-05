import os

import ants
from tqdm import tqdm

# =============================================
# Main execution
# =============================================
# This script aligns MRI volumes to a selected template using ANTsPy.
if __name__ == "__main__":  
    # Set the source folder, output folder and reference MRI before running.
    input_dir = "/mnt/hpccs01/home/n10514821/X_ray_models/Data/MRI_T1_New_dataset"
    output_dir = "/mnt/hpccs01/home/n10514821/X_ray_models/aligned_mri_cache_block"
    template_path = (
        "/mnt/hpccs01/home/n10514821/X_ray_models/Data/MRI_T1_New_dataset/"
        "IXI284-HH-2354-T1/IXI284-HH-2354-T1.nii"
    )

    os.makedirs(output_dir, exist_ok=True)

    # The template is the fixed image that all other scans are aligned to.
    fixed = ants.image_read(template_path)

    # Collect NIfTI files from the input folder and its subfolders.
    all_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".nii"):
                all_files.append(os.path.join(root, f))

    # Align each scan, save the warped image, and skip files already saved.
    for path in tqdm(all_files):
        filename = os.path.basename(path)
        out_path = os.path.join(output_dir, filename)

        if os.path.exists(out_path):
            continue

        try:
            moving = ants.image_read(path)

            transform = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform='SyN'
            )

            ants.image_write(transform['warpedmovout'], out_path)

        except Exception as e:
            print(f"Failed: {path} -> {e}")
               