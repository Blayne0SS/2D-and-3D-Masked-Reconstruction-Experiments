# 2D and 3D MRI Masked Reconstruction Experiments

This folder contains four scripts used at different stages of MRI reconstruction work. One script aligns MRI volumes. The other three reconstruct missing parts of MRI images: one works on 2D MRI slices, one works on 3D MRI volumes using diffusion, and one works on 3D MRI volumes using a masked autoencoder.

The scripts are separate experiments. They do not all need to be run in a set order. The alignment script is a data preparation step for experiments that use aligned 3D MRI volumes.

## Files in this folder

| File | Purpose |
| --- | --- |
| `MRI_ANTS.py` | Aligns 3D MRI scans to a reference MRI template using ANTsPy. |
| `MB_Diffusion_test_extended_GREY_sig_V2.py` | Trains a 2D masked diffusion model on grayscale 2D slices from an MRI dataset. |
| `MRI_DF_MODEL.py` | Trains and tests a 3D masked diffusion model on MRI volumes. |
| `mri_masked_autoencoder.py` | Trains a 3D masked autoencoder on MRI volumes. It also supports testing and inference on one scan. |

## How the reconstruction task works

Each reconstruction model is given an image with a selected region hidden. The mask tells the model which parts of the image are available and which parts need to be reconstructed. During evaluation, the model's prediction is compared with the original values in the hidden region.

For the diffusion models, the hidden region is gradually filled by reversing a sequence of noise steps. The autoencoder predicts the missing region in one model pass. It does not use the diffusion process.

## What each script does

### `MRI_ANTS.py` — align MRI volumes

This script prepares MRI volumes for 3D experiments that need scans to share the same spatial grid. It reads a reference image as the fixed image, then reads each input scan as a moving image. ANTsPy applies SyN registration and saves the warped moving image in the output folder.

The script searches through the input folder and its subfolders for files ending in `.nii`. It skips a scan if a file with the same name is already in the output folder. It does not currently include `.nii.gz` files.

Set `input_dir`, `output_dir` and `template_path` near the start of the script before running it. Make sure `output_dir` is the folder used by the 3D model you plan to run. If you need to align a scan again, remove its existing output file first because the script skips files that are already there.

### `MB_Diffusion_test_extended_GREY_sig_V2.py` — 2D MRI slice diffusion

This experiment works with individual grayscale MRI slices saved as image files. The data loader opens each image, converts it to grayscale and resizes it to the configured image size. The data preparation code looks for `.jpg`, `.jpeg` and `.png` files, so the MRI slices need to be exported to one of those formats before this script can load them.

For each image, the script selects a location and uses the configured patch and block sizes to create a square missing region. In the mask, `1` means the original pixel is known and `0` means it is part of the missing region. The diffusion process adds noise to the missing region while keeping the known pixels unchanged.

The model is a 2D U-Net. It uses residual convolution blocks to process the image at several resolutions, an attention block near the centre of the network, and skip connections to bring image details back into the decoder. A timestep embedding tells the model how much noise is present at the current step. The binary mask is also given to the model so it can identify the region being reconstructed.

During training, the model predicts the noise added to the image. The loss is measured only inside the missing region. During reconstruction, the script starts with a noisy image and applies reverse diffusion steps until the missing region has been filled. The known pixels are copied back at each step.

The script reports masked-region PSNR and saves training plots and model weights. Its settings and data path are in the main section of the file. The function names still contain older X-ray/CheXpert wording, but this experiment was used with 2D MRI slices.

### `MRI_DF_MODEL.py` — 3D MRI diffusion

This experiment works with 3D MRI volumes. It selects a 3D block from a scan and creates a cuboid missing region inside that block. The settings near the start of the script control the block size, mask size, training length, batch size and diffusion steps.

During training, the script adds Gaussian noise to the block at a randomly selected diffusion timestep. The known voxels are kept at their original values, while the hidden voxels are noised. The model receives the noisy block, the clean known context, the binary mask, the timestep, and information about the mask's location and size in the full scan.

The denoising model is a 3D recurrent residual U-Net. It uses 3D convolutions, recurrent residual blocks, attention, and skip connections. The timestep, mask position and mask size are converted into conditioning information for the network. The network predicts a velocity value used to estimate both the clean block and the noise at that timestep.

Training loss is calculated over the missing voxels. It combines the velocity prediction loss with clean-image MSE and L1 terms. During testing, the script uses DDIM sampling to fill the hidden region over a smaller number of reverse steps. It then evaluates PSNR and SSIM inside the hidden region and saves reconstructed NIfTI volumes and plots.

This script runs the training and testing process directly from the main section. It has no command-line options, so update the data and template paths and check the settings in the file before starting it. It currently stops if CUDA is not available.

### `mri_masked_autoencoder.py` — 3D masked autoencoder

This experiment also works with 3D MRI volumes, but it does not add or remove diffusion noise. It samples a 3D block and a cuboid hole, then gives the model the block with the hole hidden, a binary keep mask, and information about the hole's location and size.

The model is a 3D encoder-decoder. The encoder reduces the spatial size of the block while building feature maps. The decoder upsamples those features and uses skip connections to recover spatial detail. The default `recurrent` architecture uses recurrent residual blocks and attention. The `simple` architecture uses simpler residual blocks. The final layer predicts image intensities directly.

The training loss is L1 error inside the missing region only. During inference, the script copies the known voxels directly from the input back into the result, so the model output is used only for the hole. It tracks a moving-average copy of the model weights and chooses the best checkpoint using validation results before measuring performance on held-out test subjects.

The script reports masked MAE, MSE, PSNR and 3D SSIM. It can save NIfTI examples, checkpoints, metric files and learning curves. Its command-line interface supports `smoke-test`, `train`, `test` and `infer`.

## Preparing the data

Some paths in the files point to an earlier HPC system. Update those paths for the computer you are using.

- `MRI_ANTS.py`: set `input_dir`, `output_dir` and `template_path`.
- `MRI_DF_MODEL.py`: check `Training_data_folder`, `TEMPLATE_PATH` and `ALIGNED_DIR`.
- `mri_masked_autoencoder.py`: change `DATA_DIR` or pass `--data-dir` when training.
- `MB_Diffusion_test_extended_GREY_sig_V2.py`: set `Training_data_folder` to the folder containing the exported 2D MRI slices.

The 3D scripts expect NIfTI MRI volumes. The ANTs script currently searches for `.nii` files only. The 2D diffusion script expects grayscale MRI slices saved as JPG, JPEG or PNG files.

## Running the scripts

### Align MRI volumes

After setting the paths in `MRI_ANTS.py`, run:

```bash
python MRI_ANTS.py
```

### Run the 3D diffusion model

After updating the paths and settings in `MRI_DF_MODEL.py`, run:

```bash
python MRI_DF_MODEL.py
```

### Run the 3D masked autoencoder

Run the synthetic checks without loading MRI data:

```bash
python mri_masked_autoencoder.py smoke-test
```

Train using an aligned MRI folder and a new output folder:

```bash
python mri_masked_autoencoder.py train \
  --data-dir /path/to/aligned_mri \
  --output ./mri_autoencoder_run
```

The script also has `test` and `infer` commands. For the full list of settings, run:

```bash
python mri_masked_autoencoder.py --help
```

Running the script without a command starts training with the default settings in `Config`.

### Run the 2D MRI slice diffusion model

After exporting the MRI slices to JPG, JPEG or PNG files and updating the data path in the script, run:

```bash
python MB_Diffusion_test_extended_GREY_sig_V2.py
```

This script uses the settings in its main section and does not have command-line options.

## Requirements

The scripts use Python and PyTorch. The two diffusion scripts check for a CUDA GPU and stop if one is not available. The autoencoder can run its synthetic smoke test on the CPU; full training is best run on a suitable GPU.

Depending on which script you run, the required packages include PyTorch, torchvision, NumPy, pandas, Matplotlib, Pillow, scikit-learn, tqdm, nibabel, ANTsPy, MONAI, torchmetrics, timm and SciPy. Use a PyTorch build that matches the CUDA version on your computer.

## Outputs

`MRI_ANTS.py` saves aligned volumes to its configured output folder. The autoencoder saves checkpoints, training history, configuration, metric summaries and selected NIfTI examples to its output folder. The diffusion scripts save model weights, reconstructed examples or volumes, and metric plots to the paths configured in their code.

Before running a script, check its data paths, mask settings and output paths so that the results are saved where you expect.
