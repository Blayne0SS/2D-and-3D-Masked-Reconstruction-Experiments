# MRI and Masked Reconstruction Experiments

This folder contains four related scripts. `MRI_ANTS.py` can align MRI scans before they are used by the MRI reconstruction scripts. The other files contain two 3D MRI reconstruction experiments and one grayscale masked diffusion experiment.

## Files in this folder

| File | What it does |
| --- | --- |
| `MRI_ANTS.py` | Aligns NIfTI MRI scans to a selected template with ANTsPy. It skips output files that already exist. |
| `MRI_DF_MODEL.py` | Trains and tests a 3D masked MRI diffusion model. It uses a recurrent residual U-Net and saves reconstructed MRI volumes and evaluation results. |
| `mri_masked_autoencoder.py` | Trains a 3D masked MRI autoencoder. It has commands for training, testing, inference on one MRI scan, and running a synthetic smoke test. |
| `MB_Diffusion_test_extended_GREY_sig_V2.py` | Runs a grayscale masked diffusion experiment. Its data preparation function searches for JPG, JPEG and PNG images. |

These are separate scripts and do not all need to be run in a set order. If you use `MRI_ANTS.py` to prepare MRI data, run it before the reconstruction script that uses its output folder.

## Requirements

The scripts use Python and PyTorch. The two diffusion scripts check for a CUDA GPU and stop if one is not available. The autoencoder script can run its synthetic smoke test on the CPU; full training is best run on a suitable GPU.

Depending on which script you run, the Python packages include PyTorch, torchvision, NumPy, pandas, Matplotlib, Pillow, scikit-learn, tqdm, nibabel, ANTsPy, MONAI, torchmetrics, timm and SciPy. Use a PyTorch build that matches the CUDA version on the computer you are using. `MRI_ANTS.py` needs ANTsPy and tqdm.

## Set the data paths

Some default paths point to folders on a previous HPC system. Update the paths for your computer before running the scripts.

- In `MRI_ANTS.py`, set `input_dir`, `output_dir` and `template_path` in the main section. It searches for files ending in `.nii`; it does not currently include `.nii.gz` files.
- In `MRI_DF_MODEL.py`, check `Training_data_folder`, `TEMPLATE_PATH` and `ALIGNED_DIR` near the start of the main section.
- In `mri_masked_autoencoder.py`, the default folder is set by `DATA_DIR`. You can change it in the file or provide `--data-dir` when running a command.
- In `MB_Diffusion_test_extended_GREY_sig_V2.py`, check `Training_data_folder`. Its data loader currently searches for JPG, JPEG and PNG files, so it will not load NIfTI files without changes to the loader.

Make sure that `MRI_ANTS.py` writes aligned volumes to the folder used by the MRI script you plan to run. Existing files in its output folder are skipped, so remove an old output file if you need that scan to be aligned again.

## Run the scripts

### Align MRI scans with `MRI_ANTS.py`

After setting the input folder, output folder and template, run:

```bash
python MRI_ANTS.py
```

The script applies ANTs SyN registration to each `.nii` scan and saves the warped image in the output folder.

### Run `MRI_DF_MODEL.py`

After updating the data paths and checking the settings in the main section, run:

```bash
python MRI_DF_MODEL.py
```

This script runs its configured training and testing process directly. The main settings, including block size, mask size, number of epochs and diffusion steps, are set in the file.

### Run `mri_masked_autoencoder.py`

Run the synthetic checks without loading MRI data:

```bash
python "mri_masked_autoencoder.py" smoke-test
```

Train with a selected data folder and output folder:

```bash
python "mri_masked_autoencoder.py" train \
  --data-dir /path/to/aligned_mri \
  --output ./mri_autoencoder_run
```

The script also has `test` and `infer` commands. To see all options, run:

```bash
python "mri_masked_autoencoder.py" --help
```

Running this script without a command starts training with the default settings in `Config`.

### Run `MB_Diffusion_test_extended_GREY_sig_V2.py`

Update the data path and settings in the main section, then run:

```bash
python MB_Diffusion_test_extended_GREY_sig_V2.py
```

This script does not use command line options. It runs the training and testing workflow configured in the file.

## Outputs

The autoencoder writes its checkpoints, training history, configuration and evaluation results to the selected output folder. The other scripts save aligned MRI volumes, model checkpoints, reconstructed images or volumes, and metric plots to the output locations set in their code.

Before running a script, check its data paths and output locations so that the results are saved where you expect.
