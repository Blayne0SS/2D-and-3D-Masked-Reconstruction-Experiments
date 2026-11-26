import os

import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import torchvision
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

import timm
import torchvision.models as models
import math
from torch.optim import Adam
from torchvision.utils import save_image
import copy

from sklearn.model_selection import GroupShuffleSplit
import sys

#import napari
import nibabel as nib
from tqdm import tqdm
import ants
import faulthandler


from torchmetrics.image import StructuralSimilarityIndexMeasure
from monai.losses import SSIMLoss

import csv
import re

from scipy.ndimage import gaussian_filter, distance_transform_edt


# =============================================
# Main execution
# This script trains and evaluates a 3D masked MRI diffusion model.
# =============================================
if __name__ == "__main__":  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ROCm uses the torch.cuda API. Benchmark AMP using MRI_AMP=auto|bf16|fp16|off.
    AMP_MODE = os.environ.get("MRI_AMP", "auto").lower()
    if AMP_MODE not in {"auto", "bf16", "fp16", "off"}:
        raise ValueError("MRI_AMP must be auto, bf16, fp16, or off")
    if AMP_MODE == "auto":
        AMP_MODE = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if AMP_MODE == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 not reported as supported; try MRI_AMP=fp16")
    AMP_ENABLED = AMP_MODE != "off"
    AMP_DTYPE = torch.bfloat16 if AMP_MODE == "bf16" else torch.float16
    USE_BF16 = AMP_MODE == "bf16"
    faulthandler.enable()
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    if device.type == "cpu":
        sys.exit("GPU not available")
    torch.cuda.empty_cache()
    
    ssim_3d = SSIMLoss(
        spatial_dims=3,
        data_range=1.0,
        win_size=5
    ).to(device)
        
    #MRI BLOCK varible
    # A MRI block is a section which the model will be trained on.
    # It represents a 3D object in which the mask will be located/placed in

    BLOCK_SIZE = int(os.environ.get("MRI_BLOCK_SIZE", "48"))

    MRI_BLOCK_SHAPE = (
        BLOCK_SIZE,
        BLOCK_SIZE,
        BLOCK_SIZE
    )

    # Number of sampled patches per MRI per epoch
    PATCHES_PER_SUBJECT = int(
        os.environ.get("MRI_PATCHES_PER_SUBJECT", "1")
    )

    # Minimum distance between mask and patch boundary
    MIN_MASK_BORDER = int(
        os.environ.get("MRI_MIN_MASK_BORDER", "4")
    )

    # Enable interpolation conditioning
    USE_INTERP_PRIOR = (
        os.environ.get("MRI_USE_INTERP_PRIOR", "0") == "1"
    )

    # Independent architectural ablation switches. Both off reproduces
    # the original SPEED_TUNED model, including its recurrent computation.
    USE_MASK_GUIDED_SKIP = os.environ.get("MRI_USE_MASK_GUIDED_SKIP", "0") == "1"
    SKIP_GATE_STRENGTH = float(os.environ.get("MRI_SKIP_GATE_STRENGTH", "0.5"))
    if not (0.0 < SKIP_GATE_STRENGTH <= 1.0):
        raise ValueError("MRI_SKIP_GATE_STRENGTH must be in (0, 1]")

    # Optional separate experiment: inject the initial projection at each
    # recurrent iteration, matching the change made to MANUAL_REVIEWED.
    USE_RECURRENT_INPUT_INJECTION = (
        os.environ.get("MRI_RECURRENT_INPUT_INJECTION", "0") == "1"
    )

    # Keep architectural ablations reproducible when run from scratch.
    TRAIN_SEED = int(os.environ.get("MRI_SEED", "42"))
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    torch.cuda.manual_seed_all(TRAIN_SEED)

    # Gradient-consistency loss
    GRADIENT_LOSS_WEIGHT = float(
        os.environ.get("MRI_GRAD_LOSS_WEIGHT", "0")
    )

    # High-noise training
    HIGH_NOISE_FRACTION = float(
        os.environ.get("MRI_HIGH_NOISE_FRACTION", "0")
    )

    HIGH_NOISE_START = int(
        os.environ.get("MRI_HIGH_NOISE_START", "75")
    )


    MRI_MASK_MIN = (7,7,7)
    MRI_MASK_MAX = (15,15,15)

    if PATCHES_PER_SUBJECT < 1:
        raise ValueError("MRI_PATCHES_PER_SUBJECT must be >= 1")
    if MIN_MASK_BORDER < 0:
        raise ValueError("MRI_MIN_MASK_BORDER must be >= 0")
    if any(m + 2 * MIN_MASK_BORDER > b for m, b in zip(MRI_MASK_MAX, MRI_BLOCK_SHAPE)):
        raise ValueError("MRI mask and twice the requested border must fit inside the patch")
    if not (0.0 <= HIGH_NOISE_FRACTION <= 1.0):
        raise ValueError("MRI_HIGH_NOISE_FRACTION must be between 0 and 1")
    if GRADIENT_LOSS_WEIGHT < 0:
        raise ValueError("MRI_GRAD_LOSS_WEIGHT must be nonnegative")

    base_channel_number=28     
    
    
    number_of_epochs = int(
        os.environ.get("MRI_EPOCHS", "51")
    ) #number of epochs  


    
    lr_option         = 4e-5      #learning rate 

    CPU_num_workers = int(os.environ.get("MRI_NUM_WORKERS", "4"))
    if CPU_num_workers < 1:
        raise ValueError("MRI_NUM_WORKERS must be >= 1 (persistent_workers is enabled)")
    TRAIN_METRIC_EVERY = int(os.environ.get("MRI_TRAIN_METRIC_EVERY", "25"))
    if TRAIN_METRIC_EVERY < 1:
        raise ValueError("MRI_TRAIN_METRIC_EVERY must be at least 1")
    GPU_batch_size = int(
        os.environ.get("MRI_BATCH_SIZE", "4")
    )

    
    T = 100                     #Steps
    DDIM_STEPS = 3
    ETA = 0.0

    EMA_DECAY = 0.999

    #Denoising Diffusion Probabilistic Models (DDPMs) 
    #NEED TO TEST T at 50,100,200,500,1000
    #create a^2(t) vs t graph 
    #to show  effectivness
    #See fig 2 https://arxiv.org/pdf/2405.14802
    #Base off this should switch to a FAST DDPM(also know as DDIM) 


    PROJECT_DIR = Path(__file__).resolve().parent
    EXPERIMENT_NAME = os.environ.get("MRI_EXP_NAME", "speed_tuned_mask_guided")
    RUN_DIR = PROJECT_DIR / "runs" / EXPERIMENT_NAME
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Experiment output directory: {RUN_DIR}")

    # Location of Neurite-OASIS dataset
    Training_data_folder = "/home/blayne/Desktop/MRI/Neurite-OASIS"

    TEMPLATE_PATH = None

    ALIGNED_DIR = str(PROJECT_DIR / "aligned_mri_cache")

    if not os.path.isdir(Training_data_folder):
        raise FileNotFoundError(
            f"MRI dataset folder not found: {Training_data_folder}"
        )

    MATMUL_PRECISION = os.environ.get("MRI_MATMUL_PRECISION", "highest")
    if MATMUL_PRECISION not in {"highest", "high", "medium"}:
        raise ValueError("MRI_MATMUL_PRECISION must be highest, high, or medium")
    torch.set_float32_matmul_precision(MATMUL_PRECISION)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    
    
    
    print(f"Config: patch={MRI_BLOCK_SHAPE}, mask_border={MIN_MASK_BORDER}, "
          f"repeats={PATCHES_PER_SUBJECT}, prior={USE_INTERP_PRIOR}, "
          f"gradient_weight={GRADIENT_LOSS_WEIGHT}, batch={GPU_batch_size}")
    print(f"Speed settings: amp={AMP_MODE}, matmul={MATMUL_PRECISION}, "
          f"workers={CPU_num_workers}, metric_every={TRAIN_METRIC_EVERY}")
    print(f"Architectural settings: mask_guided_skip={USE_MASK_GUIDED_SKIP}, "
          f"gate_strength={SKIP_GATE_STRENGTH}, "
          f"recurrent_input_injection={USE_RECURRENT_INPUT_INJECTION}, "
          f"seed={TRAIN_SEED}")

    # Find MRI scans and divide them into training, validation and test groups.

    def MRI_DATA_SET_CONFIG(
        chexpert_train_folder,
        split_ratios={'train': 0.7, 'val': 0.20, 'test': 0.10},
        seed=42,

        ):
            np.random.seed(seed)

            # -----------------------------------
            # 2. Load images into memory (numpy)
            # -----------------------------------
            X = []
            y = []
            groups = []  
            for root, _, files in os.walk(chexpert_train_folder):
                for file_name in files:
                    if file_name != "aligned_norm.nii.gz":
                        continue

                    img_path = os.path.join(root, file_name)
                    X.append(img_path)
                    y.append(0)
                    groups.append(root)


            if len(X) == 0:
                raise RuntimeError(
                    f"No images were loaded from {chexpert_train_folder}"
                )
            if len(set(groups)) < 4:
                raise ValueError(
                    "At least 4 distinct subject folders are required."
                )
            print(f"Total MRI volumes found: {len(X)}")
            print(f"Unique subjects: {len(set(groups))}")
            print(f"Example MRI: {X[0]}")
            X = np.array(X)
            y = np.array(y)
            groups = np.array(groups)

                    # -----------------------------------
            # 4. Shuffle
            # -----------------------------------
            X, y, groups = shuffle(X, y, groups, random_state=seed)

            # -----------------------------------
            # 5. Train / val / test split
            # -----------------------------------
            train_ratio = split_ratios['train']
            val_ratio = split_ratios['val'] / (split_ratios['val'] + split_ratios['test'])

            gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)

            train_idx, rest_idx = next(gss.split(X, y, groups))

            X_train, X_rest = np.array(X)[train_idx], np.array(X)[rest_idx]
            y_train, y_rest = y[train_idx], y[rest_idx]
            groups_rest = groups[rest_idx]

            gss_val = GroupShuffleSplit(n_splits=1, train_size=val_ratio, random_state=seed)

            val_idx, test_idx = next(gss_val.split(X_rest, y_rest, groups_rest))

            X_val, X_test = X_rest[val_idx], X_rest[test_idx]
            y_val, y_test = y_rest[val_idx], y_rest[test_idx]


            print("\n========== MRI DATASET ==========")
            print(f"Training:   {len(X_train)} volumes")
            print(f"Validation: {len(X_val)} volumes")
            print(f"Testing:    {len(X_test)} volumes")
            print("=================================\n")

            return{
                    "Train":{ 'train':(X_train, y_train)},
                    "Val": {'val':(X_val, y_val)},
                    "Test": {'test':(X_test, y_test)}
            }
            
            
    # Align an MRI scan with the selected reference template.

    def align_to_template(input_path, output_path, template_path):
        """
        Align MRI to template using affine registration.
        Cached: will skip if output exists.
        """
        if os.path.exists(output_path):
            return output_path
        if ants is None:
            raise ImportError("Install antspyx to run ANTs registration, or use aligned volumes.")

        try:
            fixed = ants.image_read(template_path)
            moving = ants.image_read(input_path)

            transform = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform='Affine'   
            )
            

            aligned = transform['warpedmovout']
            ants.image_write(aligned, output_path)

        except Exception as e:
            print(f"⚠️ Alignment failed for {input_path}: {e}")
            return input_path  # fallback (important)

        return output_path
        
        
        
    # Load MRI scans for the model and apply the required preprocessing.

    class MRI_Dataset(Dataset):

        def __init__(
            self,
            paths,
            y,
            template_path,
            aligned_dir,
            transform=None,
            return_index=False,
            repeats=1
        ):

            self.paths = paths
            self.y = y
            self.transform = transform
            self.return_index = return_index
            self.repeats = repeats

            self.template_path = template_path
            self.aligned_dir = aligned_dir

        def __len__(self):

            return len(self.paths) * self.repeats

        def __getitem__(self, idx):

            subject_idx = idx % len(self.paths)

            path = self.paths[subject_idx]
            label = self.y[subject_idx]

            nii = nib.load(path)

            data = nii.get_fdata(dtype=np.float32)

            data = (
                (data - data.min())
                / (data.max() - data.min() + 1e-8)
            )

            image = torch.from_numpy(data)

            image = image.permute(2, 1, 0).contiguous()

            if self.transform is not None:
                image = self.transform(image)

            out = [image, label]

            if self.return_index:
                out.append(idx)

            return tuple(out)
        
    data = MRI_DATA_SET_CONFIG(
    chexpert_train_folder=Training_data_folder,
    )

    train_block = data["Train"]
    val_block   = data["Val"]
    test_block  = data["Test"]

    X_train, y_train = train_block["train"]
    X_val, y_val     = val_block["val"]
    X_test, y_test   = test_block["test"]


    print(len(X_train))
    print(len(X_val))
    print(len(X_test))

    # ============================================
    # Define transforms
    # ============================================

    # Scale a 3D MRI volume before it is passed to the model.

    def normalize_3d(x):
        return (x - 0.5) / 0.5

    base_transform = normalize_3d


    train_dataset = MRI_Dataset(
        paths=X_train,
        y=y_train,
        template_path=TEMPLATE_PATH,
        aligned_dir=ALIGNED_DIR,
        transform=base_transform,
        return_index=True,
        repeats=PATCHES_PER_SUBJECT

    )


    val_dataset = MRI_Dataset(
        paths=X_val,
        y=y_val,
        template_path=TEMPLATE_PATH,
        aligned_dir=ALIGNED_DIR,
        transform=base_transform,
        return_index=True
    )

    test_dataset = MRI_Dataset(
        paths=X_test,
        y=y_test,
        template_path=TEMPLATE_PATH,
        aligned_dir=ALIGNED_DIR,
        transform=base_transform,
        return_index=True
    )


    train_loader = DataLoader(
            train_dataset,
            batch_size=GPU_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=CPU_num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=GPU_batch_size,
        shuffle=False,
        #drop_last=True,
        num_workers=CPU_num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=GPU_batch_size,
        shuffle=False,
        #drop_last=True,
        num_workers=CPU_num_workers,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2
    )




    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Choose the size of the missing region for this sample.

    def random_mask_shape(
        min_shape,
        max_shape
    ):
        dz = random.randrange(
            min_shape[0],
            max_shape[0] + 1,
            2
        )

        dy = random.randrange(
            min_shape[1],
            max_shape[1] + 1,
            2
        )

        dx = random.randrange(
            min_shape[2],
            max_shape[2] + 1,
            2
        )

        return dz, dy, dx
    
    
    
    
    # Select a 3D block and create the mask used for reconstruction.

    def Masking_function(
        MRI_Model,
        MRI_MASK_SHAPE=(31, 31, 31),
        MRI_BLOCK_SHAPE=(64, 64, 64),
        device=None,
        min_border=MIN_MASK_BORDER
    ):
        device = device if device is not None else MRI_Model.device

        # ---------------------------------------
        # Ensure shape = [B,D,H,W]
        # ---------------------------------------
        if MRI_Model.dim() == 5:
            MRI_Model = MRI_Model.squeeze(1)

        B, D, H, W = MRI_Model.shape

        bd, bh, bw = MRI_BLOCK_SHAPE
        MD, MH, MW = MRI_MASK_SHAPE

        for patch_dim, mask_dim in zip(
            MRI_BLOCK_SHAPE,
            MRI_MASK_SHAPE
        ):
            if mask_dim + 2 * min_border > patch_dim:
                raise ValueError(
                    f"Mask {MRI_MASK_SHAPE} with border {min_border} "
                    f"cannot fit inside patch {MRI_BLOCK_SHAPE}"
                )

            
        if any(s > v for s, v in zip((bd, bh, bw), (D, H, W))):
            raise ValueError(f"MRI block {MRI_BLOCK_SHAPE} exceeds volume {(D, H, W)}")
        if any(m > b for m, b in zip((MD, MH, MW), (bd, bh, bw))):
            raise ValueError(f"Mask {MRI_MASK_SHAPE} exceeds block {MRI_BLOCK_SHAPE}")

        blocks = []
        masks = []
        masked_blocks = []

        block_locations = []
        mask_locations = []

        for b in range(B):

            volume = MRI_Model[b]

            # -----------------------------------
            # Brain region
            # -----------------------------------
            brain_mask = volume > -0.7

            # Compute gradients
            dz = torch.abs(volume[1:] - volume[:-1])
            dy = torch.abs(volume[:,1:] - volume[:,:-1])
            dx = torch.abs(volume[:,:,1:] - volume[:,:,:-1])

            # Pad back
            dz = F.pad(dz, (0,0,0,0,0,1))
            dy = F.pad(dy, (0,0,0,1,0,0))
            dx = F.pad(dx, (0,1,0,0,0,0))

            gradient_map = dz + dy + dx

            # Keep high-information regions
            importance_mask = gradient_map > gradient_map.mean()

            # Combine with brain mask
            sampling_mask = brain_mask & importance_mask

            valid_indices = torch.nonzero(sampling_mask)

            # fallback
            if len(valid_indices) == 0:
                valid_indices = torch.nonzero(brain_mask)

            if len(valid_indices) == 0:
                raise RuntimeError("No brain voxels")

            idx = valid_indices[
                random.randint(0, len(valid_indices)-1)
            ]

            cz, cy, cx = idx.tolist()

            # clamp
            cz = max(bd//2, min(cz, D - (bd - bd//2)))
            cy = max(bh//2, min(cy, H - (bh - bh//2)))
            cx = max(bw//2, min(cx, W - (bw - bw//2)))

            block = volume[
                cz-bd//2:cz+bd//2,
                cy-bh//2:cy+bh//2,
                cx-bw//2:cx+bw//2
            ]

            # -----------------------------------
            # mask location
            # -----------------------------------
            block_mask = block > -0.7

            # Edge-aware masking
            bdz = torch.abs(block[1:] - block[:-1])
            bdy = torch.abs(block[:,1:] - block[:,:-1])
            bdx = torch.abs(block[:,:,1:] - block[:,:,:-1])

            bdz = F.pad(bdz, (0,0,0,0,0,1))
            bdy = F.pad(bdy, (0,0,0,1,0,0))
            bdx = F.pad(bdx, (0,1,0,0,0,0))

            block_grad = bdz + bdy + bdx

            importance = block_grad > block_grad.mean()

            valid_m_indices = torch.nonzero(
                block_mask & importance
            )


            if len(valid_m_indices) == 0:
                valid_m_indices = torch.nonzero(
                    torch.ones_like(block)
                )

            idxm = valid_m_indices[
                random.randint(0, len(valid_m_indices)-1)
            ]

            cmz, cmy, cmx = idxm.tolist()

            z1 = max(
                min_border,
                min(cmz - MD // 2, bd - MD - min_border)
            )

            y1 = max(
                min_border,
                min(cmy - MH // 2, bh - MH - min_border)
            )

            x1 = max(
                min_border,
                min(cmx - MW // 2, bw - MW - min_border)
            )

            cmz = z1 + MD // 2
            cmy = y1 + MH // 2
            cmx = x1 + MW // 2

            mask = torch.ones_like(block)

            mask[
                z1:z1 + MD,
                y1:y1 + MH,
                x1:x1 + MW
            ] = 0.0

            masked_block = block * mask

            blocks.append(block)
            masks.append(mask)
            masked_blocks.append(masked_block)

            block_locations.append((cz, cy, cx))
            mask_locations.append((cmz, cmy, cmx))

        # ---------------------------------------
        # Stack batch
        # ---------------------------------------
        blocks = torch.stack(blocks).unsqueeze(1)
        masks = torch.stack(masks).unsqueeze(1)
        masked_blocks = torch.stack(masked_blocks).unsqueeze(1)
        return (
            blocks,
            block_locations,
            masked_blocks,
            mask_locations,
            masks
        )
        
    # Create the cosine noise schedule used during diffusion.
    def make_interpolation_prior(block_cpu, mask_cpu, sigma=2.0):

        block_np = block_cpu.numpy()
        mask_np = mask_cpu.numpy()

        out = np.empty_like(
            block_np,
            dtype=np.float32
        )

        for i in range(block_np.shape[0]):

            observed = (
                mask_np[i, 0] > 0.5
            ).astype(np.float32)

            observed_value = (
                block_np[i, 0] * observed
            )

            numerator = gaussian_filter(
                observed_value,
                sigma=sigma
            )

            denominator = gaussian_filter(
                observed,
                sigma=sigma
            )

            nearest_idx = distance_transform_edt(
                observed < 0.5,
                return_distances=False,
                return_indices=True
            )

            nearest_value = block_np[i, 0][
                tuple(nearest_idx)
            ]

            interpolated = np.where(
                denominator > 1e-4,
                numerator / np.maximum(denominator, 1e-4),
                nearest_value
            ).clip(-1.0, 1.0)

            out[i, 0] = (
                observed_value
                + (1.0 - observed) * interpolated
            )

        return torch.from_numpy(out)

    def masked_gradient_loss_per_sample(
        pred_x0,
        ground_truth,
        known_mask
    ):

        missing = 1.0 - known_mask

        completed = (
            pred_x0 * missing
            + ground_truth * known_mask
        )

        loss_parts = []

        for axis in (-3, -2, -1):

            n = completed.shape[axis]

            a = completed.narrow(axis, 1, n - 1)
            b = completed.narrow(axis, 0, n - 1)

            ta = ground_truth.narrow(axis, 1, n - 1)
            tb = ground_truth.narrow(axis, 0, n - 1)

            ma = missing.narrow(axis, 1, n - 1)
            mb = missing.narrow(axis, 0, n - 1)

            include_pair = (
                ma + mb
            ).clamp(max=1.0)

            error = (
                ((a - b) - (ta - tb)).abs()
                * include_pair
            )

            loss_parts.append(
                error.flatten(1).sum(dim=1)
                / include_pair.flatten(1)
                .sum(dim=1)
                .clamp_min(1.0)
            )

        return torch.stack(
            loss_parts,
            dim=1
        ).mean(dim=1)
    def cosine_beta_schedule(timesteps, s=0.008):
        """
        Cosine diffusion schedule.

        Produces alpha_bar close to:
            1 at t=0
            0 at t=T

        This allows inference to begin from Gaussian noise.
        """

        steps = timesteps + 1

        x = torch.linspace(
            0,
            timesteps,
            steps,
            dtype=torch.float32
        )

        alphas_cumprod = torch.cos(
            ((x / timesteps + s) / (1 + s))
            * math.pi
            * 0.5
        ) ** 2

        # Normalize so alpha_bar starts at exactly 1
        alphas_cumprod = (
            alphas_cumprod
            / alphas_cumprod[0]
        )

        betas = (
            1
            - alphas_cumprod[1:]
            / alphas_cumprod[:-1]
        )

        return torch.clamp(
            betas,
            1e-5,
            0.999
        )
    
    
    # Create a sigmoid noise schedule for the diffusion steps.

    def sigmoid_beta_schedule(timesteps, s=0, e=3, tau=0.7):

        # time in [0,1]
        t = torch.linspace(0, 1, timesteps + 1)

        # Use tau to control curvature
        gamma = s + (e - s) * torch.sigmoid((t - 0.5) / tau)

        # Convert to log-SNR style alpha_bar
        alpha_bar = torch.sigmoid(-gamma)  # IMPORTANT: negative

        # Ensure it starts near 1 and ends near 0
        alpha_bar = alpha_bar / alpha_bar[0]  # normalize to 1 at t=0

        # Derive betas
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])

        return torch.clamp(betas, 1e-5, 0.999)

    def get_index_from_list(vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        device = t.device
        vals = vals.to(device) 
        batch_size = t.shape[0]
        out = vals.gather(0, t)  
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    # Add noise to the selected block while keeping the observed region available.

    def forward_diffusion_sample_section_with_mask(block, mask,block_location,mask_location, t, device=None):
        device = block.device if device is None else device
        block = block.to(device)
        known_mask= mask.to(device)
        """block,block_location,masked_block,mask_location,mask =Masking_function(MRI_Model,
                                                                MRI_MASK_SHAPE,
                                                                MRI_BLOCK_SHAPE) """
        
        #print(f"Block shape is {block.shape}")
        

        noise = torch.randn_like(block)

        sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, block.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            sqrt_one_minus_alphas_cumprod, t, block.shape
        )

        x_noisy = sqrt_alphas_cumprod_t * block + sqrt_one_minus_alphas_cumprod_t * noise

        x_noisy = x_noisy * (1 - known_mask) + block * known_mask

        return x_noisy, known_mask, noise,block_location,mask_location
    
    # Define beta schedule
    """betas = sigmoid_beta_schedule(
        T,
        s=-3,
        e=3,
        tau=0.7
    ).to(device)"""
    betas = cosine_beta_schedule(
        T,
        s=0.008
    ).to(device)

    # Pre-calculate diffusion terms
    alphas = (1. - betas).to(device)

    alphas_cumprod = torch.cumprod(alphas, dim=0)

    alphas_cumprod_prev = torch.cat(
        [
            torch.ones(1, device=device),
            alphas_cumprod[:-1]
        ],
        dim=0
    )

    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

    sqrt_one_minus_alphas_cumprod = torch.sqrt(
        1. - alphas_cumprod
    )

    posterior_variance = (
        betas
        * (1. - alphas_cumprod_prev)
        / (1. - alphas_cumprod)
    )

    sigmas = torch.zeros_like(betas)  # DDIM deterministic

    def show_tensor_slice(volume, z=None, title=None):
        """
        Shows a single slice from a 3D or 4D tensor
        """
        # Collapse to [D,H,W]
        if volume.dim() == 5:       # [B,C,D,H,W]
            volume = volume[0, 0]
        elif volume.dim() == 4:     # [B,D,H,W] or [C,D,H,W]
            volume = volume[0]
        elif volume.dim() != 3:
            raise ValueError(f"Unexpected shape: {volume.shape}")

        # convert [-1,1] → [0,1] if needed
        volume = (volume + 1) / 2.0

        if z is None:
            z = volume.shape[0] // 2

        slice_img = volume[z].detach().cpu().numpy()
        #print(f"image slice (z) is {z}")
        
        plt.imshow(slice_img, cmap="gray")
        if title is not None:
            plt.title(title)
        plt.axis("off")

    def crop_to_match(x, target):
        _, _, D, H, W = target.shape
        return x[:, :, :D, :H, :W]
    
    
    def get_safe_groups(channels, max_groups=8):
        groups = min(max_groups, channels)

        while channels % groups != 0 and groups > 1:
            groups -= 1

        return groups
        
    class UpSample3D(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()

            groups = get_safe_groups(out_ch)

            self.up = nn.Sequential(
                nn.ConvTranspose3d(
                    in_ch,
                    out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.GroupNorm(groups, out_ch),
                nn.SiLU()
            )

        def forward(self, x):
            return self.up(x)
    #___________________________________________________________175146
    #MODEL CONFIG 
    # -------------------------------------------------
    # RecurrentConv3d Block 
    # -------------------------------------------------
    class RecurrentConv3d(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1, t=2):
            super().__init__()

            self.t = t

            groups = get_safe_groups(out_ch)

            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1),
                nn.GroupNorm(groups, out_ch),
                nn.SiLU()
            )

            self.recurrent = nn.Sequential(
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(groups, out_ch),
                nn.SiLU()
            )

        def forward(self, x):

            initial = self.conv(x)
            h = initial

            for _ in range(self.t):
                if USE_RECURRENT_INPUT_INJECTION:
                    h = initial + self.recurrent(h)
                else:
                    # Preserve the SPEED_TUNED recurrence for attention-only tests.
                    h = h + self.recurrent(h)

            return h

    class RecurrentResidualBlock(nn.Module):
        def __init__(self, in_ch, out_ch, time_dim, downsample=False):
            super().__init__()

            stride = 2 if downsample else 1

            in_groups = get_safe_groups(in_ch)
            out_groups = get_safe_groups(out_ch)

            self.norm1 = nn.GroupNorm(in_groups, in_ch)
            self.act1 = nn.SiLU()

            self.recurrent_conv1 = RecurrentConv3d(
                in_ch,
                out_ch,
                stride=stride
            )

            self.time_mlp = nn.Linear(time_dim, out_ch)

            self.norm2 = nn.GroupNorm(out_groups, out_ch)
            self.act2 = nn.SiLU()

            self.recurrent_conv2 = RecurrentConv3d(
                out_ch,
                out_ch
            )

            self.dropout = nn.Dropout3d(0.1)

            if in_ch != out_ch or downsample:
                self.res_conv = nn.Conv3d(
                    in_ch,
                    out_ch,
                    1,
                    stride=stride
                )
            else:
                self.res_conv = nn.Identity()

        def forward(self, x, t_emb):

            h = self.recurrent_conv1(
                self.act1(self.norm1(x))
            )

            time_emb = self.time_mlp(t_emb)
            time_emb = time_emb[..., None, None, None]

            h = h + time_emb

            h = self.recurrent_conv2(
                self.act2(self.norm2(h))
            )

            h = self.dropout(h)

            return self.res_conv(x) + 0.3 * h

    class DenseSkipConnection(nn.Module):
        def __init__(self, in_ch, num_layers=3):
            super().__init__()

            groups = get_safe_groups(in_ch)

            self.convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv3d(in_ch, in_ch, 3, padding=1),
                    nn.GroupNorm(groups, in_ch),
                    nn.SiLU()
                )
                for _ in range(num_layers)
            ])

        def forward(self, x):

            residual = x

            for conv in self.convs:
                x = conv(x) + residual
                residual = x

            return x

    class MaskGuidedSkipGate3D(nn.Module):
        """Decoder- and mask-conditioned spatial gate for a dense 3D skip.

        known_mask: 1 for observed anatomy, 0 for missing voxels.
        The gate learns from both known and missing regions; it does not
        hard-mask encoder features. The zero-initialised output projection
        makes this operation exactly the identity at initialisation.
        """
        def __init__(self, skip_channels, decoder_channels, reduction=4, strength=0.5):
            super().__init__()
            hidden = max(8, min(skip_channels, decoder_channels) // reduction)
            self.skip_projection = nn.Conv3d(skip_channels, hidden, kernel_size=1, bias=False)
            self.decoder_projection = nn.Conv3d(decoder_channels, hidden, kernel_size=1, bias=False)
            self.mask_projection = nn.Conv3d(1, hidden, kernel_size=1, bias=False)
            self.psi = nn.Conv3d(hidden, 1, kernel_size=1)
            self.strength = float(strength)
            nn.init.zeros_(self.psi.weight)
            nn.init.zeros_(self.psi.bias)

        def forward(self, skip, decoder, known_mask):
            if skip.shape[-3:] != decoder.shape[-3:]:
                raise ValueError("Skip and decoder spatial dimensions must match")
            if known_mask is None:
                raise ValueError("Mask-guided attention requires the known-region mask")
            if known_mask.ndim == 4:
                known_mask = known_mask.unsqueeze(1)
            if known_mask.ndim != 5 or known_mask.shape[1] != 1:
                raise ValueError("known_mask must have shape [B,1,D,H,W]")

            missing = (1.0 - known_mask).to(device=skip.device, dtype=skip.dtype)
            missing = F.interpolate(
                missing,
                size=skip.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
            features = (
                self.skip_projection(skip)
                + self.decoder_projection(decoder)
                + self.mask_projection(missing)
            )
            logits = self.psi(F.silu(features))
            # Multiplier is initially 1 and stays in [1-strength, 1+strength].
            attention = 1.0 + self.strength * torch.tanh(logits)
            return skip * attention
    # -------------------------------------------------
    # Attention Block
    # -------------------------------------------------
    class AttentionBlock(nn.Module):
        def __init__(self, channels, num_heads=4):
            super().__init__()

            self.channels = channels
            self.num_heads = num_heads
            self.head_dim = channels // num_heads

            groups = get_safe_groups(channels)

            self.norm = nn.GroupNorm(groups, channels)

            self.qkv = nn.Conv3d(channels, channels * 3, 1)

            self.proj = nn.Conv3d(channels, channels, 1)

        def forward(self, x):

            B, C, D, H, W = x.shape

            h = self.norm(x)

            qkv = self.qkv(h)

            q, k, v = torch.chunk(qkv, 3, dim=1)

            N = D * H * W

            q = q.reshape(
                B,
                self.num_heads,
                self.head_dim,
                N
            ).transpose(-1, -2)

            k = k.reshape(
                B,
                self.num_heads,
                self.head_dim,
                N
            ).transpose(-1, -2)

            v = v.reshape(
                B,
                self.num_heads,
                self.head_dim,
                N
            ).transpose(-1, -2)

            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False
            )

            out = out.transpose(-1, -2).reshape(
                B,
                C,
                D,
                H,
                W
            )

            return x + 0.1 * self.proj(out)


    class SinusoidalPositionEmbeddings(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, t):
            device = t.device
            half_dim = self.dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
            emb = t[:, None] * emb[None, :]
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
            return emb

    class SpatialEmbedding(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(3, dim),
                nn.SiLU(),
                nn.Linear(dim, dim)
            )

        def forward(self, coords):
            # coords: (B, 3) -> (z, y, x)
            return self.mlp(coords)
        
    class MaskEmbedding(nn.Module):
        def __init__(self, dim):
            super().__init__()

            self.mlp = nn.Sequential(
                nn.Linear(6, dim),
                nn.SiLU(),
                nn.Linear(dim, dim)
            )

        def forward(self, mask_info):
            return self.mlp(mask_info)

    # Define the 3D recurrent residual U-Net used to predict the diffusion noise.

    class DiffusionR2UPlusPlus(nn.Module):
        def __init__(self, image_channels=1, mask_channels=1, base=base_channel_number,
                     use_interp_prior=USE_INTERP_PRIOR,
                     use_mask_guided_skip=USE_MASK_GUIDED_SKIP,
                     skip_gate_strength=SKIP_GATE_STRENGTH):
            super().__init__()

            self.use_interp_prior = use_interp_prior
            self.use_mask_guided_skip = use_mask_guided_skip

            in_ch = (
                image_channels * (2 + int(use_interp_prior))
                + mask_channels
            )
            time_dim = base * 4

            # Time embedding (unchanged)
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(base),
                nn.Linear(base, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
        
            self.spatial_mlp = SpatialEmbedding(time_dim)
            self.mask_mlp = MaskEmbedding(time_dim)
            
            # Initial convolution
            self.init = nn.Conv3d(in_ch, base, 3, padding=1)

            # Encoder (Down)
            self.down1 = RecurrentResidualBlock(base, base*2, time_dim, downsample=True)
            self.down2 = RecurrentResidualBlock(base*2, base*4, time_dim, downsample=True)
            self.down2_attn = AttentionBlock(base*4)
            self.down3 = RecurrentResidualBlock(base*4, base*8, time_dim, downsample=True)

            # Bottleneck
            self.mid1 = RecurrentResidualBlock(base*8, base*8, time_dim)
            self.attn = AttentionBlock(base*8)
            self.attn2 = AttentionBlock(base*8)
            
            self.mid2 = RecurrentResidualBlock(base*8, base*8, time_dim)

            # Dense Skip Connections
            self.skip1 = DenseSkipConnection(base*2)
            self.skip2 = DenseSkipConnection(base*4)
            if self.use_mask_guided_skip:
                self.gate_skip2 = MaskGuidedSkipGate3D(
                    skip_channels=base*4, decoder_channels=base*8,
                    strength=skip_gate_strength,
                )
                self.gate_skip1 = MaskGuidedSkipGate3D(
                    skip_channels=base*2, decoder_channels=base*4,
                    strength=skip_gate_strength,
                )

            # Upsampling layers
            self.up_sample1 = UpSample3D(base*8, base*8)
            self.up_sample2 = UpSample3D(base*4, base*4)
            self.up_sample3 = UpSample3D(base*2, base*2)

            # Decoder (Up) with RRCB
            self.up1 = RecurrentResidualBlock(base*8 + base*4, base*4, time_dim)
            self.up1_attn = AttentionBlock(base*4)
            self.up2 = RecurrentResidualBlock(base*4 + base*2, base*2, time_dim)
            self.up3 = RecurrentResidualBlock(base*2 + base, base, time_dim)

            # Final convolution
            self.final = nn.Conv3d(base, image_channels, 1)

        def forward(self, x, t, *, coords=None, mask=None, mask_info=None,context=None,interpolation_prior=None):
            # Save original observed/missing mask for gates before x becomes features.
            known_mask = mask
            channels = [
                x,
                context,
                mask
            ]

            if self.use_interp_prior:

                if interpolation_prior is None:
                    raise ValueError(
                        "Interpolation prior is required"
                    )

                channels.append(
                    interpolation_prior.to(
                        device=x.device,
                        dtype=x.dtype
                    )
                )

            x = torch.cat(channels, dim=1)

            t_emb = self.time_mlp(t)
            
            if coords is not None:
                coords = self.spatial_mlp(coords)
                t_emb = t_emb + coords
                
            if mask_info is not None:
                mask_emb = self.mask_mlp(mask_info)
                t_emb = t_emb + mask_emb
            # Encoder
            x1 = self.init(x)
            x2 = self.down1(x1, t_emb)
            x3 = self.down2(x2, t_emb)
            x3 = self.down2_attn(x3)
            x4 = self.down3(x3, t_emb)

            # Bottleneck
            x_mid = self.mid1(x4, t_emb)
            x_mid = self.attn(x_mid)
            x_mid = self.attn2(x_mid)
            x_mid = self.mid2(x_mid, t_emb)

            # Apply dense skip connections to encoder features
            skip1 = self.skip1(x2)
            skip2 = self.skip2(x3)
        

            # Decoder


            
            x = self.up_sample1(x_mid)
            x = crop_to_match(x, skip2)  # Crop to match skip2 (x3)
            if self.use_mask_guided_skip:
                skip2 = self.gate_skip2(skip2, x, known_mask)
            x = torch.cat([x, skip2], dim=1)
            x = self.up1(x, t_emb)
            x = self.up1_attn(x)
            

            x = self.up_sample2(x)
            x = crop_to_match(x, skip1)  # Crop to match skip1 (x2)
            if self.use_mask_guided_skip:
                skip1 = self.gate_skip1(skip1, x, known_mask)
            x = torch.cat([x, skip1], dim=1)
            x = self.up2(x, t_emb)
            
            
            x = self.up_sample3(x)
            x = crop_to_match(x, x1)     # Crop to match x1
            x = torch.cat([x, x1], dim=1)
            x = self.up3(x, t_emb)

            return self.final(x)




    
    raw_model = DiffusionR2UPlusPlus().to(device)

    raw_model = raw_model.to(
        memory_format=torch.channels_last_3d
    )

    # Keep compilation OFF while first testing the full model on ROCm.
    # It can increase setup time and memory, and complicate debugging.
    model = raw_model

    print(
        "Num params:",
        sum(p.numel() for p in raw_model.parameters())
    )

    # Optional initialisation from a baseline checkpoint. This loads *weights*
    # only; the optimiser and epoch number deliberately start fresh for a
    # paired attention-only fine-tuning experiment.
    init_checkpoint_path = os.environ.get("MRI_INIT_CHECKPOINT", "").strip()
    if init_checkpoint_path:
        weight_source = os.environ.get("MRI_INIT_WEIGHT_SOURCE", "ema").lower()
        if weight_source not in {"ema", "model"}:
            raise ValueError("MRI_INIT_WEIGHT_SOURCE must be 'ema' or 'model'")
        saved = torch.load(init_checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(saved, dict) and ("ema" in saved or "model" in saved):
            if weight_source not in saved:
                raise KeyError(f"Checkpoint has no {weight_source!r} weights")
            saved = saved[weight_source]
        result = raw_model.load_state_dict(saved, strict=False)
        allowed_missing = (
            lambda k: k.startswith("gate_skip1.") or k.startswith("gate_skip2.")
        ) if USE_MASK_GUIDED_SKIP else (lambda k: False)
        invalid_missing = [k for k in result.missing_keys if not allowed_missing(k)]
        if invalid_missing or result.unexpected_keys:
            raise RuntimeError(
                f"Incompatible checkpoint: missing={invalid_missing[:8]}, "
                f"unexpected={result.unexpected_keys[:8]}"
            )
        print(f"Initialised model weights from {init_checkpoint_path} "
              f"(source={weight_source}, new gate tensors={len(result.missing_keys)}). "
              "Optimiser starts fresh.")
    #print(model)


    def ensure_5d(x):
        # [D,H,W]
        if x.dim() == 3:
            return x.unsqueeze(0).unsqueeze(0)

        # [1,D,H,W]
        if x.dim() == 4:
            return x.unsqueeze(1)

        # already correct
        if x.dim() == 5:
            return x

        raise ValueError(f"Bad shape: {x.shape}")
        
        
        
        
    def masked_ssim_loss_3d(pred, target, mask):
        """
        SSIM loss calculated separately over each sample's
        masked-region bounding box.

        pred/target: [B, C, D, H, W]
        mask:        [B, D, H, W] or [B,1,D,H,W]
        """

        if mask.dim() == 4:
            mask = mask.unsqueeze(1)

        losses = []

        for b in range(pred.shape[0]):

            missing = (1 - mask[b, 0]) > 0.5
            coords = torch.nonzero(missing, as_tuple=False)

            if coords.numel() == 0:
                continue

            mins = coords.min(dim=0).values
            maxs = coords.max(dim=0).values + 1

            z1, y1, x1 = mins.tolist()
            z2, y2, x2 = maxs.tolist()

            pred_crop = pred[
                b:b+1, :, z1:z2, y1:y2, x1:x2
            ]

            target_crop = target[
                b:b+1, :, z1:z2, y1:y2, x1:x2
            ]

            losses.append(
                ssim_3d(pred_crop, target_crop)
            )

        if not losses:
            return pred.sum() * 0.0

        return torch.stack(losses).mean()
    
    
    
    
    # Calculate the training loss for the predicted diffusion output.

    def get_loss(model,MRI_Model, block,mask,block_location,mask_location,MRI_BLOCK_SHAPE,current_mask_shape, t, interpolation_prior=None):
        # MRI_Model is the full volume, intentionally kept on CPU to save VRAM.
        # block and mask are small 3D patches transferred to the GPU.
        block = block.to(device)
        mask = mask.to(device)
        t = t.to(device)

        x_noisy, mask, noise, block_location, mask_location = \
            forward_diffusion_sample_section_with_mask(
                block, mask, block_location, mask_location, t
            )
        if (1 - mask).sum() == 0:
            return None, None, None, None

        D, H, W = MRI_Model.shape[-3:]
        coords, mask_info = build_spatial_conditioning(
            block_location=block_location,
            mask_location=mask_location,
            current_mask_shape=current_mask_shape,
            MRI_BLOCK_SHAPE=MRI_BLOCK_SHAPE,
            full_shape=(D, H, W),
            device=device
        )

        # Original diffusion coefficients sampled at this batch's timesteps.
        alphas_cumprod_t = get_index_from_list(alphas_cumprod, t, block.shape)
        sqrt_alpha_t = torch.sqrt(alphas_cumprod_t)
        sqrt_one_minus_alpha_t = torch.sqrt((1.0 - alphas_cumprod_t).clamp_min(0))

        if mask.dim() == 4:
            mask_5d = mask.unsqueeze(1)
        else:
            mask_5d = mask

        # v-prediction target: v = sqrt(alpha_bar) * eps - sqrt(1-alpha_bar) * x0
        v_target = sqrt_alpha_t * noise - sqrt_one_minus_alpha_t * block

        # The model sees the noisy block and receives the uncorrupted context.
        context = block * mask_5d
        v_pred = model(
            x_noisy, t, coords=coords, mask=mask_5d,
            mask_info=mask_info, context=context,
            interpolation_prior=interpolation_prior
        )

        # Recover x0 from predicted velocity, consistent with sample_timestep().
        x0_pred = sqrt_alpha_t * x_noisy - sqrt_one_minus_alpha_t * v_pred

        # ------------------------------------------------------------
        # Masked region
        # mask = 1 -> known
        # mask = 0 -> reconstructed
        # ------------------------------------------------------------

        masked_region = 1.0 - mask_5d

        B = block.shape[0]

        # Number of missing voxels PER SAMPLE
        masked_voxels_per_sample = (
            masked_region
            .reshape(B, -1)
            .sum(dim=1)
            .clamp(min=1.0)
        )


        # ------------------------------------------------------------
        # v-prediction loss PER SAMPLE
        # ------------------------------------------------------------

        v_error = (
            (v_pred - v_target) ** 2
            * masked_region
        )

        v_loss_per_sample = (
            v_error
            .reshape(B, -1)
            .sum(dim=1)
            / masked_voxels_per_sample
        )


        # ------------------------------------------------------------
        # x0 MSE PER SAMPLE
        # ------------------------------------------------------------

        x0_error = (
            (x0_pred - block) ** 2
            * masked_region
        )

        x0_loss_per_sample = (
            x0_error
            .reshape(B, -1)
            .sum(dim=1)
            / masked_voxels_per_sample
        )


        # ------------------------------------------------------------
        # L1 loss PER SAMPLE
        # ------------------------------------------------------------

        l1_error = (
            torch.abs(x0_pred - block)
            * masked_region
        )

        l1_loss_per_sample = (
            l1_error
            .reshape(B, -1)
            .sum(dim=1)
            / masked_voxels_per_sample
        )


        # ------------------------------------------------------------
        # Min-SNR weighting PER SAMPLE
        # ------------------------------------------------------------

        snr = alphas_cumprod_t / (
            1.0 - alphas_cumprod_t + 1e-8
        )

        # [B,1,1,1,1] -> [B]
        snr = snr.reshape(B)

        gamma = 5.0

        t_weight = (
            torch.minimum(
                snr,
                torch.full_like(snr, gamma)
            )
            # Min-SNR weighting for v-prediction uses (SNR + 1), not SNR.
            / (snr + 1.0)
        )


        # ------------------------------------------------------------
        # Weight each individual image, THEN average batch
        # ------------------------------------------------------------

        weighted_v_loss = v_loss_per_sample * t_weight

        loss_per_sample = (
            weighted_v_loss
            + 0.25 * x0_loss_per_sample
            + 0.1 * l1_loss_per_sample
        )

        if GRADIENT_LOSS_WEIGHT > 0:

            gradient_per_sample = masked_gradient_loss_per_sample(
                x0_pred,
                block,
                mask_5d
            )

            loss_per_sample = (
                loss_per_sample
                + GRADIENT_LOSS_WEIGHT * gradient_per_sample
            )

        loss = loss_per_sample.mean()     
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss detected: {loss.item()}"
            )

        return loss, x_noisy, v_pred, noise
    
    # Run one reverse diffusion step during reconstruction.
    def select_train_timesteps(batch_size):

        times = torch.randint(
            0,
            T,
            (batch_size,),
            device=device
        )

        if HIGH_NOISE_FRACTION > 0:

            select = (
                torch.rand(batch_size, device=device)
                < HIGH_NOISE_FRACTION
            )

            high_t = torch.randint(
                HIGH_NOISE_START,
                T,
                (batch_size,),
                device=device
            )

            times = torch.where(
                select,
                high_t,
                times
            )

        return times
    @torch.no_grad()
    def sample_timestep(
        model,
        MRI_Model,
        x,
        original_block,
        mask,
        block_location,
        mask_location,
        current_mask_shape,
        t,
        t_prev,
        eta=ETA,
        interpolation_prior=None
    ):

        D, H, W = MRI_Model.shape[-3:]

        coords, mask_info = build_spatial_conditioning(
            block_location=block_location,
            mask_location=mask_location,
            current_mask_shape=current_mask_shape,
            MRI_BLOCK_SHAPE=MRI_BLOCK_SHAPE,
            full_shape=(D, H, W),
            device=device
        )
                
        

        # Current alpha
        alpha_t = get_index_from_list(
            alphas_cumprod,
            t,
            x.shape
        )

        # Alpha corresponding to the actual next DDIM timestep
        alpha_prev = get_index_from_list(
            alphas_cumprod,
            t_prev.clamp(min=0),
            x.shape
        )

        # For the final transition, treat alpha_bar(-1) as 1
        final_step = (t_prev < 0).view(
            -1, *([1] * (x.dim() - 1))
        )

        alpha_prev = torch.where(
            final_step,
            torch.ones_like(alpha_prev),
            alpha_prev
        )

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_t = torch.sqrt(1 - alpha_t)

        # -----------------------------
        # v-prediction
        # -----------------------------
        if mask.dim() == 4:
            mask_5d = mask.unsqueeze(1)
        else:
            mask_5d = mask

        context = original_block * mask_5d
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=AMP_ENABLED):
            v_pred = model(
                x,
                t,
                coords=coords,
                mask=mask_5d,
                mask_info=mask_info,
                context=context,
                interpolation_prior=interpolation_prior
            )

        # x0 reconstruction
        x0_pred = (
            sqrt_alpha_t * x
            - sqrt_one_minus_t * v_pred
        )

        # epsilon reconstruction
        eps_pred = (
            sqrt_one_minus_t * x
            + sqrt_alpha_t * v_pred
        )

        # DDIM sigma
        sigma_t = (
            eta
            * torch.sqrt((1 - alpha_prev)/(1 - alpha_t))
            * torch.sqrt(1 - alpha_t/alpha_prev)
        )

        noise = torch.randn_like(x) if eta != 0.0 else None

        pred_dir = torch.sqrt(
            1 - alpha_prev - sigma_t**2
        ) * eps_pred

        x_prev = torch.sqrt(alpha_prev) * x0_pred + pred_dir
        if noise is not None:
            x_prev = x_prev + sigma_t * noise

        # preserve known region
        x_prev = (
            x_prev * (1 - mask_5d)
            + original_block * mask_5d
        )

        return x_prev
    
    def get_ddim_schedule(steps):

        if steps == 3:
            return [T - 1, 14, 0]

        return (
            torch.linspace(
                T - 1,
                0,
                steps=min(steps, T)
            )
            .round()
            .long()
            .tolist()
        )
    # Set up the starting noise for the missing MRI region.

    def initialise_inpainting_noise(block, mask):
        """
        Initialise diffusion inpainting.

        Known region  -> real MRI
        Missing region -> pure Gaussian noise

        block: [B,1,D,H,W]
        mask:  [B,D,H,W] or [B,1,D,H,W]
        """

        if mask.dim() == 4:
            mask = mask.unsqueeze(1)

        mask = mask.to(
            device=block.device,
            dtype=block.dtype
        )

        noise = torch.randn_like(block)

        x = (
            block * mask
            + noise * (1 - mask)
        )

        return x
    
    
    # Prepare the position and size information used to condition the model.

    def build_spatial_conditioning(
        block_location,
        mask_location,
        current_mask_shape,
        MRI_BLOCK_SHAPE,
        full_shape,
        device
    ):
        D, H, W = full_shape

        coords = []
        mask_infos = []

        for b in range(len(block_location)):

            gz = (
                block_location[b][0]
                - MRI_BLOCK_SHAPE[0] // 2
                + mask_location[b][0]
            )

            gy = (
                block_location[b][1]
                - MRI_BLOCK_SHAPE[1] // 2
                + mask_location[b][1]
            )

            gx = (
                block_location[b][2]
                - MRI_BLOCK_SHAPE[2] // 2
                + mask_location[b][2]
            )

            # Global anatomical position [-1, 1]
            z_norm = 2.0 * gz / (D - 1) - 1.0
            y_norm = 2.0 * gy / (H - 1) - 1.0
            x_norm = 2.0 * gx / (W - 1) - 1.0

            coords.append([
                z_norm,
                y_norm,
                x_norm
            ])

            mask_infos.append([
                z_norm,
                y_norm,
                x_norm,

                current_mask_shape[0] / MRI_BLOCK_SHAPE[0],
                current_mask_shape[1] / MRI_BLOCK_SHAPE[1],
                current_mask_shape[2] / MRI_BLOCK_SHAPE[2]
            ])

        coords = torch.tensor(
            coords,
            device=device,
            dtype=torch.float32
        )

        mask_info = torch.tensor(
            mask_infos,
            device=device,
            dtype=torch.float32
        )

        return coords, mask_info
    
    # Generate and save an example reconstruction during sampling.

    @torch.no_grad()
    def sample_plot_image(model, device,MRI_model, block,mask,block_location,mask_location,current_mask_shape,num_images=8,ddim_steps=DDIM_STEPS, save_path=None):

        model.eval()

        # -----------------------------
        # Ensure batch dimension
        # -----------------------------
        if block.dim() == 3:
            block = block.unsqueeze(0).unsqueeze(0)
        elif block.dim() == 4:
            block = block.unsqueeze(0)

        block = block.to(device)

        T_local = T

        # -----------------------------
        # Initial noisy sample (t = T-1)
        # -----------------------------
        B = block.shape[0]

        t_full = torch.full(
            (B,),
            T_local - 1,
            device=device,
            dtype=torch.long
        )
                
        if mask.dim() == 4:
            mask = mask.unsqueeze(1)

        mask = mask.to(
            device=block.device,
            dtype=block.dtype
        )

        plot_prior = (
            make_interpolation_prior(block.float().cpu(), mask.float().cpu()).to(device)
            if USE_INTERP_PRIOR else None
        )
        x_noisy = initialise_inpainting_noise(
            block,
            mask
        )
        initial_noisy = x_noisy.clone()
        # -----------------------------
        # Reverse diffusion schedule
        # -----------------------------
        ddim_steps = min(ddim_steps, T_local)

        step_indices = torch.tensor(
            get_ddim_schedule(ddim_steps), device=device, dtype=torch.long
        )

        step_indices = torch.unique_consecutive(
            step_indices
        )

        
        
        # -----------------------------
        # Proper timestep selection
        # -----------------------------
        plot_indices = torch.linspace(
            0,
            len(step_indices) - 1,
            steps=num_images
        ).round().long()

        timesteps_set = set(
            step_indices[plot_indices].tolist()
        )

        collected = {}
        
        
        
        if step_indices[-1] != 0:
            step_indices = torch.cat([
                step_indices,
                torch.tensor(
                    [0],
                    device=step_indices.device,
                    dtype=step_indices.dtype
                )
            ])

        for step_idx in range(len(step_indices)):

            current_t = int(step_indices[step_idx].item())

            if step_idx + 1 < len(step_indices):
                previous_t = int(
                    step_indices[step_idx + 1].item()
                )
            else:
                previous_t = -1

            t = torch.full(
                (x_noisy.shape[0],),
                current_t,
                device=device,
                dtype=torch.long
            )

            t_prev = torch.full(
                (x_noisy.shape[0],),
                previous_t,
                device=device,
                dtype=torch.long
            )

            x_noisy = sample_timestep(
                model,
                MRI_model,
                x_noisy,
                block,
                mask,
                block_location,
                mask_location,
                current_mask_shape,
                t,
                t_prev,
                eta=ETA,
                interpolation_prior=plot_prior
            )

            i_int = current_t

            if i_int in timesteps_set:
                collected[i_int] = x_noisy.clone().cpu()
            
        # -----------------------------
        # Prepare visualization
        # -----------------------------
        
        images_to_plot = [block.cpu(), initial_noisy.cpu()]

        for t_val in sorted(collected.keys(), reverse=True):
            images_to_plot.append(collected[t_val])

        # -----------------------------
        # Plot
        # -----------------------------
        plt.figure(figsize=(20, 6))

        titles = ["Original", "Noisy"] + [
            f"t={t}" for t in sorted(collected.keys(), reverse=True)
        ]

        # IMPORTANT: ensure correct coordinates
        z = mask_location[0][0]
        #print(f"mask location in sample_plot_image is recorded as {z} ")

        for idx, (im, title) in enumerate(zip(images_to_plot, titles)):
            plt.subplot(1, len(images_to_plot), idx + 1)

            show_tensor_slice(im, z, title=title)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close()

    class EMA:
        def __init__(self, model, decay=0.999):
            self.decay = decay
            self.ema_model = copy.deepcopy(model)
            self.ema_model.eval()

            for param in self.ema_model.parameters():
                param.requires_grad_(False)

        @torch.no_grad()
        def update(self, model):
            for ema_param, param in zip(
                self.ema_model.parameters(),
                model.parameters()
            ):
                ema_param.lerp_(
                    param,
                    1.0 - self.decay
                )

            for ema_buffer, buffer in zip(
                self.ema_model.buffers(),
                model.buffers()
            ):
                ema_buffer.copy_(buffer)
            
    # Measure PSNR over the region that was hidden from the model.

    def compute_psnr_masked(original, reconstructed, mask):
        """Average masked-region PSNR per subject, in decibels."""
        true = (original.float() + 1.0) / 2.0
        pred = (reconstructed.float() + 1.0) / 2.0
        missing = 1.0 - mask.float()
        b = true.shape[0]
        mse = (((true - pred).square() * missing).reshape(b, -1).sum(dim=1) /
               missing.reshape(b, -1).sum(dim=1).clamp_min(1))
        return (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).mean().item()

    def compute_ssim_masked(original, reconstructed, mask):
        """Average 3D SSIM of each subject's own missing cuboid."""
        true = (original.float() + 1.0) / 2.0
        pred = (reconstructed.float() + 1.0) / 2.0
        if mask.dim() == 4:
            mask = mask.unsqueeze(1)
        missing = (1.0 - mask.float()) > 0.5
        values = []
        for i in range(true.shape[0]):
            xyz = missing[i, 0].nonzero(as_tuple=False)
            if xyz.numel() == 0:
                continue
            z0, y0, x0 = xyz.min(dim=0).values.tolist()
            z1, y1, x1 = (xyz.max(dim=0).values + 1).tolist()
            if min(z1 - z0, y1 - y0, x1 - x0) < 5:
                raise ValueError(f"SSIM window too large for mask crop {z1-z0,y1-y0,x1-x0}")
            values.append((1 - ssim_3d(
                pred[i:i+1, :, z0:z1, y0:y1, x0:x1],
                true[i:i+1, :, z0:z1, y0:y1, x0:x1]
            )).item())
        return sum(values) / len(values) if values else float('nan')

    ema = EMA(raw_model, decay=0.999)
    
    
    
    train_losses = []
    val_losses = []
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=lr_option, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=number_of_epochs,
    eta_min=1e-6
    )


    val_psnr_list = []
    train_psnr_list = []
    val_psnr_list_epoch = []

    train_ssim_list = []
    val_ssim_list = []
    
    
    # FP16 needs loss scaling; BF16 generally does not. Disabled scaler is a no-op.
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP_MODE == "fp16"))
    
    for epoch in range(number_of_epochs):
        # ---- TRAIN ----
        model.train()
        train_loss_epoch = 0
        train_psnr_epoch = 0
        train_ssim_epoch = 0
        train_metric_count = 0
        val_metric_count = 0
        train_loss_sum = torch.zeros((), device=device)
        train_batches_processed = 0
        torch.cuda.synchronize()  # done once per epoch, for a comparable epoch timer
        train_epoch_start = time.perf_counter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
        
        
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)
            if epoch == 0 and batch_idx == 0:
                torch.cuda.reset_peak_memory_stats()

            MRI_model = batch[0]

            B = MRI_model.shape[0]

            t = select_train_timesteps(B)
            
            with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP_ENABLED):
                current_mask_shape = random_mask_shape(
                    MRI_MASK_MIN,
                    MRI_MASK_MAX
                )

                block, block_location, masked_block, mask_block_location, mask = \
                    Masking_function(
                        MRI_model,
                        current_mask_shape,
                        MRI_BLOCK_SHAPE
                    )

                
                # Observed-only interpolation prior, computed while patches are on CPU.
                interpolation_prior = (
                    make_interpolation_prior(block, mask).to(device, non_blocking=True)
                    if USE_INTERP_PRIOR else None
                )
                block = block.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                if batch_idx % TRAIN_METRIC_EVERY == 0:

                    print("Full MRI device:", MRI_model.device)
                    print("Training block device:", block.device)
                    print("Mask device:", mask.device)

                    print(
                        "VRAM allocated:",
                        round(torch.cuda.memory_allocated() / 1024**3, 2),
                        "GiB"
                    )

                loss, x_noisy, v_pred, noise = get_loss(
                    model, MRI_model, block, mask, block_location,
                    mask_block_location, MRI_BLOCK_SHAPE, current_mask_shape, t,
                    interpolation_prior=interpolation_prior
                )
            
                if loss is None:
                    continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            ema.update(raw_model)
            
            if batch_idx % TRAIN_METRIC_EVERY == 0:
                with torch.no_grad():
                    sqrt_alphas_cumprod_t = get_index_from_list(
                        sqrt_alphas_cumprod, t, x_noisy.shape
                    )

                    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
                        sqrt_one_minus_alphas_cumprod, t, x_noisy.shape
                    )

                    x0_pred = (
                        sqrt_alphas_cumprod_t * x_noisy
                        - sqrt_one_minus_alphas_cumprod_t * v_pred
                    )

                    x0_pred = torch.clamp(x0_pred, -1, 1)

                    psnr = compute_psnr_masked(
                        (block),
                        (x0_pred),
                        (mask)
                    )

                    ssim = compute_ssim_masked(
                        (block),
                        (x0_pred),
                        (mask)
                    )
                train_psnr_epoch += psnr * B
                train_ssim_epoch += ssim * B
                train_metric_count += B

                if False and (epoch + 1) % 5 == 0:
                    sample_plot_image(
                        model=ema.ema_model, device=device, MRI_model=MRI_model,
                        block=block, mask=mask, block_location=block_location,
                        mask_location=mask_block_location,
                        current_mask_shape=current_mask_shape,
                        num_images=8, ddim_steps=DDIM_STEPS,
                        save_path=f"MRI_DDIM_epoch_{epoch+1}.png"
                    )

                if epoch == 0 and batch_idx == 0:
                    torch.cuda.synchronize()

                    print(
                        "Peak VRAM after first training step:",
                        f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GiB"
                    )
            train_loss_sum += loss.detach()
            train_batches_processed += 1
                    
        
        if train_metric_count > 0:
            train_psnr_epoch /= train_metric_count
            train_ssim_epoch /= train_metric_count
            train_psnr_list.append(train_psnr_epoch)
            train_ssim_list.append(train_ssim_epoch)


        train_loss_epoch = (
            train_loss_sum / max(train_batches_processed, 1)
        ).item()
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - train_epoch_start
        train_examples = train_batches_processed * GPU_batch_size
        print(f"Training epoch {epoch}: {train_seconds:.1f} s, "
              f"{train_examples / max(train_seconds, 1e-6):.2f} patches/s "
              f"({train_examples} patches)")
        train_losses.append(train_loss_epoch)
        pbar.set_postfix({
            "loss": f"{train_loss_epoch:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
        })
        # SNR and t_weight are local variables inside get_loss(); do not
        # reference them from the outer training loop.

        # ---- VALIDATION ----
        ema.ema_model.eval()
        val_loss_epoch = 0
        val_loss_sum = torch.zeros((), device=device)
        val_samples_processed = 0
        val_psnr_epoch = 0
        val_ssim_epoch = 0
        
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation", leave=False)
            for val_idx, batch in enumerate(val_pbar):

                MRI_model = batch[0]

                B = MRI_model.shape[0]
                t = torch.randint(0, T, (B,), device=device)
            
                with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP_ENABLED):

                    current_mask_shape = random_mask_shape(
                        MRI_MASK_MIN,
                        MRI_MASK_MAX
                    )

                    block, block_location, masked_block, mask_block_location, mask = \
                        Masking_function(
                            MRI_model,
                            current_mask_shape,
                            MRI_BLOCK_SHAPE
                        )
                    interpolation_prior = (
                        make_interpolation_prior(block, mask).to(device, non_blocking=True)
                        if USE_INTERP_PRIOR else None
                    )
                    block = block.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    loss, x_noisy, v_pred, noise = get_loss(
                        ema.ema_model, MRI_model, block, mask, block_location,
                        mask_block_location, MRI_BLOCK_SHAPE, current_mask_shape, t,
                        interpolation_prior=interpolation_prior
                    )

                if loss is None:
                    continue
                # One-step reconstruction metrics on all validation batches.
                    
                sqrt_alphas_cumprod_t = get_index_from_list(
                        sqrt_alphas_cumprod, t, x_noisy.shape
                    )

                sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
                    sqrt_one_minus_alphas_cumprod, t, x_noisy.shape
                )

                x0_pred = (
                    sqrt_alphas_cumprod_t * x_noisy
                    - sqrt_one_minus_alphas_cumprod_t * v_pred
                )

                x0_pred = torch.clamp(x0_pred, -1, 1)

                psnr = compute_psnr_masked(
                    (block),
                    (x0_pred),
                    (mask)
                )

                ssim = compute_ssim_masked(
                    (block),
                    (x0_pred),
                    (mask)
                )

                val_psnr_epoch += psnr * B
                val_ssim_epoch += ssim * B
                val_metric_count += B
                
                        
                val_loss_sum += loss.detach() * B
                val_samples_processed += B
            if val_metric_count > 0:
                val_psnr_epoch /= val_metric_count
                val_ssim_epoch /= val_metric_count
                val_psnr_list_epoch.append(val_psnr_epoch)
                val_ssim_list.append(val_ssim_epoch)


            val_loss_epoch = (val_loss_sum / max(val_samples_processed, 1)).item()
            val_losses.append(val_loss_epoch)


        
        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss_epoch:.6f} | "
            f"Val Loss: {val_loss_epoch:.6f} | "
            f"Train PSNR: {train_psnr_epoch:.2f} | "
            f"Val PSNR: {val_psnr_epoch:.2f} | "
            f"Train SSIM: {train_ssim_epoch:.4f} | "
            f"Val SSIM: {val_ssim_epoch:.4f}"
        )

        # Update learning-rate schedule (no early-stopping callback is configured).
        scheduler.step()
        checkpoint = {
            "epoch": epoch,
            "mask_guided_skip": USE_MASK_GUIDED_SKIP,
            "skip_gate_strength": SKIP_GATE_STRENGTH,
            "recurrent_input_injection": USE_RECURRENT_INPUT_INJECTION,
            "amp_mode": AMP_MODE,
            "model": raw_model.state_dict(),
            "ema": ema.ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_psnr": train_psnr_list,
            "val_psnr": val_psnr_list_epoch,
            "train_ssim": train_ssim_list,
            "val_ssim": val_ssim_list
        }

        tmp_ckpt = RUN_DIR / "mri_last.tmp"
        last_ckpt = RUN_DIR / "mri_last.pt"
        torch.save(checkpoint, tmp_ckpt)
        os.replace(tmp_ckpt, last_ckpt)

        print(f"Checkpoint saved: Epoch {epoch}")

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig(RUN_DIR / "loss_curve.png", dpi=300)
    plt.close()
    
    print("Saved loss curve to V2_GREY_loss_curve.png")
    
    plt.figure()
    plt.plot(train_psnr_list, label="Train PSNR")
    plt.plot(val_psnr_list_epoch, label="Validation PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.legend()
    plt.title("PSNR")
    plt.savefig(RUN_DIR / "PSNR_curve.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(train_ssim_list, label="Train SSIM")
    plt.plot(val_ssim_list, label="Validation SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.legend()
    plt.title("SSIM")
    plt.savefig(RUN_DIR / "SSIM_curve.png", dpi=300)
    plt.close()
    
    
    
    torch.save(raw_model.state_dict(), RUN_DIR / "raw_final.pt")
    torch.save(ema.ema_model.state_dict(), RUN_DIR / "ema_final.pt")





    # Reconstruct a masked 3D MRI block with the trained diffusion model.

    @torch.no_grad()
    def reconstruct_image(
        model,
        MRI_Model,
        block,
        mask,
        block_location,
        mask_location,
        current_mask_shape,
        device,
        ddim_steps=DDIM_STEPS
    ):
        """
        Reconstruct the masked region from pure Gaussian noise
        using DDIM sampling.

        Known region:
            preserved from the original MRI.

        Missing region:
            initialized using Gaussian noise.
        """

        model.eval()

        block = block.to(device)
        mask = mask.to(device)

        # ------------------------------------------
        # Ensure mask is [B,1,D,H,W]
        # ------------------------------------------
        if mask.dim() == 4:
            mask = mask.unsqueeze(1)

        mask = mask.to(
            device=device,
            dtype=block.dtype
        )

        # One interpolation prior for the entire DDIM trajectory.
        inference_prior = (
            make_interpolation_prior(block.float().cpu(), mask.float().cpu()).to(device)
            if USE_INTERP_PRIOR else None
        )

        # ------------------------------------------
        # Start missing region from PURE NOISE
        # ------------------------------------------
        img = initialise_inpainting_noise(
            block,
            mask
        )

        # ------------------------------------------
        # Construct DDIM timestep schedule
        # ------------------------------------------
        ddim_steps = min(ddim_steps, T)

        if ddim_steps == 3:
            step_indices = torch.tensor(
                [99, 14, 0],
                device=device,
                dtype=torch.long
            )
        else:
            step_indices = torch.tensor(
                get_ddim_schedule(ddim_steps),
                device=device,
                dtype=torch.long
            )

        # Remove possible duplicates caused by rounding
        step_indices = torch.unique_consecutive(
            step_indices
        )

        # ------------------------------------------
        # Reverse diffusion
        # ------------------------------------------
        for step_idx in range(len(step_indices)):

            current_t = int(
                step_indices[step_idx].item()
            )

            if step_idx + 1 < len(step_indices):
                previous_t = int(
                    step_indices[step_idx + 1].item()
                )
            else:
                previous_t = -1

            t = torch.full(
                (block.shape[0],),
                current_t,
                device=device,
                dtype=torch.long
            )

            t_prev = torch.full(
                (block.shape[0],),
                previous_t,
                device=device,
                dtype=torch.long
            )

            img = sample_timestep(
                model=model,
                MRI_Model=MRI_Model,
                x=img,
                original_block=block,
                mask=mask,
                block_location=block_location,
                mask_location=mask_location,
                current_mask_shape=current_mask_shape,
                t=t,
                t_prev=t_prev,
                eta=ETA,
                interpolation_prior=inference_prior
            )

        # ------------------------------------------
        # Preserve known region exactly
        # ------------------------------------------
        recon = (
            img * (1 - mask)
            + block * mask
        )

        return recon, mask
    
    
    

    def save_nifti(
        volume,
        save_path,
        convert_from_minus1_1=True
    ):
        if isinstance(volume, torch.Tensor):
            volume = volume.detach().cpu().numpy()

        if volume.ndim == 5:
            volume = volume[0, 0]

        elif volume.ndim == 4:
            volume = volume[0]

        if convert_from_minus1_1:
            volume = (volume + 1) / 2.0

        # Model tensor [D,H,W]=[Z,Y,X] --> NIfTI array [X,Y,Z].
        
        volume_xyz = np.transpose(volume, (2, 1, 0))
        nii = nib.Nifti1Image(
            volume_xyz.astype(np.float32),
            affine=np.eye(4)
        )

        nib.save(nii, save_path)


    # ==========================================================
    # TESTING / PSNR EVALUATION
    # ==========================================================

    max_images = 35

    model.eval()

    total_psnr = 0
    count = 0
    val_psnr_list = []
    
    total_ssim = 0
    val_ssim_list = []
    
    
    save_dir = str(RUN_DIR / "recon_outputs")
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats()    
        for batch in test_loader:

            MRI_model = batch[0]  
            B = MRI_model.shape[0]

            # --------------------------------------------------
            # Generate mask + masked block
            # --------------------------------------------------
            current_mask_shape = random_mask_shape(
                MRI_MASK_MIN,
                MRI_MASK_MAX
            )
            
            (block,block_location,masked_block,mask_block_location, mask) = Masking_function(
                MRI_model,
                current_mask_shape,
                MRI_BLOCK_SHAPE
            )

            block = block.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            # --------------------------------------------------
            # Reconstruct masked region
            # --------------------------------------------------
            x_recon, mask = reconstruct_image(
                model=ema.ema_model,
                MRI_Model=MRI_model,
                block=block,
                mask=mask,
                block_location=block_location,
                mask_location=mask_block_location,
                current_mask_shape=current_mask_shape,
                device=device,
                ddim_steps=DDIM_STEPS
            )
            missing = mask < 0.5

            out_of_range = (
                ((x_recon < -1.0) | (x_recon > 1.0))
                & missing
            )

            fraction = (
                out_of_range.sum().float()
                / missing.sum().clamp_min(1)
            )

            print(
                "Out-of-range reconstructed voxels:",
                f"{fraction.item() * 100:.2f}%"
            )

            x_recon_clipped = torch.where(
                mask > 0.5,
                block,
                x_recon.clamp(-1.0, 1.0)
            )

            print(
                "Original PSNR:",
                compute_psnr_masked(block, x_recon, mask)
            )

            print(
                "Clipped PSNR:",
                compute_psnr_masked(block, x_recon_clipped, mask)
            )
            # --------------------------------------------------
            # Measure PSNR ONLY on reconstructed region
            # --------------------------------------------------
            for i in range(B):

                if count >= max_images:
                    break

                psnr = compute_psnr_masked(
                    block[i:i+1],
                    x_recon[i:i+1],
                    mask[i:i+1]
                )

                ssim = compute_ssim_masked(
                    block[i:i+1],
                    x_recon[i:i+1],
                    mask[i:i+1]
                )

                print(
                    f"Sample {count} | "
                    f"PSNR: {psnr:.4f} dB | "
                    f"SSIM: {ssim:.4f}"
                )

                total_psnr += psnr
                val_psnr_list.append(psnr)
                total_ssim += ssim
                val_ssim_list.append(ssim)
                # --------------------------------------------------
                # Save outputs
                # --------------------------------------------------

                # Error only within the missing region, converted to [0,1].
                error_map = ((block[i:i+1] - x_recon[i:i+1]).abs() / 2.0) \
                            * (1.0 - mask[i:i+1])
                # Unique file names avoid overwriting earlier subjects.
                prefix = os.path.join(save_dir, f"test_{count:03d}")
                save_nifti(block[i:i+1], prefix + "_gt.nii.gz", True)
                save_nifti(x_recon[i:i+1], prefix + "_recon.nii.gz", True)
                save_nifti(mask[i:i+1], prefix + "_mask.nii.gz", False)
                save_nifti(error_map, prefix + "_error.nii.gz", False)

                count += 1

            if count >= max_images:
                break


    # ==========================================================
    # FINAL METRICS
    # ==========================================================

    average_psnr = total_psnr / max(count, 1)
    average_ssim = total_ssim / max(count, 1)
    print(
        f"\nAverage Metrics over {count} samples:\n"
        f"PSNR : {average_psnr:.4f} dB\n"
        f"SSIM : {average_ssim:.4f}"
    )

    psnr_avg_str = (
        f"Average PSNR for {len(val_psnr_list)} images = "
        f"{average_psnr:.4f} dB"
    )  
    ssim_avg_str = (
        f"Average SSIM for {len(val_ssim_list)} images = "
        f"{average_ssim:.4f}"
    )

    # ==========================================================
    # PLOT PSNR
    # ==========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(val_psnr_list)

    plt.xlabel("Image Index")
    plt.ylabel("PSNR (dB)")

    plt.title(
        f"Masked Region Reconstruction PSNR\n{psnr_avg_str}"
    )

    plt.grid(True)

    plt.savefig(
        RUN_DIR / "PSNR_model_testing_data.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved PSNR plot")
    
    # ==========================================================
    # PLOT SSIM
    # ==========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(val_ssim_list)

    plt.xlabel("Image Index")
    plt.ylabel("SSIM")

    plt.title(
        f"Masked Region Reconstruction SSIM\n{ssim_avg_str}"
    )

    plt.grid(True)

    plt.savefig(
        RUN_DIR / "SSIM_model_testing_data.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved SSIM plot")

