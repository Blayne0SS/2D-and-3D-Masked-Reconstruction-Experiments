import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# =============================================
# Main execution
# This script runs a grayscale masked diffusion experiment.
# =============================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    if device.type == "cpu":
        sys.exit("GPU not available")
    torch.cuda.empty_cache()


    patch_size=7                 #Represents the nunber of pixels per patch
    block_size=1                 #Represents the number of patches in a block (3 = 3x3 square)
    masking_percentage =0.20     #1 being 100
    img_size=(192,256)           #Image size
    base_channel_number=256      #legacy varible 
    number_of_epochs      = 1801 #number of epochs   
    lr_option         = 4e-5      #learning rate 
    CPU_num_workers = 1 #1          #CPU CORES used 
    GPU_batch_size=35 #35            #Images stored on gpu
    T = 1000                     #Steps


    Training_data_folder="/mnt/hpccs01/home/n10514821/X_ray_models/Data/mrart"
    #Training_data_folder = "/home/blayne/Models/MRI_SCANS_DATA/mrart/"

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    
    # Load image paths and create separate training, validation and test groups.

    def prepare_chexpert_small(
        chexpert_train_folder,
        class_ratio=0.5,
        img_size=img_size,
        split_ratios={'train': 0.7, 'val': 0.20, 'test': 0.10},
        seed=42,
        patch_size=patch_size,
        block_size=block_size

    ):
        np.random.seed(seed)

        # -----------------------------------
        # 2. Load images into memory (numpy)
        # -----------------------------------
        #Due to the use of GroupShuffleSplit, images have to shifted to the 
        #numpy array 
        X = []
        y = []
        groups = []  
        for root, _, files in os.walk(chexpert_train_folder):
            for file_name in files:
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(root, file_name)
                    X.append(img_path)
                    y.append(0)
                    groups.append(root)  


        if len(X) == 0:
            raise RuntimeError(
                f"No images were loaded from {chexpert_train_folder}"
            )
        print(len(X))
        
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


        # --- Save original images before modifying ---
        X_train_orig = X_train.copy()
        X_val_orig   = X_val.copy()
        X_test_orig  = X_test.copy()


        W,H= img_size[0],img_size[1]
        #create x and y coordinates

        def generate_random_coordinates(configured_data_set):
            data_locations=[]
            N=len(configured_data_set)
            print(N)
            for i in range(N):    
                x_cord= int(random.uniform(W/10,W-W/10))
                y_cord= int(random.uniform(H/10,H-H/10))            
                data_locations.append({
                    'y':int(y_cord),
                    'x':int(x_cord),
                    'patch_size':int(patch_size),
                    'block_size':int(block_size)
                })
            return data_locations             
            


        
        dot_train= generate_random_coordinates(X_train)
        dot_val= generate_random_coordinates(X_val)
        dot_test= generate_random_coordinates(X_test)
        
        train_groups = set(groups[train_idx])
        val_groups = set(groups_rest[val_idx])
        test_groups = set(groups_rest[test_idx])

        print("Train ∩ Val:", train_groups & val_groups)
        print("Train ∩ Test:", train_groups & test_groups)
        print("Val ∩ Test:", val_groups & test_groups)
        return {
            "Train":{ 'train':(X_train, y_train), 'OG_train':(X_train_orig, y_train),'dots':dot_train},
            "Val": {'val':(X_val, y_val), 'OG_train':(X_val_orig,y_val),'dots':dot_val},
            "Test": {'test':(X_test, y_test), 'OG_train':(X_test_orig,y_test),'dots':dot_test},
        }
    


    # Load an image and its mask information for the diffusion model.

    class XRayDataset(Dataset):
        def __init__(
            self,
            paths,            # list of image paths
            y,
            OG_train=None,    # optional numpy images of same length
            dots=None,
            transform=None,
            return_dot_info=False,
            return_index=False,
            return_orig=False,
    ):
            self.paths = paths
            self.y = y
            self.OG_train = OG_train
            self.dots = dots
            self.transform = transform

            self.return_dot_info = return_dot_info
            self.return_index = return_index
            self.return_orig = return_orig

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            # --- Load image from path ---
            path = self.paths[idx]
            label = self.y[idx]

            with Image.open(path) as img:
                img = img.convert("L")  # grayscale

                # Keep original BEFORE transforms
                if self.return_orig and self.OG_train is None:
                    image_orig = img.copy()
                else:
                    image_orig = None

            # --- Use provided OG_train if available ---
            if self.return_orig and self.OG_train is not None:
                image_orig_path = self.OG_train[idx]
                with Image.open(image_orig_path) as img:
                    image_orig = img.copy()  # keep PIL.Image

            # --- Apply transforms ---
            if self.transform is not None:
                image = self.transform(img)
                if image_orig is not None:
                    image_orig = self.transform(image_orig)
            else:
                image = transforms.ToTensor()(img)
                if image_orig is not None:
                    image_orig = transforms.ToTensor()(image_orig)

            # --- Build output ---
            out = [image, label]

            if self.return_orig and image_orig is not None:
                out.append(image_orig)

            if self.return_dot_info and self.dots is not None:
                out.append(self.dots[idx])

            if self.return_index:
                out.append(idx)

            return tuple(out)
    
    data = prepare_chexpert_small(
        chexpert_train_folder=Training_data_folder,
        class_ratio=0.1,
        img_size=img_size,
        
        
    )

    train_block = data["Train"]
    val_block   = data["Val"]
    test_block  = data["Test"]




    X_train, y_train = train_block["train"]
    X_val, y_val     = val_block["val"]
    X_test, y_test   = test_block["test"]




    X_train_orig, _ = train_block["OG_train"]
    X_val_orig, _ = val_block["OG_train"]
    X_test_orig, _ =test_block["OG_train"]

    dots_train = train_block["dots"]
    dots_val =val_block["dots"]
    dots_test=test_block["dots"]


    print(X_train.shape, y_train.shape)
    print(len(X_train))
    print(len(X_val))
    print(len(X_test))


    # ============================================
    # Define transforms
    # ============================================
    base_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(img_size),
    transforms.CenterCrop((128,128)), #192,256 #128,128, do 32,32 
    transforms.ToTensor(),

    transforms.Normalize(mean=[0.5], std=[0.5])   # -> range [-1,1]
    ])

    train_dataset = XRayDataset(
        paths=X_train,
        y=y_train,
        OG_train=X_train_orig,
        dots=dots_train,
        transform=base_transform,
        return_orig=True,
        return_dot_info=True,
        return_index=True
    )

    val_dataset = XRayDataset(
        paths=X_val,
        y=y_val,
        OG_train=X_val_orig,
        dots=dots_val,
        transform=base_transform,
        return_orig=True,
        return_dot_info=True,
        return_index=True
    )

    test_dataset = XRayDataset(
        paths=X_test,
        y=y_test,
        OG_train=X_test_orig,
        dots=dots_test,
        transform=base_transform,
        return_orig=True,
        return_dot_info=True,
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
            prefetch_factor=4
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=GPU_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=CPU_num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=GPU_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=CPU_num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )




    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create the masked image region used for the reconstruction task.

    def random_block_mask_2(
        image,
        dot_info,
        mask_percent=None,
        device=None
    ):

        B, C, H, W = image.shape
        device = device or image.device

        mask = torch.ones((B, 1, H, W), device=device)

        # --- Handle inputs ---
        def to_tensor(v):
            return torch.tensor(v, device=device).repeat(B) if not torch.is_tensor(v) else v.to(device)

        xs = to_tensor(dot_info["x"])
        ys = to_tensor(dot_info["y"])
        patch_sizes = to_tensor(dot_info["patch_size"])
        block_sizes = to_tensor(dot_info["block_size"])

        for b in range(B):
            x = int(xs[b].item())
            y = int(ys[b].item())
            patch_size = int(patch_sizes[b].item())
            block_size = int(block_sizes[b].item())

            # original was (192,256)
            x = int(x * (W / 192))
            y = int(y * (H / 256))

    
            n_h = H // patch_size
            n_w = W // patch_size

            px = x // patch_size
            py = y // patch_size

      
            half = block_size  

            start_y = max(0, py - half)
            end_y   = min(n_h, py + half + 1)

            start_x = max(0, px - half)
            end_x   = min(n_w, px + half + 1)

            
            h_start = start_y * patch_size
            h_end   = end_y * patch_size

            w_start = start_x * patch_size
            w_end   = end_x * patch_size

            mask[b, :, h_start:h_end, w_start:w_end] = 0

        return image * mask, mask
        
    #___________________________________________________________
    #Diffusion set up and show images

    # Create the sigmoid noise schedule for the diffusion steps.

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
        

    # Add noise to the masked image region during training.

    def forward_diffusion_sample_section_with_mask(x_0, t, dot_info=None, device="cuda"):

        x_0 = x_0.to(device)

        # Generate random block mask
        masked_image, mask = random_block_mask_2(x_0, dot_info,masking_percentage)

        # Standard diffusion noise
        noise = torch.randn_like(x_0, device=device)

        # Get diffusion coefficients for batch

        sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod.to(device), t.to(device), x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod.to(device), t.to(device), x_0.shape)
        # Apply diffusion only to masked region
        x_noisy = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        x_noisy = x_noisy * (1 - mask) + x_0 * mask

        return x_noisy, mask, noise


    # Define beta schedule
    betas = sigmoid_beta_schedule(T, s=-5, e=5, tau=0.5)

    # Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def show_tensor_image(image):
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            #transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            transforms.Lambda(lambda t: t.squeeze(0)),
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])

        # Take first image of batch
        #if len(image.shape) == 4:
        #    image = image[0, :, :, :] 
        #plt.imshow(reverse_transforms(image))
        if len(image.shape) == 4:
            image = image[0]

        plt.imshow(reverse_transforms(image), cmap='gray')
        plt.axis('off')


    
   
       
    #___________________________________________________________175146
    #MODEL CONFIG 
    # -------------------------------------------------
    # Residual Block (supports downsample/upsample)
    # -------------------------------------------------
    class ResidualBlock(nn.Module):
        def __init__(self, in_ch, out_ch, time_dim, downsample=False):
            super().__init__()

            self.downsample = downsample
            stride = 2 if downsample else 1

            self.norm1 = nn.GroupNorm(8, in_ch)
            self.act1 = nn.SiLU()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)

            self.time_mlp = nn.Linear(time_dim, out_ch)

            self.norm2 = nn.GroupNorm(8, out_ch)
            self.act2 = nn.SiLU()
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

            if in_ch != out_ch or downsample:
                self.res_conv = nn.Conv2d(in_ch, out_ch, 1, stride=stride)
            else:
                self.res_conv = nn.Identity()

        def forward(self, x, t):
            h = self.conv1(self.act1(self.norm1(x)))

            time_emb = self.time_mlp(t)
            time_emb = time_emb[..., None, None]
            h = h + time_emb

            h = self.conv2(self.act2(self.norm2(h)))

            return h + self.res_conv(x)


    # -------------------------------------------------
    # Attention Block
    # -------------------------------------------------
    class AttentionBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.norm = nn.GroupNorm(8, channels)
            self.qkv = nn.Conv2d(channels, channels * 3, 1)
            self.proj = nn.Conv2d(channels, channels, 1)

        def forward(self, x):
            B, C, H, W = x.shape
            h = self.norm(x)

            qkv = self.qkv(h)
            q, k, v = torch.chunk(qkv, 3, dim=1)

            q = q.reshape(B, C, H * W).permute(0, 2, 1)
            k = k.reshape(B, C, H * W)
            v = v.reshape(B, C, H * W).permute(0, 2, 1)

            attn = torch.bmm(q, k) * (C ** -0.5)
            attn = torch.softmax(attn, dim=-1)

            out = torch.bmm(attn, v)
            out = out.permute(0, 2, 1).reshape(B, C, H, W)

            return x + self.proj(out)


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


    # Define the U-Net that predicts noise for the grayscale diffusion model.

    class DiffusionUNet(nn.Module):
        def __init__(self, image_channels=1, mask_channels=1, base=base_channel_number):
            super().__init__()

            in_ch = image_channels + mask_channels
            time_dim = base * 4

            # Time embedding
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(base),
                nn.Linear(base, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )

            # Initial
            self.init = nn.Conv2d(in_ch, base, 3, padding=1)

            # Down
            self.down1 = ResidualBlock(base, base*2, time_dim, downsample=True)
            self.down2 = ResidualBlock(base*2, base*4, time_dim, downsample=True)
            self.down3 = ResidualBlock(base*4, base*8, time_dim, downsample=True)

            self.skip3_1 = ResidualBlock(base*4 + base*8, base*4, time_dim)

            # x2 refinement (uses x2 + up(x3))
            self.skip2_1 = ResidualBlock(base*2 + base*4, base*2, time_dim)

            # x1 refinement (uses x1 + up(x2))
            self.skip1_1 = ResidualBlock(base + base*2, base, time_dim)


            # Bottleneck
            self.mid1 = ResidualBlock(base*8, base*8, time_dim)
            self.attn = AttentionBlock(base*8)
            self.mid2 = ResidualBlock(base*8, base*8, time_dim)

            # Up
            self.up_sample1 = nn.ConvTranspose2d(base*8, base*8, 2, 2)  # 1024 → 1024
            self.up_sample2 = nn.ConvTranspose2d(base*4, base*4, 2, 2)  # 512 → 512
            self.up_sample3 = nn.ConvTranspose2d(base*2, base*2, 2, 2)  # 256 → 256

            self.up1 = ResidualBlock(base*8 + base*4, base*4, time_dim)   # 1024+512 → 512
            self.up2 = ResidualBlock(base*4 + base*2, base*2, time_dim)   # 512+256 → 256
            self.up3 = ResidualBlock(base*2 + base, base, time_dim)       # 256+128 → 128

            self.final = nn.Conv2d(base, image_channels, 1)

        def forward(self, x, t, mask=None):

            if mask is not None:
                x = torch.cat([x, mask], dim=1)
            else:
                zeros = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)
                x = torch.cat([x, zeros], dim=1)

            t = self.time_mlp(t)

            # Down
        
            x1 = self.init(x)      # 64
            x2 = self.down1(x1,t)  # 128
            x3 = self.down2(x2,t)  # 256
            x4 = self.down3(x3,t)  # 512

            # Bottleneck
            x_mid = self.mid1(x4,t)
            x_mid = self.attn(x_mid)
            x_mid = self.mid2(x_mid,t)

            x4_up = self.up_sample1(x_mid)   # matches x3 spatial
            x3_up = self.up_sample2(x3)      # matches x2 spatial
            x2_up = self.up_sample3(x2)      # matches x1 spatial

            # Refined skips
            x3_1 = self.skip3_1(torch.cat([x3, x4_up], dim=1), t)
            x2_1 = self.skip2_1(torch.cat([x2, x3_up], dim=1), t)
            x1_1 = self.skip1_1(torch.cat([x1, x2_up], dim=1), t)

            # Up
            x = self.up_sample1(x_mid)   # 512
            x = torch.cat([x, x3_1], dim=1)   # 512+256=768
            x = self.up1(x,t)

            x = self.up_sample2(x)
            x = torch.cat([x, x2_1], dim=1) 
            x = self.up2(x,t)

            x = self.up_sample3(x)
            x = torch.cat([x, x1_1], dim=1) 
            x = self.up3(x,t)

            return self.final(x)




    model = DiffusionUNet().to(device)
    torch._dynamo.config.suppress_errors = True
    model = torch.compile(model)

    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model

    # Calculate the diffusion training loss for the current batch.

    def get_loss(model, x_0, dot_info, t):
        x_noisy, mask, noise = forward_diffusion_sample_section_with_mask(
            x_0, t, dot_info, device
        )


        mask = mask.float()  # prevent dtype issues

        noise_pred = model(x_noisy, t, mask)

        loss = F.mse_loss(
            noise * (1 - mask),
            noise_pred * (1 - mask)
        )

        # optional: catch NaNs
        if not torch.isfinite(loss):
            print("NaN detected in loss!")
            exit()

        return loss


    # Run one reverse diffusion step.

    @torch.no_grad()
    def sample_timestep(x, mask, t):
        betas_t = get_index_from_list(betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
        posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

        # --- clamp dangerous values ---
        sqrt_one_minus_alphas_cumprod_t = torch.clamp(sqrt_one_minus_alphas_cumprod_t, min=1e-8)
        posterior_variance_t = torch.clamp(posterior_variance_t, min=1e-8)

        noise_pred = model(x, t, mask)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)
        if t[0] == 0:
            return model_mean

        noise = torch.randn_like(x)
        updated = model_mean + torch.sqrt(posterior_variance_t) * noise

        if mask is not None:
            return updated * (1 - mask) + x * mask
        else:
            return updated
        
    
    # Generate and save a sample reconstruction during training.

    @torch.no_grad()
    def sample_plot_image(model, device, x_original, dot_info, num_images=8,save_path=None):
        """
        Plots:
        - Original image
        - Fully diffused image
        - Intermediate denoising steps
        """
        model.eval()
        T_local = T

        # Ensure batch dimension
        if x_original.dim() == 3:
            x_original = x_original.unsqueeze(0)
        
        x_original = x_original.to(device)

        # Fully noisy image (t = T-1)
        t_full = torch.full((x_original.shape[0],), T_local - 1, device=device, dtype=torch.long)
        x_noisy, mask, _ = forward_diffusion_sample_section_with_mask(
            x_original, t_full, dot_info, device
        )

        # --- Generate evenly spaced timesteps INCLUDING t=0 ---
        timesteps = torch.linspace(T_local - 1, 0, steps=num_images, dtype=torch.long)

        # Collect images
        images_to_plot = [x_original, x_noisy]
        collected = {}

        img = x_noisy.clone()

        for i in reversed(range(T_local)):
            t = torch.full((x_original.shape[0],), i, device=device, dtype=torch.long)
            img = sample_timestep(img, mask, t)

            # Save if timestep is in our selected set
            if i in timesteps:
                collected[int(i)] = img.clone()

        # Sort timesteps from high → low
        for t_val in sorted(collected.keys(), reverse=True):
            images_to_plot.append(collected[t_val])

        # Move to CPU
        images_to_plot_cpu = [im.cpu() for im in images_to_plot]

        # Titles
        titles = ["Original", "Fully Diffused"]
        titles += [f"t={t}" for t in sorted(collected.keys(), reverse=True)]

        # Plot
        total_plots = len(images_to_plot_cpu)
        plt.figure(figsize=(20, 6))

        for idx, (im, title) in enumerate(zip(images_to_plot_cpu, titles)):
            plt.subplot(1, total_plots, idx + 1)
            show_tensor_image(im)
            plt.title(title)

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        #plt.show()



    #_______________________________________________________________
    #MODEL TRAINING 

    # Track validation results and stop training when they no longer improve.

    class EarlyStopping:
        def __init__(self, patience=20, min_delta=0.0, save_path="V2_GREY_best_model.pt"):
            self.patience = patience
            self.min_delta = min_delta
            self.save_path = save_path

            self.best_loss = float("inf")
            self.counter = 0
            self.early_stop = False

        def __call__(self, val_loss, model):

            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
                torch.save(model.state_dict(), self.save_path)
                #print("Validation improved — saving model")
            else:
                self.counter += 1
                #print(f"No improvement ({self.counter}/{self.patience})")

                if self.counter >= self.patience:
                    self.early_stop = True

    # Maintain a smoothed copy of model weights for evaluation and sampling.

    class EMA:
        def __init__(self, model, decay=0.999):
            self.decay = decay
            self.ema_model = copy.deepcopy(model)
            self.ema_model.eval()

            for param in self.ema_model.parameters():
                param.requires_grad_(False)

        @torch.no_grad()
        def update(self, model):
            model_state = model.state_dict()
            ema_state = self.ema_model.state_dict()

            for name in ema_state.keys():
                ema_state[name].mul_(self.decay).add_(
                    model_state[name], alpha=1 - self.decay
                )

            self.ema_model.load_state_dict(ema_state)


    

    # Calculate PSNR only over the pixels hidden by the mask.

    def compute_psnr_masked(img1, img2, mask):
        # Convert [-1,1] → [0,1]
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2

        # Only evaluate masked regions
        region = (1 - mask)

        mse = F.mse_loss(img1 * region, img2 * region)

        if mse == 0:
            return float('inf')

        return (10 * torch.log10(1.0 / mse)).item()

    # Reconstruct a masked image by running the reverse diffusion steps.

    @torch.no_grad()
    def reconstruct_image(model, x_0, dot_info, device):
        T_local = T

        t_full = torch.full((x_0.shape[0],), T_local - 1, device=device, dtype=torch.long)

        x_noisy, mask, _ = forward_diffusion_sample_section_with_mask(
            x_0, t_full, dot_info, device
        )

        img = x_noisy.clone()

        for i in reversed(range(T_local)):
            t = torch.full((x_0.shape[0],), i, device=device, dtype=torch.long)
            img = sample_timestep(img, mask, t)

        return img, mask   

    model = model.to(device).to(memory_format=torch.channels_last)
    ema = EMA(model, decay=0.999)

    train_losses = []
    val_losses = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_option, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=number_of_epochs,
    eta_min=1e-6
    )
    
    early_stopper = EarlyStopping(patience=2500, min_delta=1e-4)
    Print_Progress=1

    val_psnr_list = []

    scaler = torch.amp.GradScaler()
    for epoch in range(number_of_epochs):
        Print_Progress=1
        # ---- TRAIN ----
        model.train()
        train_loss_epoch = 0

        for batch in train_loader:
            optimizer.zero_grad()

            x_0 = batch[0].to(device)
            x_0 = x_0.to(device, memory_format=torch.channels_last)
            dot_info = batch[3]  
            
            B = x_0.shape[0]
            t = torch.randint(0, T, (B,), device=device).long()
            
            # Calculate loss
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = get_loss(model, x_0, dot_info, t)

            if not torch.isfinite(loss):
                print(f"NaN detected at epoch {epoch}")
                exit()
                
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)  # required before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            
            if epoch > 5:
                ema.update(model)

            if epoch % 10 == 0 and Print_Progress==1 :
                    print(f"Epoch {epoch} | Loss: {loss.item()} ")
                    sample_plot_image(
                        model=ema.ema_model,
                        device=device,
                        x_original=x_0[0:1],  # keep batch dimension
                        dot_info={k: v[0:1] for k, v in batch[3].items()},
                        num_images=8,
                        save_path=f"V2_GREY_Linear_V1_epoch_{epoch}.png"
                    )
                    Print_Progress=2


            train_loss_epoch += loss.item()

        train_loss_epoch /= len(train_loader)
        train_losses.append(train_loss_epoch)


        # ---- VALIDATION ----
        model.eval()
        val_loss_epoch = 0
        val_psnr_epoch = 0
        with torch.no_grad():
            for batch in val_loader:
                x_0 = batch[0].to(device)  
                x_0 = x_0.to(device, memory_format=torch.channels_last)
                dot_info = batch[3]  
                
                B = x_0.shape[0]
                t = torch.randint(0, T, (B,), device=device).long()

                # Calculate loss
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = get_loss(ema.ema_model, x_0, dot_info, t)

                            
                val_loss_epoch += loss.item()

                # --- PSNR (on small subset for speed) ---
                if epoch % 100 == 0 and epoch != 0:
                    x_sample = x_0[0:1]  
                    dot_sample = {k: v[0:1] for k, v in dot_info.items()}

                    x_recon, mask  = reconstruct_image(ema.ema_model, x_sample, dot_sample, device)
                    psnr = compute_psnr_masked(x_sample, x_recon, mask)
                    val_psnr_epoch += psnr

        val_loss_epoch /= len(val_loader)
        val_losses.append(val_loss_epoch)

        if epoch % 100 == 0 and epoch != 0:
            val_psnr_epoch /= len(val_loader)
            val_psnr_list.append(val_psnr_epoch)
            psnr_str = f"| PSNR: {val_psnr_epoch:.2f}"
        else:
            psnr_str = ""
        
        print(f"Epoch {epoch} | Train: {train_loss_epoch:.6f} | Val: {val_loss_epoch:.6f} {psnr_str}")

        # ---- Early Stopping ----
        early_stopper(val_loss_epoch, model)

        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break  

        scheduler.step()


    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig("V2_GREY_loss_curve.png", dpi=300)
    plt.close()

    print("Saved loss curve to V2_GREY_linear_V1_loss_curve.png")

    plt.figure()
    plt.plot(val_psnr_list, label="Validation PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.title("PSNR over Training")
    plt.savefig("V2_GREY_PSNR_curve_During_training.png", dpi=300)
    plt.close()


    torch.save(model.state_dict(), "MB_GREY_SIG_V2_raw_final.pt")
    torch.save(ema.ema_model.state_dict(), "MB_GREY_SIG_V2_EMA_final.pt")


    # Calculate PSNR only over the pixels hidden by the mask.

    def compute_psnr_masked(img1, img2, mask):
    # Convert [-1,1] → [0,1]
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2

        # Only evaluate masked regions
        region = (1 - mask)

        mse = F.mse_loss(img1 * region, img2 * region)

        mse = torch.clamp(mse, min=1e-10)

        return 10 * torch.log10(1.0 / mse).item()

    # Reconstruct a masked image by running the reverse diffusion steps.

    @torch.no_grad()
    def reconstruct_image(model, x_0, dot_info, device):
        T_local = T

        t_full = torch.full((x_0.shape[0],), T_local - 1, device=device, dtype=torch.long)

        x_noisy, mask, _ = forward_diffusion_sample_section_with_mask(
            x_0, t_full, dot_info, device
        )

        img = x_noisy.clone()

        for i in reversed(range(T_local)):
            t = torch.full((x_0.shape[0],), i, device=device, dtype=torch.long)
            img = sample_timestep(img, mask, t)

        return img, mask 

    train_losses = []
    val_losses = []

    Print_Progress=1


    scaler = torch.amp.GradScaler()

    total_psnr_value = 0


    max_images = 200
    model.eval()
    total_psnr = 0
    count = 0
    val_psnr_list = []

    with torch.no_grad():
        for batch in test_loader:
            x_0 = batch[0].to(device)  
            dot_info = batch[3]  

            # 🔥 reconstruct whole batch at once
            x_recon, mask = reconstruct_image(ema.ema_model, x_0, dot_info, device)

            B = x_0.shape[0]

            for i in range(B):
                if count >= max_images:
                    break
                #print("starting")  
                psnr = compute_psnr_masked(
                    x_0[i:i+1],
                    x_recon[i:i+1],
                    mask[i:i+1]
                )
                print(f" PSNR: {psnr}")
                total_psnr += psnr
                val_psnr_list.append(psnr)
                count += 1

            if count >= max_images:
                break

    average_psnr = total_psnr / count
    print(f"Average PSNR over {count} images: {average_psnr:.2f}")
        


    average_psnr = sum(val_psnr_list) / len(val_psnr_list)
    psnr_avg_str= f"average PSNR for {len(val_psnr_list)} images is {average_psnr:.2f}"
    print(f" {psnr_avg_str}")


    plt.figure()
    plt.plot(val_psnr_list, label="Validation PSNR over images")
    plt.xlabel("Image Index")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.title(f"PSNR over Training, average: {psnr_avg_str}")
    plt.savefig("PSNR_EMA_MODEL_Testing_data.png", dpi=300)
    plt.close()

    train_losses = []
    val_losses = []

    Print_Progress=1

    val_psnr_list = []

    scaler = torch.amp.GradScaler()

    total_psnr_value = 0


    max_images = 200
    model.eval()
    total_psnr = 0
    count = 0
    val_psnr_list = []

    with torch.no_grad():
        for batch in test_loader:
            x_0 = batch[0].to(device)  
            dot_info = batch[3]  

            # 🔥 reconstruct whole batch at once
            x_recon, mask = reconstruct_image(model, x_0, dot_info, device)

            B = x_0.shape[0]

            for i in range(B):
                if count >= max_images:
                    break
                #print("starting")  
                psnr = compute_psnr_masked(
                    x_0[i:i+1],
                    x_recon[i:i+1],
                    mask[i:i+1]
                )
                print(f" PSNR: {psnr}")
                total_psnr += psnr
                val_psnr_list.append(psnr)
                count += 1

            if count >= max_images:
                break

    average_psnr = total_psnr / count
    print(f"Average PSNR over {count} images: {average_psnr:.2f}")
        


    average_psnr = sum(val_psnr_list) / len(val_psnr_list)
    psnr_avg_str= f"average PSNR for {len(val_psnr_list)} images is {average_psnr:.2f}"
    print(f" {psnr_avg_str}")


    plt.figure()
    plt.plot(val_psnr_list, label="Validation PSNR over images")
    plt.xlabel("Image Index")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.title(f"PSNR over Training, average: {psnr_avg_str}")
    plt.savefig("PSNR_model_testing_data.png", dpi=300)
    plt.close()