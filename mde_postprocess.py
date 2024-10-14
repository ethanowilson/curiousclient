import importlib
import os
import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from MDE.models import MDTv2_XL_2
from torch.optim import Adam
from MDE import create_diffusion
import numpy as np
from torchvision import models
from torchmetrics.functional import structural_similarity_index_measure as ssim

# Create Output Directory
output_dir = "./mde_reconstructions"
os.makedirs(output_dir, exist_ok=True)

# Set target image path
target_image_path = 'experiments/1008_1527/0/2_recon.png'

# Imagenet class labels to condition the model with:
class_label = [199]

# Load the image
target_image = Image.open(target_image_path)

# Define a transform to upscale the image to 256x256
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Apply the transform
target_image_tensor = transform(target_image)

# Convert the target to the range [-1, 1]
target_image_tensor = 2 * target_image_tensor - 1

# Save the target image
save_image(target_image_tensor,f"{output_dir}/target_image.jpg",normalize=True,value_range=(-1, 1))

# Setup PyTorch:
torch.manual_seed(1)
torch.set_grad_enabled(True)
device = "cuda" if torch.cuda.is_available() else "cpu"
num_sampling_steps = 250
cfg_scale = 4.0
pow_scale = 0.01 # large pow_scale increase the diversity, small pow_scale increase the quality.
model_path = 'model_zoos/mdt_xl2_v2_ckpt.pt'

# Load model:
image_size = 256
assert image_size in [256], "We provide pre-trained models for 256x256 resolutions for now."
latent_size = image_size // 8
model = MDTv2_XL_2(input_size=latent_size, decode_layer=4).to(device)

state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict)
model.eval()
diffusion = create_diffusion(str(num_sampling_steps))
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)


# Create sampling noise:
n = len(class_label)
z = torch.randn(2, 4, latent_size, latent_size, device=device,requires_grad=True)
y = torch.tensor(class_label, device=device)

# Setup classifier-free guidance:
y_null = torch.tensor([1000] * n, device=device)
y = torch.cat([y, y_null], 0)

model_kwargs = dict(y=y, cfg_scale=cfg_scale, scale_pow=pow_scale)

# Set up the optimizer for input noise
optimizer = Adam([z], lr=1e-2)


# Combined loss function
def combined_loss(pred, target):
    l2_loss = nn.MSELoss()(pred, target)
    return l2_loss


def reconstruct_sample(initial_noise, pred_eps, alpha_bars, alpha_bar_prevs):

    x_t = initial_noise

    for t in range(len(pred_eps)):

        xstart = (torch.sqrt(1.0 / alpha_bars[t])*x_t) - (torch.sqrt(1.0 / alpha_bars[t] - 1)*pred_eps[t])

        mean_pred = (xstart * torch.sqrt(alpha_bar_prevs[t]) + torch.sqrt(1 - alpha_bar_prevs[t]) * pred_eps[t])

        x_t = mean_pred

    return mean_pred


num_epochs = 100
do_latent = False

encoded_target = vae.encode(target_image_tensor.unsqueeze(0).to(vae.device)).latent_dist.sample().mul_(0.18215)
decoded_target = vae.decode(encoded_target / 0.18215).sample

for epoch in range(num_epochs):

    with torch.no_grad():

        samples, pred_noise_steps, pred_xstart, alpha_bars, alpha_bar_prevs = diffusion.ddim_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )

        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample

    test_sample = reconstruct_sample(z, pred_noise_steps, alpha_bars, alpha_bar_prevs)
    test_sample, _ = test_sample.chunk(2, dim=0)  # Remove null class samples

    test_sample = vae.decode(test_sample / 0.18215).sample
    loss = combined_loss(test_sample, target_image_tensor.to(test_sample.device))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    save_image(test_sample, f"{output_dir}/test_sample{epoch}.jpg", normalize=True, value_range=(-1, 1))


    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
