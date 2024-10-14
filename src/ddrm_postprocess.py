
import os 
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ddrm')))
import traceback
import shutil
import logging
import yaml
import torch
import numpy as np
import argparse
from runners.diffusion import *
from functions.svd_replacement import SuperResolution
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
from PIL import Image
from utils import tensor2img
import types
import torch.nn.functional as F
from datasets import center_crop_arr

def sample(self, images_tensor):
        cls_fn = None
        
        config_dict = vars(self.config.model)
        model = create_model(**config_dict)
        if self.config.model.use_fp16:
                model.convert_to_fp16()
        ckpt = "./model_zoos/256x256_diffusion_uncond.pt"
        if not os.path.exists(ckpt):
            download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', ckpt)
            
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        model.to(self.device)
        model.eval()
        model = torch.nn.DataParallel(model)

        output_images = self.sample_sequence(model, images_tensor, cls_fn)
        
        return output_images

def sample_sequence(self, model, images_tensor, cls_fn=None):

    args, config = self.args, self.config

    dataset = CustomTensorDataset(
        images_tensor,
        transform=transforms.Compose([
            partial(center_crop_arr, image_size=config.data.image_size),
            transforms.ToTensor()
        ])
    )

    test_dataset = dataset
    
    device_count = torch.cuda.device_count()
    
    args.subset_start = 0
    args.subset_end = len(test_dataset)

    print(f'Dataset has size {len(test_dataset)}')    
    
    def seed_worker(worker_id):
        worker_seed = args.seed % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)
    val_loader = data.DataLoader(
        test_dataset,
        batch_size=config.sampling.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    

    ## get degradation matrix ##
    deg = args.deg
    H_funcs = None

    blur_by = int(deg[2:])
    H_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)

    args.sigma_0 = 2 * args.sigma_0 #to account for scaling to [-1,1]
    sigma_0 = args.sigma_0
    
    print(f'Start from {args.subset_start}')
    idx_init = args.subset_start
    idx_so_far = args.subset_start
    avg_psnr = 0.0

    pbar = tqdm.tqdm(val_loader)
    for x_orig, classes in pbar:
        x_orig = x_orig.to(self.device)
        x_orig = data_transform(self.config, x_orig)

        y_0 = F.avg_pool2d(x_orig, kernel_size=8, stride=8)
        y_0 = y_0.view(config.sampling.batch_size, -1)


        os.makedirs(self.args.image_folder + '/enhance', exist_ok=True)

        ##Begin DDIM
        x = torch.randn(
            y_0.shape[0],
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        with torch.no_grad():
            x, _ = self.sample_image(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes)

        x = [inverse_data_transform(config, y) for y in x]

        for i in [-1]: 
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{idx_so_far + j}_diffusion.png")
                )
                if i == len(x)-1 or i == -1:
                    orig = inverse_data_transform(config, x_orig[j])
                    mse = torch.mean((x[i][j].to(self.device) - orig) ** 2)
                    psnr = 10 * torch.log10(1 / mse)
                    avg_psnr += psnr

        idx_so_far += y_0.shape[0]

        pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

    return x


class CustomTensorDataset(Dataset):
    def __init__(self, images_tensor, labels_tensor=None, transform=None):
        self.images_tensor = images_tensor
        self.labels_tensor = labels_tensor
        self.transform = transform

    def __len__(self):
        return len(self.images_tensor)

    def __getitem__(self, idx):
        image = self.images_tensor[idx]
        image = tensor2img(image)  # Convert tensor to image array
        image = Image.fromarray(image)  # Convert numpy array to PIL image
        if self.transform:
            image = self.transform(image)
        if self.labels_tensor is not None:
            label = self.labels_tensor[idx]
        else:
            label = -1  # Use -1 as a placeholder label
        return image, label  # Return a tuple of (image, label)


def set_args_and_config(device, output_dir):
    class Args:
        config = os.path.abspath("./utils/config.yaml")
        seed = 1234
        exp = "./ddrm/exp"
        doc = "imagenet_ood"
        comment = ""
        verbose = "info"
        sample = False
        image_folder = 'output_dir'
        ni = True
        timesteps = 50
        deg = "sr8" 
        sigma_0 = 0.05
        eta = 0.85 # 0.85
        etaB = 1
        subset_start = -1
        subset_end = -1

    args = Args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    args.image_folder = './' + output_dir

    # add device
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def postprocess(images_tensor, output_dir, device):

    args, config = set_args_and_config(device, output_dir)

    runner = Diffusion(args, config)
    runner.sample = types.MethodType(sample, runner)
    runner.sample_sequence = types.MethodType(sample_sequence, runner)
    
    postprocessed_images = runner.sample(images_tensor)

    return postprocessed_images
