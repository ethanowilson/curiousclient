import imp
import os
import math
import datetime
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import random
import re

def init_outputfolder(config):

    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)

    current_time = datetime.datetime.now()
    current_time_str = datetime.datetime.strftime(current_time, '%m%d_%H%M')

    output_dir = os.path.join(config.output_folder, (current_time_str))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "0"))

    return output_dir

def save_batch(output_dir, original_img, recon_img, keyword="recon", save_ori=True, raw=True):
    
    if save_ori:
        for i, img in enumerate(original_img):
            img_numpy = tensor2img(img)

            if img_numpy.ndim == 2:
                cv2.imwrite(os.path.join(output_dir, '{:d}_ori.png'.format(i)), img_numpy)
            else:
                cv2.imwrite(os.path.join(output_dir, '{:d}_ori.png'.format(i)), img_numpy[:, :, ::-1])

    for i, img in enumerate(recon_img):
        img_numpy = tensor2img(img)

        if img_numpy.ndim == 2:
                cv2.imwrite(os.path.join(output_dir, '{:d}_{:s}.png'.format(i, keyword)), img_numpy)
        else:
            if raw:
                cv2.imwrite(os.path.join(output_dir, '0','{:d}_{:s}.png'.format(i, keyword)), img_numpy[:, :, ::-1])
            else:
                cv2.imwrite(os.path.join(output_dir,'{:d}_{:s}.png'.format(i, keyword)), img_numpy[:, :, ::-1])


def label_to_onehot(target, num_classes=1000):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def freeze(model):
    for p in model.parameters():
        p.requires_grad = False

def jaccard(a, b):
    intersection = np.intersect1d(a, b)
    union = a.shape[0] + b.shape[0] - intersection.shape[0]
    return float(intersection.shape[0]) / union

def overlap_idx(a, b):
    intersection = np.intersect1d(a, b)
    denominator = b.shape[0]
    return float(intersection.shape[0]) / denominator

def preprocess(config, x, y, onehot):
    device = config.device
    
    if config.half:
        x = x.half()
        onehot = onehot.half()

    return x.to(config.device), y.to(config.device), onehot.to(config.device)


def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)


def tensor2img(tensor, min_max=(0, 1), out_type=np.uint8):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # squeeze first, then clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[:, :, :], (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

class Adam16(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        params = list(params)
        super(Adam16, self).__init__(params, defaults)
        # for group in self.param_groups:
            # for p in group['params']:
        
        self.fp32_param_groups = [p.data.float().cuda() for p in params]
        if not isinstance(self.fp32_param_groups[0], dict):
            self.fp32_param_groups = [{'params': self.fp32_param_groups}]

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group,fp32_group in zip(self.param_groups,self.fp32_param_groups):
            for p,fp32_p in zip(group['params'],fp32_group['params']):
                if p.grad is None:
                    continue
                    
                grad = p.grad.data.float()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], fp32_p)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
            
                fp32_p.addcdiv_(-step_size, exp_avg, denom)
                p.data = fp32_p.half()

        return loss

def load_images_to_tensors(directory):
    ori_images = []
    diffusion_images = []

    def natural_sort_key(file_name):
        match = re.match(r'(\d+)', file_name)
        return int(match.group()) if match else float('inf') 

    # Get sorted list of files in directory using the natural sort key
    files = sorted(os.listdir(directory), key=natural_sort_key)

    for file in files:
        if file.endswith('_ori.png'):
            # Load original image using OpenCV (BGR)
            img_bgr = cv2.imread(os.path.join(directory, file), cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                raise ValueError(f"Image {file} could not be loaded.")
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Resize to 128x128 if needed
            img_rgb = cv2.resize(img_rgb, (128, 128), interpolation=cv2.INTER_LINEAR)

            # Convert to (C, H, W) format and scale to [0, 1]
            img_np = img_rgb.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
            img_np = (img_np / 255)   # Scale to [0, 1]
            ori_images.append(img_np)
            
        elif file.endswith('_diffusion.png'):
            # Load reconstructed (diffusion) image using same procedure
            img_bgr = cv2.imread(os.path.join(directory, file), cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                raise ValueError(f"Image {file} could not be loaded.")
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Resize to 128x128 if needed
            img_rgb = cv2.resize(img_rgb, (128, 128), interpolation=cv2.INTER_LINEAR)

            # Convert to (C, H, W) format and scale to [0, 1]
            img_np = img_rgb.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
            img_np = (img_np / 255)   # Scale to [0, 1]
            diffusion_images.append(img_np)

    # Convert lists to tensors
    ori_tensor = torch.tensor(ori_images, dtype=torch.float32)
    diffusion_tensor = torch.tensor(diffusion_images, dtype=torch.float32)

    return ori_tensor, diffusion_tensor
        
