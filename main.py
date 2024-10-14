import pickle
import torch
from networks import nn_registry
from src.metric import Metrics
from src.dataloader import fetch_trainloader
from src.dataloader import fetch_mnist_loader
from src import fedlearning_registry
from src.attack import Attacker, grad_inv
from utils import *
from src.ddrm_postprocess import postprocess
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import torch.nn as nn


def main(config_file):

    config = load_config(config_file)
    output_dir = init_outputfolder(config)
    logger = init_logger(config, output_dir)

    torch.manual_seed(config.randomseed)

    # Fetch the training data loader
    if config.dataset == "ImageNet":
        train_loader = fetch_trainloader(config, shuffle=True)

    elif config.dataset == "MNIST":
        train_loader = fetch_mnist_loader(config.batch_size*config.num_clients, train=False, shuffle=True)


    n = config.num_clients  # Number of clients
    client_data_size = config.batch_size # Number of training images per client

    criterion = cross_entropy_for_onehot

    # Fetch batch from the data loader
    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx == 0:
            break

    # Convert to one-hot encoded label
    onehot = label_to_onehot(y, num_classes=config.num_classes)

    # Load FL global model
    model = (nn_registry[config.global_model](config)).to(config.device)
    
    # Federated learning algorithm on a single device
    fedalg = fedlearning_registry[config.fedalg](criterion, model, config)
        

    client_grads = [] # List to store gradients transmitted by each client

    # Process data for each client
    for client_idx in range(n):

        # Select subset of images and labels to serve as local data for each client
        client_x = x[(client_idx * client_data_size):((client_idx + 1) * client_data_size)].to(config.device)
        client_y = y[(client_idx * client_data_size):((client_idx + 1) * client_data_size)].to(config.device)

        # Convert label to onehot encoding
        client_onehot = label_to_onehot(client_y, num_classes=config.num_classes)

        client_x, client_y, client_onehot = preprocess(config, client_x, client_y, client_onehot)

        # Retrieve gradient from client local data
        grad = fedalg.client_grad(client_x, client_onehot)

        client_grads.append(grad)

    # Aggregate gradients, following FedAvg algorithm
    avg_grad = [torch.zeros_like(g) for g in client_grads[0]]

    for grad in client_grads:
        for i, g in enumerate(grad):
            avg_grad[i] += g

    avg_grad = [g / n for g in avg_grad]

    # initialize an attacker and perform the attack on averaged gradient
    attacker = Attacker(config, criterion)
    attacker.init_attacker_models(config)

    attacker_grad = [grad*(config.fed_lr/config.lr_guess) for grad in avg_grad] # Scaling if attacker's learning rate guess is wrong
    recon_average = grad_inv(attacker, attacker_grad, x, onehot.to(config.device), model, config, logger)
    
    fedalg.update_model(avg_grad=avg_grad)

    # Postprocess (Direct)
    if config.postprocessor == 'diffusion':
        print("Diffusion Postprocessing")
        recon_average = recon_average.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)
        save_batch(output_dir, x, recon_average, raw=True)
        postprocess(recon_average.to('cpu'), output_dir, config.device)
        ori_tensor, recon_average = load_images_to_tensors(output_dir)
        save_batch(output_dir, x, recon_average, raw=False, keyword='diffusion')

    # Compare diffusion postprocessing to ROG GAN (Yue et al., 2023)
    elif config.postprocessor == 'compare':
        print("Diffusion + GAN Postprocessing")
        synth_data, gan_data = attacker.joint_postprocess(recon_average.to(config.device), y.to(config.device)) 
        save_batch(output_dir, x, gan_data, keyword='gan', raw=False)
        recon_average = recon_average.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)
        save_batch(output_dir, x, recon_average, raw=True)
        postprocess(recon_average.to('cpu'), output_dir, config.device)
        ori_tensor, recon_average = load_images_to_tensors(output_dir)
        save_batch(output_dir, x, recon_average, raw=False, keyword='diffusion')

    elif config.postprocessor == 'none':
        print("No Postprocessing")
        recon_average = recon_average.repeat_interleave(config.sf, dim=2).repeat_interleave(config.sf, dim=3)
        save_batch(output_dir, x, recon_average, raw=True)

    # Report the result 
    logger.info("=== Evaluate the performance ====")
    metrics = Metrics(config)
    snr, ssim, jaccard, lpips = metrics.evaluate(x.to(config.device), recon_average.to(config.device), logger)
    
    logger.info("PSNR: {:.3f} SSIM: {:.3f} Jaccard {:.3f} Lpips {:.3f}".format(snr, ssim, jaccard, lpips))

    record = {"snr":snr, "ssim":ssim, "jaccard":jaccard, "lpips":lpips}

    with open(os.path.join(output_dir, config.fedalg+".dat"), "wb") as fp:
        pickle.dump(record, fp)

    return output_dir, record

if __name__ == '__main__':
    main("config.yaml")
