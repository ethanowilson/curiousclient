# Federated Learning Nodes Can Reconstruct Peers' Image Data

This repository contains the implementation of, **"Federated Learning Nodes Can Reconstruct Peers' Image Data"**. 

# Abstract
Federated learning (FL) is a privacy-preserving machine learning framework that enables multiple nodes to train models on their local data and periodically average weight updates to benefit from other nodes' training. Each node's goal is to collaborate with other nodes to improve the model's performance while keeping its training data private. However, this framework does not guarantee data privacy. Prior work has shown that the gradient-sharing steps in FL can be vulnerable to data reconstruction attacks from an honest-but-curious central server. In this work, we show that an honest-but-curious node/client can also launch attacks to reconstruct peers' image data in a centralized system, presenting a severe privacy risk. We demonstrate that a single client can silently reconstruct other clients' private images using diluted information available within consecutive updates. We leverage state-of-the-art diffusion models to enhance the perceptual quality and recognizability of the reconstructed images, further demonstrating the risk of information leakage at a semantic level. This highlights the need for more robust privacy-preserving mechanisms that protect against silent client-side attacks during federated training.

## Setup Instructions

1. **Clone the repository**:  
   `git clone <repository-link>`  
   `cd <repository-directory>`  

2. **Add Pretrained Models**:  
   Download the pretrained models from [this link](<link>) and place them in the `model_zoos/` folder.

3. **Download ImageNet Validation Set**:  
   Visit the [ImageNet download page](https://www.image-net.org/download.php) to obtain the validation set. Place the raw images in the `data/val/` folder:


4. **Install Required Packages**:  
Ensure you have Python 3.8.x installed and install the dependencies:  
`pip install -r requirements.txt`

## How to Run

### Run the Attack Framework
To execute the attack against a system four clients with direct postprocessing, use the following command from the main directory:  
`python main.py`  
The reconstructed images and attack log will be saved in the experiments folder.

## Configuration

The attack configuration can be adjusted by editing the `utils/config.yaml` file. This file allows you to modify:
- **Number of clients**  
- **Postprocessing method**  
- **Client batch size**  
- **Local epochs**
- **Global learning rate**
- **Attacker learning rate**
- **Global model**


### Semantic Postprocessing
To apply semantic postprocessing on a sample reconstructed image, run:  
`python mde_postprocess.py`  
The postprocessed images will be saved in the following folder:  
`MDE/mde_reconstructions/`

### Changing the Target Image
To change the target image for the semantic postprocessor, modify the image path and class label in `mde_postprocess.py`. Use the [ImageNet class label mapping](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/) to update the label.

For questions or issues, please open an issue in this repository.
