# GAN-Based Medical Image Generation using MedMNIST

This project explores Generative Adversarial Networks (GANs) for medical image generation using the ChestMNIST dataset from MedMNIST. Three GAN variants are implemented and compared: **LS-GAN, WGAN, and WGAN-GP**.

## Project Overview
- **Dataset**: ChestMNIST (28x28 grayscale medical images)
- **GAN Variants**: LS-GAN, WGAN, WGAN-GP
- **Evaluation Metrics**:
  - **Inception Score (IS)**: Measures diversity and quality of generated images.
  - **Fréchet Inception Distance (FID)**: Measures similarity between real and generated images.

## Technologies Used
- **Python**
- **PyTorch**
- **TorchMetrics** (for IS and FID computation)
- **TensorBoard** (for visualization)
- **Flask** (for potential deployment)

## Model Training
Each GAN model is trained for at least 50 epochs with the following hyperparameters:
- **Batch size**: 64
- **Learning rate**: 0.0002
- **Optimizer**: Adam (with weight clipping for WGAN)
- **Gradient penalty** (for WGAN-GP)

## Results
### Generated Images
LS-GAN:
![LS-GAN](https://github.com/samisafk/GAN-Mednist/blob/main/generated/LS-GAN_epoch_49.png)

WGAN:
![WGAN](https://github.com/samisafk/GAN-Mednist/blob/main/generated/WGAN_epoch_49.png)

WGAN-GP:
![WGAN-GP](https://github.com/samisafk/GAN-Mednist/blob/main/generated/WGAN-GP_epoch_49.png)

### Quantitative Evaluation
| Model   | Inception Score | FID  |
|---------|----------------|------|
| LS-GAN  | 1.71           | 344.27 |
| WGAN    | 2.02           | 337.78 |
| WGAN-GP | 1.89           | 339.99 |

**Interpretation**:
- **WGAN** achieves the best performance but all models need improvement.
- Low IS values suggest poor diversity and realism.
- High FID values indicate a significant gap between generated and real images.

## Setup and Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- PyTorch
- torchvision
- torchmetrics
- medmnist
- tqdm
- tensorboard

### Installation
Clone the repository and install dependencies:
```bash
 git clone https://github.com/samisafk/GAN-Mednist.git
 cd GAN-Mednist
 pip install -r requirements.txt
```

## Training the GANs
Run the training script to train all three models:
```bash
python train.py
```
The trained models will be saved in the `models/` directory.

## Evaluating the Models
To compute IS and FID scores:
```bash
python evaluate.py
```

## Acknowledgments
- **MedMNIST**: A lightweight benchmark for medical image analysis
- **PyTorch**: Deep learning framework used for training
- **TorchMetrics**: Used for computing IS and FID scores

