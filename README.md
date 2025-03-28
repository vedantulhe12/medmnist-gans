# GAN Loss Function Comparison

This project compares three Generative Adversarial Network (GAN) loss functions—LS-GAN, WGAN, and WGAN-GP—using the **MedMNIST** dataset. The models are evaluated based on **Inception Score (IS)** and **Fréchet Inception Distance (FID)** to measure the quality of generated images.

## 📌 Project Overview
- Train and compare LS-GAN, WGAN, and WGAN-GP.
- Evaluate image generation performance using IS and FID.
- Visualize training progress using TensorBoard.
- Implement memory-efficient training strategies.

## 📂 Dataset
- **MedMNIST** dataset, which contains medical images for generative tasks.
- The dataset is preprocessed and normalized before training.

## 🛠️ Technologies Used
- **Python**
- **PyTorch**
- **TorchMetrics** (for IS and FID computation)
- **TensorBoard** (for visualization)
- **Flask** (for potential deployment)

## 🚀 Model Training
Each GAN model is trained for at least **50 epochs** with the following hyperparameters:
- **Batch size**: 64
- **Learning rate**: 0.0002
- **Optimizer**: Adam (with weight clipping for WGAN)
- **Gradient penalty** (for WGAN-GP)

## 📊 Results
| Model   | Inception Score (IS) | FID Score |
|---------|---------------------|-----------|
| LS-GAN  | 1.71                | 344.26    |
| WGAN    | 2.02                | 337.78    |
| WGAN-GP | 1.88                | 339.98    |

**Interpretation:**
- **WGAN** achieves the best performance but all models need improvement.
- Low IS values suggest poor diversity and realism.
- High FID values indicate a significant gap between generated and real images.

## 🔧 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/samisafk/GAN-Mednist.git
   cd gan-loss-comparison
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train a GAN model:
   ```bash
   python train.py --model wgan
   ```
4. Compute IS and FID:
   ```bash
   python evaluate.py --model wgan
   ```
5. Visualize results using TensorBoard:
   ```bash
   tensorboard --logdir=runs
   ```

## 🔥 Future Improvements
- Increase training epochs for better results.
- Optimize hyperparameters and model architecture.
- Experiment with StyleGAN or BigGAN for enhanced performance.


---

💡 **Contributions are welcome!** Feel free to fork, open issues, or submit pull requests. 🚀

