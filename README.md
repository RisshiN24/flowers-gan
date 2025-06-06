# 🌸 Conditional GAN for Flower Image Generation

This project implements a conditional generative adversarial network (cGAN) using PyTorch to generate realistic flower images conditioned on class labels. Leveraging the Oxford 102 Flowers dataset, the model learns to generate 64×64 RGB images that resemble specific flower categories when provided with a corresponding one-hot encoded label. The generator creates images from random noise and label embeddings, while the discriminator evaluates whether those images look realistic and match the given class.

The primary goal of this project is to explore class-conditional image generation and understand the dynamics of GAN training, particularly how label conditioning affects output diversity and visual fidelity. Beyond serving as a learning project in deep generative modeling, this work can be extended to more complex datasets and architectures (e.g., attention-based GANs or diffusion models) to further improve image quality and label-text alignment.

For future development, strategies will be implemented to prevent potential mode collapse and vanishing gradients.

Results from Epoch 37: ![Generated Image from Epoch 37](src/generated/epoch_037.png)