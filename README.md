# 🐱 Cat Image Generation — VAE, DCGAN & Diffusion Models

This repository explores three families of modern generative models applied to a dataset of 64×64 cat faces:

- **Variational Autoencoder (VAE)**
- **Deep Convolutional GAN (DCGAN)**
- **Denoising Diffusion Probabilistic Model (DDPM)**

Each model is implemented **from scratch** in PyTorch and trained on the same dataset to allow a fair qualitative and conceptual comparison.
---

## 📁 Project Structure
```bash
Diffusion Cats/
│
├── data/
│   └── cats_dataset/              # Preprocessed images (64×64)
│
├── models/
│   ├── vae_cats.pth               # Trained VAE
│   ├── dcgan_G_cats.pth           # Trained GAN generator
│   └── diffusion_unet_cats.pth    # Trained diffusion U-Net
│
├── notebooks/
│   ├── 01_vae.ipynb               # VAE training & sampling
│   ├── 02_dcgan.ipynb             # DCGAN training & sampling
│   ├── 03_Diffusion.ipynb         # DDPM training & sampling
│   └── 04_compare_models.ipynb    # Side-by-side model comparison
│
├── src/
│   ├── vae.py                     # VAE model implementation
│   ├── dcgan.py                   # GAN (G & D) implementation
│   ├── diffusion.py               # Diffusion model + U-Net
│   └── utils.py                   # Shared utilities (dataset, plots, etc.)
│
├── requirements.txt
└── README.md
```
---

## 📘 Dataset & Preprocessing

The dataset consists of 64×64 RGB cat faces stored under:

All models share the same preprocessing steps:

1. **Normalization to $[-1, 1]$**  
   $$[x = \frac{x - 127.5}{127.5}]$$

2. **Channel reordering**  
   $$[(H, W, C) \rightarrow (C, H, W)]$$

3. **Conversion to PyTorch tensors**

4. **Train/Validation split (90% / 10%)**

This common preprocessing ensures a fair comparison between VAE, GAN, and Diffusion models.


## 🧩 1. Variational Autoencoder (VAE)

A convolutional VAE is used as the baseline generative model.

### Notebook
`notebooks/01_vae.ipynb`

### Architecture
- CNN encoder  
- Latent sampling using reparameterization  
- CNN decoder  

### Objective

$$[\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\mathrm{KL}}(q(z|x)\,\|\,p(z))]$$

### Outputs
- Reconstructions  
- Latent space interpolation  
- Sampling from the prior  

### Checkpoint
`models/vae_cats.pth`

## ⚡ 2. Deep Convolutional GAN (DCGAN)

The DCGAN trains a generator and discriminator adversarially to produce cat images.

### Notebook
`notebooks/02_dcgan.ipynb`

### Architecture
- Generator: transposed convolutions + ReLU  
- Discriminator: convolutions + LeakyReLU  

### Loss
$$[\min_G \max_D \; \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]]$$

### Outputs
- Training curves (G/D losses)  
- Generated samples  
- Noise interpolation  

### Checkpoint
`models/dcgan_G_cats.pth`


## 🌫️ 3. Denoising Diffusion Probabilistic Model (DDPM)

The most advanced model in this repository is a full DDPM implementation including:

- Linear noise schedule $\beta_t$
- Forward diffusion:
  $$[
  x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon
  ]$$
  
- Sinusoidal time embeddings  
- U-Net with temporal conditioning  
- Reverse sampling loop  

### Notebook
`notebooks/03_Diffusion.ipynb`

### Objective
$$[
\mathcal{L} = \mathbb{E}\left[\|
\epsilon - \epsilon_\theta(x_t, t)
\|_2^2\right]
]$$

### Checkpoint
`models/diffusion_unet_cats.pth`


## 📊 4. Model Comparison

The notebook `notebooks/04_compare_models.ipynb` provides a side-by-side comparison of:

- Reconstruction quality (VAE)
- Image sharpness and diversity (GAN)
- Sample quality and stability (DDPM)

It includes qualitative grids and a brief discussion of strengths and limitations.




## 📚 References

- Kingma & Welling — *Auto-Encoding Variational Bayes*  
- Goodfellow et al. — *Generative Adversarial Networks*  
- Ho et al. — *Denoising Diffusion Probabilistic Models*  

---

## 👤 Author

Erwan Ouabdesselam  
This project explores and compares three core generative modeling paradigms.

---










