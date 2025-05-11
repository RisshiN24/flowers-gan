import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image
from gan import Generator, Discriminator, load_flower_dataset

# =========================
# Config
# =========================

def get_config():
    return {
        'lr': 2e-4,
        'batch_size': 64,
        'z_dim': 100,
        'label_dim': 102,
        'img_size': 64,
        'num_epochs': 100,
        'save_dir': "generated",
        'checkpoint_dir': "checkpoints",
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

# =========================
# One-hot encoding helper
# =========================

def one_hot(labels, num_classes):
    return F.one_hot(labels.long(), num_classes).float()

# =========================
# Training function
# =========================

def train():
    cfg = get_config()
    os.makedirs(cfg['save_dir'], exist_ok=True)
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)

    dataloader = load_flower_dataset(
        image_size=cfg['img_size'],
        batch_size=cfg['batch_size']
    )

    G = Generator(cfg['z_dim'], cfg['label_dim']).to(cfg['device'])
    D = Discriminator(cfg['label_dim']).to(cfg['device'])

    opt_G = optim.Adam(G.parameters(), lr=cfg['lr'], betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=cfg['lr'], betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(16, cfg['z_dim'], device=cfg['device']) # 16x100
    fixed_labels = torch.randint(0, cfg['label_dim'], (16,), device=cfg['device']) # 16
    fixed_one_hot = one_hot(fixed_labels, cfg['label_dim']).to(cfg['device']) # 16x102

    # Training loop
    for epoch in range(cfg['num_epochs']):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['num_epochs']}")

        for real_imgs, labels in loop:
            real_imgs = real_imgs.to(cfg['device'])
            labels = labels.to(cfg['device'])
            one_hot_labels = one_hot(labels, cfg['label_dim']).to(cfg['device'])
            batch_size = real_imgs.size(0)

            # === Train Discriminator ===
            z = torch.randn(batch_size, cfg['z_dim'], device=cfg['device'])
            fake_imgs = G(z, one_hot_labels)

            real_targets = torch.ones(batch_size, device=cfg['device'])
            fake_targets = torch.zeros(batch_size, device=cfg['device'])

            D_real = D(real_imgs, one_hot_labels)
            D_fake = D(fake_imgs.detach(), one_hot_labels)

            loss_D = criterion(D_real, real_targets) + criterion(D_fake, fake_targets)
            D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # === Train Generator ===
            z = torch.randn(batch_size, cfg['z_dim'], device=cfg['device'])
            fake_imgs = G(z, one_hot_labels)
            D_fake = D(fake_imgs, one_hot_labels)

            loss_G = criterion(D_fake, real_targets)
            G.zero_grad()
            loss_G.backward()
            opt_G.step()

            loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

        # Save sample generations
        with torch.no_grad():
            sample_imgs = G(fixed_noise, fixed_one_hot)
            save_image(sample_imgs * 0.5 + 0.5, f"{cfg['save_dir']}/epoch_{epoch+1:03d}.png", nrow=4)

        # Save model checkpoints
        torch.save(G.state_dict(), f"{cfg['checkpoint_dir']}/G_epoch_{epoch+1:03d}.pth")
        torch.save(D.state_dict(), f"{cfg['checkpoint_dir']}/D_epoch_{epoch+1:03d}.pth")

# =========================
# Main
# =========================

if __name__ == "__main__":
    train()