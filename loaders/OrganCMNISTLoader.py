import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2

class OrganCMNISTDataset(Dataset):
    """Dataset ultra-ringan: Tanpa PIL & Tanpa CPU Resize."""
    def __init__(self, npz_path, split='train'):
        data = np.load(npz_path)
        raw_images = data[f'{split}_images']  # Shape uint8: (N, 28, 28) atau (N, 28, 28, 1)
        labels = data[f'{split}_labels'].squeeze()

        # Konversi awal ke Torch Tensor Float32 [0.0, 1.0] di CPU (N, 1, H, W)
        images_tensor = torch.from_numpy(raw_images).float() / 255.0
        if images_tensor.ndim == 3:
            images_tensor = images_tensor.unsqueeze(1)
        elif images_tensor.ndim == 4 and images_tensor.shape[3] == 1:
            images_tensor = images_tensor.permute(0, 3, 1, 2)

        self.images = images_tensor
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_organcmnist_loaders(
    npz_path="data/organcmnist_224.npz",
    batch_size=16,
    num_workers=2,
    img_size=224
):
    train_dataset = OrganCMNISTDataset(npz_path, split='train')
    val_dataset   = OrganCMNISTDataset(npz_path, split='val')
    test_dataset  = OrganCMNISTDataset(npz_path, split='test')

    # DataLoader CPU hanya bertugas mengoper Tensor 28x28
    use_persistent = num_workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, persistent_workers=use_persistent
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True, persistent_workers=use_persistent
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True, persistent_workers=use_persistent
    )

    # Transformasi GPU (torchvision v2)
    gpu_train_transform = v2.Compose([
        v2.Resize((img_size, img_size), antialias=True),
        v2.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
    ])

    num_classes = 11

    return train_loader, val_loader, test_loader, num_classes, gpu_train_transform


if __name__ == "__main__":
    train_loader, val_loader, test_loader, num_classes, _ = get_organcmnist_loaders()
    print("✅ Modul DataLoader OrganCMNIST Berhasil Dijalankan!")
    print(f"🏷️ Total Kelas     : {num_classes}")
    print(f"📦 Train Samples   : {len(train_loader.dataset)} ({len(train_loader)} batches)") # type: ignore
    print(f"📦 Val Samples     : {len(val_loader.dataset)} ({len(val_loader)} batches)") # type: ignore
    print(f"📦 Test Samples    : {len(test_loader.dataset)} ({len(test_loader)} batches)") # type: ignore