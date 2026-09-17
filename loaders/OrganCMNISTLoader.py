import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as v2


class OrganCMNISTDataset(Dataset):
  """Dataset loader presisi untuk file NPZ pre-rendered 224x224."""

  def __init__(self, npz_path, split="train"):
    data = np.load(npz_path, allow_pickle=True)
    # Shape asli uint8: (N, 224, 224) atau (N, 224, 224, 1)
    raw_images = data[f"{split}_images"]
    labels = data[f"{split}_labels"].squeeze()

    # Simpan uint8 di RAM (~700MB total) -> Transfer PCIe super cepat ke GPU
    images_tensor = torch.from_numpy(raw_images)
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
    num_workers=4,
    img_size=224,
):
  train_dataset = OrganCMNISTDataset(npz_path, split="train")
  val_dataset = OrganCMNISTDataset(npz_path, split="val")
  test_dataset = OrganCMNISTDataset(npz_path, split="test")

  use_persistent = num_workers > 0
  dataloader_kwargs = {
      "batch_size": batch_size,
      "num_workers": num_workers,
      "pin_memory": True,
      "persistent_workers": use_persistent,
  }

  if use_persistent:
    dataloader_kwargs["prefetch_factor"] = 2

  train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
  val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
  test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)

  # 1. Transformasi GPU Training: augmentasi Crop/Flip pada gambar 224x224
  gpu_train_transform = v2.Compose([
      v2.ToDtype(torch.float32, scale=True),  # Convert uint8 -> float32 [0.0, 1.0] di GPU
      v2.RandomResizedCrop(
          (img_size, img_size), scale=(0.8, 1.0), antialias=True
      ),
      v2.RandomHorizontalFlip(p=0.5),
  ])

  # 2. Transformasi GPU Evaluation: Murni cast tipe data (TANPA RESIZE)
  gpu_eval_transform = v2.Compose([
      v2.ToDtype(torch.float32, scale=True),
  ])

  num_classes = 11

  return (
      train_loader,
      val_loader,
      test_loader,
      num_classes,
      gpu_train_transform,
      gpu_eval_transform,
  )


if __name__ == "__main__":
  train_loader, val_loader, test_loader, num_classes, _, _ = (
      get_organcmnist_loaders()
  )
  print("✅ Modul DataLoader OrganCMNIST (224x224 Native) Berhasil!")
  print(f"🏷️ Total Kelas     : {num_classes}")
  print(
      f"📦 Train Batches   : {len(train_loader)} (Samples:"
      f" {len(train_loader.dataset)})"  # type: ignore
  )