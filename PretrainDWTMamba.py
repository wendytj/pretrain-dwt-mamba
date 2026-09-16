# train.py
import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 1. Inisialisasi Patch Triton/Mamba Terlebih Dahulu
from architectures.patch import apply_mamba_patch
apply_mamba_patch()

from loaders.OrganCMNISTLoader import get_organcmnist_loaders
from architectures.DWTMamba import DWTMamba
from logger import ExperimentLogger
from tqdm import tqdm


CONFIG = {
    "experiment_code": "exp-organcmnist-dwtmamba-v1",
    "seed": 42,
    
    "npz_path": "data/organcmnist_224.npz",
    "batch_size": 16,
    "accumulation_steps": 1,
    "num_workers": 0,
    "img_size": 224,
    
    # Arsitektur DWT-Mamba
    "in_channels": 1,
    "embed_dim": 768,
    "depth": 4,
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
    "se_reduction": 16,
    "mb_gsf_reduction": 4,
    "latent_dim": 128,
    "proj_dim": 256,
    
    "optimizer": "Adam",
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "max_epochs": 1,
    "early_stop_patience": 10,
    "early_stop_delta": 0.001,
}


def run_mock_test(model, train_loader, device):
    """Pengujian simulasi (Dry-Run) 1 batch untuk memastikan kelancaran VRAM dan shape."""
    print("\n🔍 Memulai Mock Test (Dry-Run 1 Batch)...")
    model.eval()
    try:
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            
        assert outputs.shape == (images.size(0), CONFIG["num_classes"]), "Shape output tidak sesuai!"
        print(f"✅ Mock Test Berhasil!")
        print(f"   - Input Shape  : {images.shape}")
        print(f"   - Output Shape : {outputs.shape}")
        print(f"   - Memory VRAM  : {torch.cuda.memory_allocated() / 1e6:.2f} MB\n")
    except Exception as e:
        print(f"❌ Mock Test Gagal: {e}")
        raise e

from tqdm import tqdm  # Tambahkan import ini di paling atas file

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, gpu_transform, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for i, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            images = gpu_transform(images)
        
        with torch.amp.autocast("cuda"):  # type: ignore
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        current_loss = loss.item() * accumulation_steps
        running_loss += current_loss * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "acc": f"{(correct / total):.4f}"
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"): #type: ignore
                outputs = model(images)
                loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            running_loss += loss.item() * images.size(0)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_targets.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_probs = torch.cat(all_probs).numpy()

    gc.collect()
    torch.cuda.empty_cache()

    return epoch_loss, epoch_acc, (y_true, y_pred, y_probs)


def main():
    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Memulai Eksekusi pada Perangkat: {device}")

    # 1. Load Data
    train_loader, val_loader, test_loader, num_classes, gpu_train_transform = get_organcmnist_loaders(
        npz_path=CONFIG["npz_path"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        img_size=CONFIG["img_size"]
    )
    CONFIG["num_classes"] = num_classes

    class_names = ['bladder', 'femur-left', 'femur-right', 'heart', 'kidney-left', 'kidney-right', 'liver', 'lung-left', 'lung-right', 'pancreas', 'spleen']

    # 2. Inisialisasi Logger
    logger = ExperimentLogger(
        save_dir=f"./logs/{CONFIG['experiment_code']}",
        class_names=class_names,
        hparams=CONFIG # type: ignore
    )

    # 3. Inisialisasi Model DWT-Mamba
    model = DWTMamba(
        in_channels=CONFIG["in_channels"],
        num_classes=CONFIG["num_classes"],
        embed_dim=CONFIG["embed_dim"],
        depth=CONFIG["depth"],
        mamba_d_state=CONFIG["mamba_d_state"],
        mamba_d_conv=CONFIG["mamba_d_conv"],
        mamba_expand=CONFIG["mamba_expand"],
        se_reduction=CONFIG["se_reduction"],
        mb_gsf_reduction=CONFIG["mb_gsf_reduction"],
        latent_dim=CONFIG["latent_dim"],
        proj_dim=CONFIG["proj_dim"]
    ).to(device)

    # 4. Mock Test Sebelum Training
    run_mock_test(model, train_loader, device)

    gc.collect()
    torch.cuda.empty_cache()

    # 5. Setup Optimizer & Loss Function (Sesuai Paper)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )

    scaler = torch.amp.GradScaler("cuda") #type: ignore

    # Variable Early Stopping & Model Checkpoint
    best_val_loss = float("inf")
    patience_counter = 0
    best_eval_tuples = None

    print("🔥 Memulai Pelatihan Aktif...")
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, gpu_train_transform, accumulation_steps=CONFIG["accumulation_steps"])
        val_loss, val_acc, val_eval = evaluate(model, val_loader, criterion, device)
        
        # Hitung F1-score sementara untuk logging
        _, val_metrics = logger.compute_metrics(*val_eval)
        val_f1 = val_metrics["global_metrics"]["f1_score_macro"]

        # Log per epoch
        logger.log_epoch(epoch, train_loss, val_loss, train_acc, val_acc, val_f1)
        print(f"Epoch [{epoch:03d}/{CONFIG['max_epochs']}] | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f} - F1: {val_f1:.4f}")

        # Checkpoint Best Model & Early Stopping Logic
        if val_loss < (best_val_loss - CONFIG["early_stop_delta"]):
            best_val_loss = val_loss
            patience_counter = 0
            best_eval_tuples = {
                "train": evaluate(model, train_loader, criterion, device)[2],
                "val": val_eval
            }
            # Simpan Weights Terbaik
            torch.save(model.state_dict(), os.path.join(logger.save_dir, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stop_patience"]:
                print(f"🛑 Early stopping dipicu pada epoch {epoch}.")
                break

        gc.collect()
        torch.cuda.empty_cache()

    # 6. Final Evaluation pada Test Set Menggunakan Model Terbaik
    print("\n🧪 Mengevaluasi Performance pada Test Set (Best Model)...")
    model.load_state_dict(torch.load(os.path.join(logger.save_dir, "best_model.pth")))
    _, _, test_eval = evaluate(model, test_loader, criterion, device)

    # 7. Ekspor Hasil Akhir Via Logger
    logger.export_csv()
    logger.plot_learning_curves()
    logger.save_best_results_json(
        train_eval=best_eval_tuples["train"], # type: ignore
        val_eval=best_eval_tuples["val"], # type: ignore
        test_eval=test_eval
    )
    print("✨ Eksperimen Selesai dan Seluruh Log Berhasil Disimpan!")

if __name__ == "__main__":
    main()