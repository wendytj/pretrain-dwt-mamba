import os
import argparse
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch._dynamo
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch._dynamo.config.suppress_errors = True # type: ignore

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
    "batch_size": 64,
    "eval_batch_size": 256,
    "accumulation_steps": 1,
    "num_workers": 12,
    "img_size": 224,
    
    # Arsitektur DWT-Mamba
    "in_channels": 1,
    "embed_dim": 256,
    "depth": 3,
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
    "max_epochs": 6,
    "warmup_epochs": 5,
    "early_stop_patience": 10,
    "early_stop_delta": 0.001,
}


def run_mock_test(model, train_loader, device, gpu_transform):
    """Pengujian simulasi (Dry-Run) 1 batch untuk memastikan kelancaran VRAM dan shape."""
    print("\n🔍 Memulai Mock Test (Dry-Run 1 Batch)...")
    model.eval()
    try:
        images, labels = next(iter(train_loader))
        images = images.to(device, non_blocking=True)
        # Squeeze dimensi (B, 1) -> (B,) dan konversi ke long
        labels = labels.to(device, non_blocking=True).view(-1).long()

        use_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            images = gpu_transform(images)
            images = images.to(memory_format=torch.channels_last)
            outputs = model(images)
                    
        assert outputs.shape == (images.size(0), CONFIG["num_classes"]), "Shape output tidak sesuai!"
        print(f"✅ Mock Test Berhasil!")
        print(f"    - Input Shape  : {images.shape}")
        print(f"    - Output Shape : {outputs.shape}")
        print(f"    - Memory VRAM  : {torch.cuda.memory_allocated() / 1e6:.2f} MB\n")
    except Exception as e:
        print(f"❌ Mock Test Gagal: {e}")
        raise e


def train_one_epoch(model, dataloader, criterion, optimizer, device, gpu_transform, amp_dtype, use_bf16, scaler, accumulation_steps=1):
    model.train()
    running_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = 0

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, desc="Training", leave=False)

    for i, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        # Squeeze dimensi (B, 1) -> (B,) dan konversi ke long
        labels = labels.to(device, non_blocking=True).view(-1).long()

        with torch.no_grad():
            images = gpu_transform(images)
        
        # Memastikan format memori channels_last tetap terjaga setelah augmentasi
        images = images.to(memory_format=torch.channels_last)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_scaled = loss / accumulation_steps

        if use_bf16:
            loss_scaled.backward()
        else:
            scaler.scale(loss_scaled).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            if use_bf16:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # Akumulasi statistik murni di GPU tanpa sync CPU (.item())
        batch_size = images.size(0)
        running_loss += loss.detach() * batch_size
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        total += batch_size

        # Update tqdm setiap 10 iterasi untuk memangkas overhead I/O CPU
        if i % 10 == 0:
            pbar.set_postfix({
                "loss": f"{(running_loss / total).item():.4f}",
                "acc": f"{(correct.float() / total).item():.4f}"
            })

    # Konversi statistik GPU ke CPU hanya 1x di akhir epoch
    epoch_loss = (running_loss / total).item()
    epoch_acc = (correct.float() / total).item()
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device, gpu_transform):
    model.eval()
    running_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = 0

    all_targets = []
    all_preds = []
    all_probs = []

    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            # Squeeze dimensi (B, 1) -> (B,) dan konversi ke long
            labels = labels.to(device, non_blocking=True).view(-1).long()

            images = gpu_transform(images)
            images = images.to(memory_format=torch.channels_last) # Pastikan channels_last aktif

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Softmax dieksekusi dalam presisi FP32 agar akurasi probabilitas stabil
            probs = torch.softmax(outputs.float(), dim=1)
            _, preds = outputs.max(1)

            # Akumulasi tensor murni di GPU tanpa .item()
            batch_size = images.size(0)
            running_loss += loss.detach() * batch_size
            correct += preds.eq(labels).sum()
            total += batch_size

            all_targets.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    # Konversi statistik ke CPU 1x saja di akhir
    epoch_loss = (running_loss / total).item()
    epoch_acc = (correct.float() / total).item()
    
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_probs = torch.cat(all_probs).numpy()

    return epoch_loss, epoch_acc, (y_true, y_pred, y_probs)

def run_training_pipeline(model, raw_model, loaders, transforms, amp_params, logger, device):
    """Fungsi modul khusus untuk menangani siklus pelatihan penuh, validation, dan final testing."""
    train_loader, val_loader, test_loader = loaders
    gpu_train_transform, gpu_eval_transform = transforms
    amp_dtype, use_bf16, scaler = amp_params

    criterion = nn.CrossEntropyLoss()

    base_lr = CONFIG["learning_rate"]
    batch_size = CONFIG["batch_size"]
    scaled_lr = base_lr * (batch_size / 16) ** 0.5

    optimizer = optim.Adam(
        model.parameters(),
        weight_decay=CONFIG["weight_decay"],
        lr = scaled_lr,
        eps=1e-7
    )

    scheduler_warmup = LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=CONFIG["warmup_epochs"]
    )

    scheduler_decay = CosineAnnealingLR(
        optimizer, 
        T_max=CONFIG["max_epochs"] - CONFIG["warmup_epochs"], 
        eta_min=1e-6
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_decay],
        milestones=[CONFIG["warmup_epochs"]]
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_eval_tuples_val = None

    print("🔥 Memulai Pelatihan Aktif...")
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            gpu_train_transform, amp_dtype, use_bf16, scaler, 
            accumulation_steps=CONFIG["accumulation_steps"]
        )
        val_loss, val_acc, val_eval = evaluate(
            model, val_loader, criterion, device, gpu_eval_transform
        )

        scheduler.step()
        
        val_metrics, _ = logger.compute_metrics(*val_eval)
        val_f1 = val_metrics["global_metrics"]["f1_score_macro"]
        current_lr = optimizer.param_groups[0]['lr']

        logger.log_epoch(epoch, train_loss, val_loss, train_acc, val_acc, val_f1)
        print(f"Epoch [{epoch:03d}/{CONFIG['max_epochs']}] | LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f} - F1: {val_f1:.4f}")

        # Checkpoint Model Terbaik & Early Stopping
        if val_loss < (best_val_loss - CONFIG["early_stop_delta"]):
            best_val_loss = val_loss
            patience_counter = 0
            best_eval_tuples_val = val_eval
            torch.save(raw_model.state_dict(), os.path.join(logger.save_dir, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stop_patience"]:
                print(f"🛑 Early stopping dipicu pada epoch {epoch}.")
                break

        torch.save(
            {
                "epoch": epoch,
                "model_state": raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            os.path.join(logger.save_dir, "last_checkpoint.pth"),
        )

    # Final Evaluation pada Test Set
    print("\n🧪 Mengevaluasi Performance pada Test Set (Best Model)...")
    raw_model.load_state_dict(torch.load(os.path.join(logger.save_dir, "best_model.pth")))
    _, _, train_eval_best = evaluate(model, train_loader, criterion, device, gpu_eval_transform)
    _, _, test_eval = evaluate(model, test_loader, criterion, device, gpu_eval_transform)

    # Ekspor Log
    logger.export_csv()
    logger.plot_learning_curves()
    logger.save_best_results_json(
        train_eval=train_eval_best,
        val_eval=best_eval_tuples_val,
        test_eval=test_eval
    )
    print("✨ Eksperimen Selesai dan Seluruh Log Berhasil Disimpan!")

def main():
    args = parse_args()

    CONFIG["batch_size"] = args.batch_size
    CONFIG["accumulation_steps"] = args.accumulation_steps
    CONFIG["num_workers"] = args.num_workers
    CONFIG["max_epochs"] = args.max_epochs
    CONFIG["learning_rate"] = args.learning_rate
    CONFIG["weight_decay"] = args.weight_decay
    CONFIG["experiment_code"] = args.experiment_code
    CONFIG["warmup_epochs"] = args.warmup_epochs
    CONFIG["eval_batch_size"] = args.eval_batch_size

    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Memulai Eksekusi pada Perangkat: {device}")

    # Setup AMP & Scaler
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16) # type: ignore

    # 1. Load Data
    train_loader, val_loader, test_loader, num_classes, gpu_train_transform, gpu_eval_transform = get_organcmnist_loaders(
        npz_path=CONFIG["npz_path"],
        batch_size=CONFIG["batch_size"],
        eval_batch_size=CONFIG["eval_batch_size"],
        num_workers=CONFIG["num_workers"],
        img_size=CONFIG["img_size"]
    )
    
    CONFIG["num_classes"] = num_classes
    class_names = ['bladder', 'femur-left', 'femur-right', 'heart', 'kidney-left', 'kidney-right', 'liver', 'lung-left', 'lung-right', 'pancreas', 'spleen']

    # 2. Inisialisasi Logger
    logger = ExperimentLogger(
        save_dir=f"./logs/{CONFIG['experiment_code']}",
        class_names=class_names,
        hparams=CONFIG
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
    ).to(device).to(memory_format=torch.channels_last) # type: ignore

    raw_model = model

    # 4. Mode Percabangan: Dry-Run vs Full Training

    run_mock_test(model, train_loader, device, gpu_eval_transform,)

    if args.dry_run:
        print("💡 Mode --dry_run selesai. Program keluar tanpa melakukan pelatihan.")
        return

    gc.collect()
    torch.cuda.empty_cache()

    # 5. Jalankan Training Pipeline Penuh
    loaders = (train_loader, val_loader, test_loader)
    transforms = (gpu_train_transform, gpu_eval_transform)
    amp_params = (amp_dtype, use_bf16, scaler)

    run_training_pipeline(model, raw_model, loaders, transforms, amp_params, logger, device)

def parse_args():
    parser = argparse.ArgumentParser(description="DWT-Mamba Modular Training & Dry-Run Pipeline")
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"], help="Batch size per iterasi GPU")
    parser.add_argument("--accumulation_steps", type=int, default=CONFIG["accumulation_steps"], help="Langkah akumulasi gradien")
    parser.add_argument("--num_workers", type=int, default=CONFIG["num_workers"], help="Jumlah worker DataLoader")
    parser.add_argument("--max_epochs", type=int, default=CONFIG["max_epochs"], help="Jumlah maksimum epoch pelatihan")
    parser.add_argument("--warmup_epochs", type=int, default=CONFIG["warmup_epochs"], help="Jumlah epoch LR warmup")
    parser.add_argument("--learning_rate", type=float, default=CONFIG["learning_rate"], help="Learning rate Adam")
    parser.add_argument("--weight_decay", type=float, default=CONFIG["weight_decay"], help="Weight decay Adam")
    parser.add_argument("--experiment_code", type=str, default=CONFIG["experiment_code"], help="Kode/Folder eksperimen")
    parser.add_argument("--dry_run", action="store_true", help="Eksekusi mock test 1 batch lalu keluar tanpa training")
    parser.add_argument("--eval_batch_size", type=int, default=CONFIG["eval_batch_size"], help="Batch size evaluasi")
    return parser.parse_args()

if __name__ == "__main__":
    main()