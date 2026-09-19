import os
import json
import numpy as np
import pandas as pd

# Wajib dipanggil sebelum import pyplot agar aman di server headless/non-GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore

from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef
)

class ExperimentLogger:
    def __init__(self, save_dir="./logs", class_names=["Bleeding", "Ischemia", "Normal"], hparams=None):
        self.save_dir = save_dir
        self.class_names = class_names
        self.hparams = hparams or {}
        os.makedirs(save_dir, exist_ok=True)

        self.history = {
            'epoch': [], 'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [], 'val_f1': []
        }
        
        if self.hparams:
            self.save_hparams()

    def save_hparams(self, filename="hparams.json"):
        """Menyimpan konfigurasi hyperparameter ke JSON dengan penanganan tipe data aman."""
        path = os.path.join(self.save_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.hparams, f, indent=4, default=str)
        print(f"⚙️ Hyperparameters disimpan di: {path}")

    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, val_f1):
        """Catat riwayat pelatihan per epoch."""
        self.history['epoch'].append(int(epoch))
        self.history['train_loss'].append(float(train_loss))
        self.history['val_loss'].append(float(val_loss))
        self.history['train_acc'].append(float(train_acc))
        self.history['val_acc'].append(float(val_acc))
        self.history['val_f1'].append(float(val_f1))

    def export_csv(self, filename="training_history.csv"):
        """Ekspor riwayat epoch ke CSV."""
        df = pd.DataFrame(self.history)
        path = os.path.join(self.save_dir, filename)
        df.to_csv(path, index=False)
        print(f"💾 File rekapitulasi CSV disimpan di: {path}")
        return path

    def compute_metrics(self, y_true, y_pred, y_probs):
        """Hitung Confusion Matrix NxN dan seluruh metrik turunan secara presisi."""
        labels_idx = list(range(len(self.class_names)))
        
        cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # ROC-AUC OvR dengan penanganan penentuan label eksplisit
        try:
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro', labels=labels_idx)
        except Exception:
            auc = 0.0

        kappa = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        prec_per_class = precision_score(y_true, y_pred, labels=labels_idx, average=None, zero_division=0)
        rec_per_class = recall_score(y_true, y_pred, labels=labels_idx, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, labels=labels_idx, average=None, zero_division=0)

        metrics = {
            "confusion_matrix": cm.tolist(),
            "global_metrics": {
                "accuracy": round(float(acc), 4),
                "precision_macro": round(float(prec), 4),
                "recall_macro": round(float(rec), 4),
                "f1_score_macro": round(float(f1), 4),
                "auc_roc_ovr": round(float(auc), 4),
                "cohens_kappa": round(float(kappa), 4),
                "mcc": round(float(mcc), 4)
            },
            "per_class_metrics": {
                name: {
                    "precision": round(float(prec_per_class[i]), 4), # type: ignore
                    "recall": round(float(rec_per_class[i]), 4), # type: ignore
                    "f1_score": round(float(f1_per_class[i]), 4) # type: ignore
                } for i, name in enumerate(self.class_names)
            }
        }
        return metrics, cm

    def save_best_results_json(self, train_eval, val_eval, test_eval, filename="best_model_metrics.json"):
        """Menyimpan snapshot performa terbaik ke format JSON."""
        train_metrics, _ = self.compute_metrics(*train_eval)
        val_metrics, _ = self.compute_metrics(*val_eval)
        test_metrics, cm_test = self.compute_metrics(*test_eval)

        summary_json = {
            "dataset_info": {
                "num_classes": len(self.class_names),
                "class_labels": self.class_names
            },
            "hyperparameters": self.hparams,
            "best_train_results": train_metrics,
            "best_val_results": val_metrics,
            "best_test_results": test_metrics
        }

        json_path = os.path.join(self.save_dir, filename)
        with open(json_path, 'w') as f:
            json.dump(summary_json, f, indent=4, default=str)

        print(f"File snapshot JSON (Train/Val/Test) disimpan di: {json_path}")
        self.plot_confusion_matrix(cm_test, filename="test_confusion_matrix.png")
        return summary_json

    def plot_confusion_matrix(self, cm, filename="confusion_matrix.png"):
        """Visualisasi Confusion Matrix NxN dengan pengaturan font dinamis agar tidak berhimpitan."""
        n_classes = len(self.class_names)
        fig_size = max(7, int(n_classes * 0.8))
        font_size = max(6, 12 - int(n_classes * 0.4))
        
        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names, yticklabels=self.class_names,
            annot_kws={"size": font_size}
        )
        plt.title(f'Confusion Matrix {n_classes}x{n_classes} (Test Set)', fontweight='bold')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()

    def plot_learning_curves(self, filename="learning_curves.png"):
        """Visualisasi grafik loss dan akurasi per epoch."""
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(self.history['epoch'], self.history['train_loss'], label='Train Loss', color='crimson', linewidth=2)
        axes[0].plot(self.history['epoch'], self.history['val_loss'], label='Val Loss', color='navy', linestyle='--', linewidth=2)
        axes[0].set_title('Loss Curve per Epoch', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        axes[1].plot(self.history['epoch'], self.history['train_acc'], label='Train Acc', color='forestgreen', linewidth=2)
        axes[1].plot(self.history['epoch'], self.history['val_acc'], label='Val Acc', color='darkorange', linestyle='--', linewidth=2)
        axes[1].set_title('Accuracy Curve per Epoch', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()

        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()

if __name__ == "__main__":

    code_experiment = "exp-1"
    epochs = 10

    logger = ExperimentLogger(save_dir=f"./{code_experiment}")

    # 1. Di dalam loop epoch (selama training)
    for epoch in range(1, epochs + 1):
        # ... jalankan train & val ...
        # misal dummy
        train_loss = val_loss = train_acc = val_acc = val_f1 = 0
        logger.log_epoch(epoch, train_loss, val_loss, train_acc, val_acc, val_f1)

    # Ekspor CSV history dan plot grafik pelatihan
    logger.export_csv()
    logger.plot_learning_curves()

    np.random.seed(42)

    def generate_dummy_eval_data(n_samples=500):
        """
        Menghasilkan dummy y_true, y_pred, dan y_probs untuk 3 kelas:
        0: Bleeding, 1: Ischemia, 2: Normal
        """
        y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.2, 0.2, 0.6])

        raw_logits = np.random.randn(n_samples, 3)
        for i in range(n_samples):
            raw_logits[i, y_true[i]] += np.random.uniform(1.5, 3.0)

        exp_logits = np.exp(raw_logits - np.max(raw_logits, axis=1, keepdims=True))
        y_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        y_pred = np.argmax(y_probs, axis=1)

        return y_true, y_pred, y_probs

    y_train_true, y_train_pred, y_train_probs = generate_dummy_eval_data(n_samples=1000)
    y_val_true, y_val_pred, y_val_probs       = generate_dummy_eval_data(n_samples=200)
    y_test_true, y_test_pred, y_test_probs     = generate_dummy_eval_data(n_samples=200)

    train_data = (y_train_true, y_train_pred, y_train_probs)
    val_data   = (y_val_true, y_val_pred, y_val_probs)
    test_data  = (y_test_true, y_test_pred, y_test_probs)

    logger.save_best_results_json(train_data, val_data, test_data)