import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


DEFAULT_DATA_DIR = Path("examples/Nicla-wakeWord/dataset_numpy")
DEFAULT_OUTPUT_DIR = Path("examples/Nicla-wakeWord/training_output")
CLASSES = ["unknown", "heyfranco"]
WAKE_WORD_CLASS = CLASSES.index("heyfranco")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def label_from_filename(path):
    label = path.name.split(".", 1)[0].lower()
    if label not in CLASSES:
        raise ValueError(f"Classe non riconosciuta per {path}: {label}")
    return CLASSES.index(label)


class GafWakeWordDataset(Dataset):
    def __init__(self, split_dir):
        self.split_dir = Path(split_dir)
        self.samples = sorted(self.split_dir.glob("*.npy"))
        if not self.samples:
            raise FileNotFoundError(f"Nessun file .npy trovato in {self.split_dir}")

        self.targets = [label_from_filename(path) for path in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        array = np.load(path).astype(np.float32)

        if array.shape != (3, 64, 64):
            raise ValueError(f"Shape inattesa in {path}: {array.shape}, attesa (3, 64, 64)")

        # I tensori GAF/MTF sono gia' in [0, 1]. Li centriamo in [-1, 1].
        array = (array - 0.5) / 0.5
        return torch.from_numpy(array), self.targets[index]


def stratified_train_val_indices(targets, val_ratio, seed):
    rng = random.Random(seed)
    by_class = {}
    for index, target in enumerate(targets):
        by_class.setdefault(target, []).append(index)

    train_indices = []
    val_indices = []

    for indices in by_class.values():
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_ratio)))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


class WakeWordGafCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def class_weights(targets, device):
    counts = Counter(targets)
    weights = []
    total = len(targets)
    for class_id in range(len(CLASSES)):
        count = counts[class_id]
        if count == 0:
            raise ValueError(f"Nessun sample per la classe {CLASSES[class_id]}")
        weights.append(total / (len(CLASSES) * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(x)
            loss = criterion(logits, y)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, total_correct / total


def confusion_from_predictions(targets, preds):
    confusion = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.int64)
    for true_label, pred_label in zip(targets, preds):
        confusion[int(true_label), int(pred_label)] += 1
    return confusion


def metrics_from_confusion(confusion, class_id=WAKE_WORD_CLASS):
    tp = confusion[class_id, class_id].item()
    fp = confusion[:, class_id].sum().item() - tp
    fn = confusion[class_id, :].sum().item() - tp
    tn = confusion.sum().item() - tp - fp - fn

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "false_positives": fp,
        "true_positives": tp,
        "false_negatives": fn,
        "true_negatives": tn,
        "support": confusion[class_id, :].sum().item(),
    }


def find_best_threshold(probs, targets, min_threshold, max_threshold, step, fp_penalty):
    thresholds = np.arange(min_threshold, max_threshold + 1e-9, step)
    best = None

    for threshold in thresholds:
        preds = (probs >= threshold).astype(np.int64)
        confusion = confusion_from_predictions(targets, preds)
        metrics = metrics_from_confusion(confusion)
        score = metrics["recall"] - fp_penalty * metrics["false_positive_rate"]

        candidate = {
            "threshold": float(threshold),
            "score": score,
            "confusion": confusion,
            "metrics": metrics,
        }

        if best is None:
            best = candidate
            continue

        best_metrics = best["metrics"]
        if (
            candidate["score"] > best["score"]
            or (
                abs(candidate["score"] - best["score"]) < 1e-12
                and metrics["recall"] > best_metrics["recall"]
            )
            or (
                abs(candidate["score"] - best["score"]) < 1e-12
                and abs(metrics["recall"] - best_metrics["recall"]) < 1e-12
                and metrics["false_positives"] < best_metrics["false_positives"]
            )
        ):
            best = candidate

    return best


def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    total = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)[:, WAKE_WORD_CLASS]

            total_loss += loss.item() * x.size(0)
            total += y.size(0)
            all_probs.extend(probs.cpu().tolist())
            all_targets.extend(y.cpu().tolist())

    preds = [1 if prob >= threshold else 0 for prob in all_probs]
    confusion = confusion_from_predictions(all_targets, preds)
    accuracy = confusion.diag().sum().item() / total
    return total_loss / total, accuracy, confusion, np.array(all_probs), np.array(all_targets)


def print_report(confusion):
    print("\nClassification report:")
    for class_id, class_name in enumerate(CLASSES):
        tp = confusion[class_id, class_id].item()
        fp = confusion[:, class_id].sum().item() - tp
        fn = confusion[class_id, :].sum().item() - tp

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = confusion[class_id, :].sum().item()

        print(
            f"{class_name:10s} precision={precision:.4f} "
            f"recall={recall:.4f} f1={f1:.4f} support={support}"
        )

    print("\nConfusion matrix:")
    print(confusion.numpy())


def parse_args():
    parser = argparse.ArgumentParser(description="Training wake word su immagini GAF/MTF NumPy.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--onnx-name", default="wake_word_gaf_cnn.onnx")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--threshold-min", type=float, default=0.10)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--fp-penalty", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_train_ds = GafWakeWordDataset(args.data_dir / "training")
    test_ds = GafWakeWordDataset(args.data_dir / "testing")

    train_indices, val_indices = stratified_train_val_indices(
        full_train_ds.targets,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_ds = Subset(full_train_ds, train_indices)
    val_ds = Subset(full_train_ds, val_indices)

    train_targets = [full_train_ds.targets[index] for index in train_indices]

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Device: {device}")
    print(f"Classes: {CLASSES}")
    print(f"Train samples: {len(train_ds)} {Counter(train_targets)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)} {Counter(test_ds.targets)}")

    model = WakeWordGafCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_targets, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_path = args.output_dir / "best_wake_word_gaf_cnn.pt"
    onnx_path = args.output_dir / args.onnx_name
    best_threshold_path = args.output_dir / "best_threshold.txt"
    best_score = float("-inf")
    best_threshold = 0.5
    bad_epochs = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc, _, val_probs, val_targets = evaluate(
            model,
            val_loader,
            criterion,
            device,
            threshold=0.5,
        )
        threshold_search = find_best_threshold(
            val_probs,
            val_targets,
            min_threshold=args.threshold_min,
            max_threshold=args.threshold_max,
            step=args.threshold_step,
            fp_penalty=args.fp_penalty,
        )
        wake_metrics = threshold_search["metrics"]
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc@0.5={val_acc:.4f} | "
            f"best_thr={threshold_search['threshold']:.2f} "
            f"wake_precision={wake_metrics['precision']:.4f} "
            f"wake_recall={wake_metrics['recall']:.4f} "
            f"wake_f1={wake_metrics['f1']:.4f} "
            f"wake_fp={wake_metrics['false_positives']} "
            f"score={threshold_search['score']:.4f}"
        )

        if threshold_search["score"] > best_score:
            best_score = threshold_search["score"]
            best_threshold = threshold_search["threshold"]
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
            best_threshold_path.write_text(f"{best_threshold:.6f}\n", encoding="ascii")
            print(f"Saved best model: {best_path}")
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, confusion, _, _ = evaluate(
        model,
        test_loader,
        criterion,
        device,
        threshold=best_threshold,
    )
    print(
        f"\nSelected threshold={best_threshold:.2f} "
        f"(validation score={best_score:.4f})"
    )
    print(f"Test loss={test_loss:.4f} test_acc={test_acc:.4f}")
    print_report(confusion)

    dummy_input = torch.randn(1, 3, 64, 64, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        external_data=False,
        dynamo=False,
    )
    print(f"\nExported ONNX model to: {onnx_path}")


if __name__ == "__main__":
    main()
