import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import argparse
import time
import random
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

from statsmodels.stats.proportion import proportion_confint
import scipy.stats as stats

from dataset import CT3DDataset
from model import RFClassifier

def parse_arguments():
    parser = argparse.ArgumentParser(description='Read films')

    parser.add_argument('--data_root', type=str, default='new_dataset')
    parser.add_argument('--id_label_csv', type=str, default='./id_label_2_jia7.csv')
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--max_slices', type=int, default=512)

    # train
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--seed', type=int, default=2025)

    # model
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--feature_dim', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--aux_feature_size', type=int, default=57)

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--exp_name', type=str, default='exp')

    return parser.parse_args()

def setup_environment(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"logs/{args.exp_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)

    return device, log_dir, model_dir

def create_kfold_dataset(args):
    dataset = CT3DDataset(
        root_dir=args.data_root,
        id_label_csv=args.id_label_csv,  # './id_label.csv',
        max_slices=args.max_slices,
        cache_dir=args.cache_dir,
        aux_feature_size=args.aux_feature_size
    )
    all_ids = dataset.all_id

    total_size = len(dataset)
    fold_size = int(0.2 * total_size)

    fold1, fold2, fold3, fold4, fold5 = random_split(
        dataset, [fold_size, fold_size, fold_size, fold_size, total_size - 4*fold_size]
    )

    print(f"f1-f4: {len(fold1)} | f5: {len(fold5)}")

    return dataset, [fold1, fold2, fold3, fold4, fold5], all_ids

def create_dataset_and_loader(args):
    dataset = CT3DDataset(
        root_dir=args.data_root,
        id_label_csv=args.id_label_csv,  # './id_label.csv',
        max_slices=args.max_slices,
        cache_dir=args.cache_dir,
        aux_feature_size=args.aux_feature_size
    )
    all_ids = dataset.all_id

    total_size = len(dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    return train_dataset, val_dataset, test_dataset, all_ids

def train_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = []
    start_time = time.time()

    for i, data in enumerate(loader):
        volumes = data['volume'].to(device)
        features = data['features'].to(device)
        labels = data['label'].to(device)
        depths = data['depth'].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(volumes, features, depths)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * volumes.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % 10 == 0:
            avg_loss = running_loss / total
            accuracy = 100 * correct / total
            progress.append((i + 1, avg_loss, accuracy))
            print(f'Epoch [{epoch}] Batch [{i + 1}/{len(loader)}] '
                  f'Loss: {avg_loss:.4f} Acc: {accuracy:.2f}%')

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    elapsed = time.time() - start_time

    print(f'Epoch [{epoch}] Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% Time: {elapsed:.2f}s')
    return epoch_loss, epoch_acc, progress

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            volumes = data['volume'].to(device)
            features = data['features'].to(device)
            labels = data['label'].to(device)
            depths = data['depth'].to(device)

            outputs = model(volumes, features, depths)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * volumes.size(0)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss_val = running_loss / total
    acc_val = 100 * correct / total
    class_report = classification_report(all_labels, all_preds, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return loss_val, acc_val, conf_matrix, class_report

def save_model(model, optimizer, epoch, loss, acc, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'acc': acc
    }, path)

def plot_confusion_matrix(conf_matrix, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (True Labels)')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main(seed, fold_idx, folds):
    args = parse_arguments()
    device, log_dir, model_dir = setup_environment(args)

    train_idx = folds[fold_idx] + folds[fold_idx - 1] + folds[fold_idx - 2]
    val_idx = folds[fold_idx - 3]
    test_idx = folds[fold_idx - 4]

    train_loader = DataLoader(
        train_idx,
        batch_size=args.batch_size,
        collate_fn=CT3DDataset.collate_fn,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_idx,
        batch_size=args.batch_size,
        collate_fn=CT3DDataset.collate_fn,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_idx,
        batch_size=args.batch_size,
        collate_fn=CT3DDataset.collate_fn,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = RFClassifier(
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        aux_feature_size=args.aux_feature_size
    )
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5
    )

    scaler = torch.amp.GradScaler()

    best_val_acc = 0.0
    best_val_loss = 2.0
    best_epoch = 0
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, progress = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc, conf_matrix, class_report = validate(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"valset - Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            model_path = os.path.join(model_dir, f"{args.exp_name}_best_model.pth")
            save_model(model, optimizer, epoch, val_loss, val_acc, model_path)

            plot_confusion_matrix(
                conf_matrix,
                class_names=[str(i) for i in range(args.num_classes)],
                save_path=os.path.join(log_dir, f'confusion_matrix_epoch{epoch}.png')
            )
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            break

    checkpoint = torch.load(model_path)
    model_weights = checkpoint['model_state_dict']
    model.load_state_dict(model_weights)

    test_loss, test_acc, conf_matrix, class_report = validate(
        model, test_loader, criterion, device
    )
    print(f"testset - Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")

    report_path = os.path.join(log_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"seed: {seed}\n")
        f.write("Best Validation Results:\n")
        f.write(f"Epoch: {best_epoch}, Accuracy: {best_val_acc:.2f}%\n\n")
        f.write("Test Set Results:\n")
        f.write(f"Accuracy: {test_acc:.2f}%\n")
        f.write("Classification Report:\n")
        f.write(str(class_report))
        f.write("\n\nconf_matrix:\n")
        f.write(str(conf_matrix))

    os.rename(log_dir, f"{log_dir}_seed{seed}_fold_idx{fold_idx}_epoch{epoch}_test_acc{test_acc:.2f}")
    return test_acc, test_loss

if __name__ == "__main__":
    args = parse_arguments()
    seed = args.seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset, folds, all_ids = create_kfold_dataset(args)

    for fold_idx in range(5):
        test_acc, test_loss = main(seed, fold_idx, folds)
