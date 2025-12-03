#!/usr/bin/env python3
"""
CLI training script for the 10-class UCF101 skeleton subset.
Usage example (PowerShell):
  & C:/Users/edosa/anaconda3/envs/nlp/python.exe scripts\train_10cls.py --epochs 30 --batch_size 8 --lr 1e-3 --num_workers 0

Supports optional AMP (use --use_amp) but will fall back to CPU if CUDA not available.
Saves: best_10cls.pt (best val acc), final checkpoint, training_history_10cls.csv
"""

import argparse
from pathlib import Path
import pickle
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure project root is on sys.path so imports like `Models.*` work when
# running the script directly from PowerShell/conda environments.
import sys
ROOT = Path(__file__).resolve().parents[1]

# Try package import first (when project installed or src in PYTHONPATH)
try:
    from har.Models.lstm_model import SkeletonLSTM
except Exception:
    # Fall back: add src/ to sys.path (common layout: repo/src/har/...)
    SRC = ROOT / 'src'
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from har.Models.lstm_model import SkeletonLSTM


class NormalizeKeypoints:
    def __init__(self, shape_key='img_shape'):
        self.shape_key = shape_key
    def __call__(self, sample):
        kp = sample['keypoint']
        shape = sample.get(self.shape_key) or sample.get('original_shape')
        if shape is None:
            return sample
        try:
            h, w = float(shape[0]), float(shape[1])
            if w > 0 and h > 0:
                kp = kp.clone()
                kp[..., 0] = kp[..., 0] / w
                kp[..., 1] = kp[..., 1] / h
                sample['keypoint'] = kp
        except Exception:
            pass
        return sample


class UCFSkeletonDataset(Dataset):
    def __init__(self, annotations_list, transform=None, select_best_by_score=True):
        self.annotations = annotations_list
        self.transform = transform
        self.select_best_by_score = select_best_by_score
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        kp = ann.get('keypoint')
        kp_score = ann.get('keypoint_score', None)
        kp = np.array(kp)  # M x T x V x C
        if kp.ndim != 4:
            raise ValueError(f'Unexpected keypoint shape for idx={idx}: {kp.shape}')
        M = kp.shape[0]
        person_idx = 0
        if M > 1 and kp_score is not None and self.select_best_by_score:
            scores = np.array(kp_score)
            mean_scores = scores.mean(axis=(1,2))
            person_idx = int(mean_scores.argmax())
        person_kp = kp[person_idx]  # T x V x C (C expected 2: x,y)

        # Convert to float32 and to torch tensor
        person_kp = person_kp.astype(np.float32)
        tensor_pos = torch.from_numpy(person_kp)  # T x V x C

        # Compute velocity (first-order derivative). Pad first time-step with zeros.
        if tensor_pos.ndim == 3 and tensor_pos.shape[2] >= 2:
            vel = tensor_pos[1:] - tensor_pos[:-1]  # (T-1) x V x C
            vel = torch.cat([torch.zeros((1, vel.shape[1], vel.shape[2]), dtype=vel.dtype), vel], dim=0)  # T x V x C
            # Concatenate position and velocity along channel dim -> T x V x (C*2)
            tensor_kp = torch.cat([tensor_pos, vel], dim=2)
        else:
            # Fallback: if channels unexpected, return positions only
            tensor_kp = tensor_pos
        sample = {
            'keypoint': tensor_kp,
            'label': int(ann.get('label')) if ann.get('label') is not None else None,
            'frame_dir': ann.get('frame_dir'),
            'total_frames': ann.get('total_frames'),
            'img_shape': ann.get('img_shape'),
            'original_shape': ann.get('original_shape')
        }
        if kp_score is not None:
            score_np = np.array(kp_score)[person_idx]
            sample['score'] = torch.from_numpy(score_np.astype(np.float32))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def pad_sequence_kp_right(kps):
    T_max = max(t.shape[0] for t in kps)
    V = kps[0].shape[1]
    C = kps[0].shape[2]
    padded, masks = [], []
    for t in kps:
        T = t.shape[0]
        pad_len = T_max - T
        if pad_len > 0:
            pad_tensor = torch.zeros((pad_len, V, C), dtype=t.dtype)
            p = torch.cat([pad_tensor, t], dim=0)
            mask = torch.cat([torch.zeros((pad_len, V)), torch.ones((T, V))], dim=0)
        else:
            p = t
            mask = torch.ones((T, V))
        padded.append(p)
        masks.append(mask)
    batch_kp = torch.stack(padded, dim=0)
    batch_mask = torch.stack(masks, dim=0)
    return batch_kp, batch_mask


def ucf_collate_fn_right(batch):
    kps = [item['keypoint'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    frame_dirs = [item['frame_dir'] for item in batch]
    total_frames = [item['total_frames'] for item in batch]
    scores = [item.get('score') for item in batch]
    batch_kp, batch_mask = pad_sequence_kp_right(kps)
    batch_scores = None
    if any(s is not None for s in scores):
        scores_repl = []
        for i, s in enumerate(scores):
            if s is None:
                T = kps[i].shape[0]
                V = kps[i].shape[1]
                scores_repl.append(torch.zeros((T, V), dtype=torch.float32))
            else:
                scores_repl.append(s)
        T_max = batch_kp.shape[1]
        padded_scores = []
        for s in scores_repl:
            pad_len = T_max - s.shape[0]
            if pad_len > 0:
                pad = torch.zeros((pad_len, s.shape[1]), dtype=s.dtype)
                padded_scores.append(torch.cat([pad, s], dim=0))
            else:
                padded_scores.append(s)
        batch_scores = torch.stack(padded_scores, dim=0)
    return {'keypoint': batch_kp, 'mask': batch_mask, 'label': labels, 'frame_dir': frame_dirs, 'total_frames': total_frames, 'score': batch_scores}


def load_filtered_pickle(pickle_path: Path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    annotations = data.get('annotations') or []
    # pre-convert keypoints and scores to numpy (speeds up Dataset)
    for i, ann in enumerate(annotations):
        kp = ann.get('keypoint')
        if kp is not None and not hasattr(kp, 'shape'):
            try:
                ann['keypoint'] = np.asarray(kp, dtype=np.float32)
            except Exception:
                ann['keypoint'] = kp
        if 'keypoint_score' in ann and ann['keypoint_score'] is not None and not hasattr(ann['keypoint_score'], 'shape'):
            try:
                ann['keypoint_score'] = np.asarray(ann['keypoint_score'], dtype=np.float32)
            except Exception:
                pass
    return annotations


def compute_class_weights_from_labels(labels_arr):
    classes, counts = np.unique(labels_arr, return_counts=True)
    weights = counts.sum() / (len(classes) * counts)
    # return mapping dict {class: weight}
    return classes, weights


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pickle', type=str, default='Dataset/ucf101_2d_10cls.pkl')
    p.add_argument('--mapping', type=str, default='Dataset/label_mapping_10cls.json')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--val_batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--use_amp', action='store_true', help='Enable AMP if CUDA available')
    p.add_argument('--save_dir', type=str, default='.')
    return p.parse_args()


def main():
    args = parse_args()
    pkl = Path(args.pickle)
    if not pkl.exists():
        raise FileNotFoundError(f'Filtered pickle not found: {pkl} — run scripts/create_10class_subset.py first')

    annotations = load_filtered_pickle(pkl)
    print(f'Loaded {len(annotations)} annotations')

    # build df_ann
    rows = []
    for ann in annotations:
        kp = ann.get('keypoint')
        kp_shape = None
        try:
            if kp is not None:
                kp_shape = np.array(kp).shape
        except Exception:
            kp_shape = None
        rows.append({'frame_dir': ann.get('frame_dir'), 'total_frames': ann.get('total_frames'), 'img_shape': ann.get('img_shape'), 'original_shape': ann.get('original_shape'), 'label': int(ann.get('label')) if ann.get('label') is not None else None, 'keypoint_shape': kp_shape})
    df_ann = pd.DataFrame(rows)

    # split
    indices = np.arange(len(annotations))
    try:
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=df_ann['label'].values, random_state=42)
    except Exception:
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(indices))
        split = int(0.8 * len(indices))
        train_idx = perm[:split]
        val_idx = perm[split:]

    train_anns = [annotations[i] for i in train_idx]
    val_anns = [annotations[i] for i in val_idx]

    normalize = NormalizeKeypoints(shape_key='img_shape')
    train_ds = UCFSkeletonDataset(train_anns, transform=normalize)
    val_ds = UCFSkeletonDataset(val_anns, transform=normalize)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=ucf_collate_fn_right, num_workers=args.num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch, shuffle=False, collate_fn=ucf_collate_fn_right, num_workers=args.num_workers, pin_memory=False)

    # device and AMP
    use_amp = args.use_amp and torch.cuda.is_available()
    device = torch.device('cuda' if use_amp else 'cpu')
    print('Device:', device)
    print('AMP requested:', args.use_amp, 'AMP active:', use_amp)

    # class weights
    labels_arr = df_ann['label'].values
    classes, weights = compute_class_weights_from_labels(labels_arr)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print('Class weights mapping:', dict(zip(classes.tolist(), weights.tolist())))

    # model (input channels: pos(x,y) + vel(x,y) = 4 channels)
    num_classes = int(df_ann['label'].nunique())
    model = SkeletonLSTM(num_joints=17, in_channels=4, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    best_val_acc = 0.0
    history = []
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for batch in train_loader:
            kp = batch['keypoint'].to(device)
            labels_b = batch['label'].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(kp)
                loss = criterion(logits, labels_b)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * labels_b.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels_b).sum().item()
            running_total += labels_b.size(0)
        train_loss = running_loss / running_total if running_total>0 else 0.0
        train_acc = running_correct / running_total if running_total>0 else 0.0

        # validation
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for batch in val_loader:
                kp = batch['keypoint'].to(device)
                labels_b = batch['label'].to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(kp)
                    loss = criterion(logits, labels_b)
                v_loss += loss.item() * labels_b.size(0)
                preds = logits.argmax(dim=1)
                v_correct += (preds == labels_b).sum().item()
                v_total += labels_b.size(0)
        val_loss = v_loss / v_total if v_total>0 else 0.0
        val_acc = v_correct / v_total if v_total>0 else 0.0
        epoch_time = time.time() - t0
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'epoch_time': epoch_time})
        print(f'Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, time={epoch_time:.1f}s')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = save_dir / 'best_10cls.pt'
            torch.save({'model_state': model.state_dict(), 'num_classes': num_classes, 'epoch': epoch, 'val_acc': val_acc}, best_path)
            print(f'New best model saved to {best_path} (val_acc={val_acc:.4f})')

    # final save
    final_path = save_dir / 'lstm_10cls_cli_final.pt'
    torch.save({'model_state': model.state_dict(), 'num_classes': num_classes}, final_path)
    hist_path = save_dir / 'training_history_10cls.csv'
    try:
        pd.DataFrame(history).to_csv(hist_path, index=False)
        print('Saved training history to', hist_path)
    except Exception as e:
        print('Could not save training history:', e)
    print('Training finished. Best val acc =', best_val_acc)


if __name__ == '__main__':
    main()
