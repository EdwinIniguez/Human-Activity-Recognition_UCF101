#!/usr/bin/env python3
"""
Inference utility for the 10-class UCF101 skeleton subset.
Saves per-sample predictions, per-class metrics and a confusion matrix image.

Example (PowerShell):
& C:/Users/edosa/anaconda3/envs/nlp/python.exe scripts\infer_10cls.py --checkpoint artifacts\best_10cls.pt --pickle Dataset\ucf101_2d_10cls.pkl --out_dir inference_outputs --batch_size 32 --in_channels 4
"""

import argparse
from pathlib import Path
import pickle
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Add project root to path so `Models` imports resolve when run from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Models.lstm_model import SkeletonLSTM


def pad_sequence_kp_right(kps):
    T_max = max(t.shape[0] for t in kps)
    V = kps[0].shape[1]
    C = kps[0].shape[2]
    padded = []
    masks = []
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


class UCFSkeletonDataset:
    def __init__(self, annotations_list, transform=None, select_best_by_score=True, include_velocity=False):
        self.annotations = annotations_list
        self.transform = transform
        self.select_best_by_score = select_best_by_score
        self.include_velocity = include_velocity
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        kp = ann.get('keypoint')
        kp_score = ann.get('keypoint_score', None)
        kp = np.array(kp)
        # select person if multiple
        M = kp.shape[0]
        person_idx = 0
        if M > 1 and kp_score is not None and self.select_best_by_score:
            scores = np.array(kp_score)
            mean_scores = scores.mean(axis=(1,2))
            person_idx = int(mean_scores.argmax())
        person_kp = kp[person_idx].astype(np.float32)  # T x V x C_original
        # convert to tensor
        tensor_pos = torch.from_numpy(person_kp)
        # Optionally compute velocity and concatenate (pos + vel) when requested
        if self.include_velocity and tensor_pos.ndim == 3 and tensor_pos.shape[2] >= 2:
            vel = tensor_pos[1:] - tensor_pos[:-1]
            vel = torch.cat([torch.zeros((1, vel.shape[1], vel.shape[2]), dtype=vel.dtype), vel], dim=0)
            tensor_kp = torch.cat([tensor_pos, vel], dim=2)
        else:
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


def load_annotations(pickle_path: Path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    annotations = data.get('annotations') or []
    # ensure keypoints are numpy
    for ann in annotations:
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--pickle', type=str, default='Dataset/ucf101_2d_10cls.pkl')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--use_amp', action='store_true')
    p.add_argument('--out_dir', type=str, default='inference_outputs')
    p.add_argument('--in_channels', type=int, default=4, help='input channels used when creating the model (default 4: pos+vel)')
    p.add_argument('--top_k', type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    pkl = Path(args.pickle)
    if not pkl.exists():
        raise FileNotFoundError(f'Pickle not found: {pkl}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations(pkl)
    print(f'Loaded {len(annotations)} annotations')

    # build df_ann for labels
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

    val_anns = [annotations[i] for i in val_idx]
    normalize = None  # keep same behavior as training script (NormalizeKeypoints applied earlier if used)
    include_velocity = True if args.in_channels == 4 else False
    val_ds = UCFSkeletonDataset(val_anns, transform=normalize, include_velocity=include_velocity)

    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=ucf_collate_fn_right, num_workers=args.num_workers, pin_memory=False)

    # device
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    use_amp = args.use_amp and device.type == 'cuda'
    print('Device for inference:', device, 'Use AMP:', use_amp)

    # load model
    num_classes = int(df_ann['label'].nunique())
    model = SkeletonLSTM(num_joints=17, in_channels=args.in_channels, num_classes=num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # if checkpoint stored dict with 'model_state'
    if 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    all_preds = []
    all_probs = []
    all_y = []
    sample_ids = []
    frame_dirs = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            kp = batch['keypoint'].to(device)
            labels_b = batch['label'].to(device)
            # use amp.autocast with device_type to avoid future deprecation warnings
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(kp)
                probs = F.softmax(logits, dim=1)
            topk = torch.topk(probs, k=args.top_k, dim=1)
            preds = topk.indices.cpu().numpy()
            probs_np = probs.cpu().numpy()
            for bi in range(labels_b.size(0)):
                idx_global = val_idx[i * args.batch_size + bi] if (i * args.batch_size + bi) < len(val_idx) else None
                sample_ids.append(idx_global)
                frame_dirs.append(batch['frame_dir'][bi])
                all_y.append(int(labels_b[bi].cpu().item()))
                all_preds.append(int(preds[bi, 0]))
                all_probs.append(float(probs_np[bi, int(preds[bi,0])]))

    preds_df = pd.DataFrame({'sample_idx': sample_ids, 'frame_dir': frame_dirs, 'true_label': all_y, 'pred_label': all_preds, 'pred_prob': all_probs})
    preds_csv = out_dir / 'predictions.csv'
    preds_df.to_csv(preds_csv, index=False)
    print('Saved predictions to', preds_csv)

    # metrics
    labels_unique = np.unique(all_y)
    precision, recall, f1, support = precision_recall_fscore_support(all_y, all_preds, labels=labels_unique, zero_division=0)
    per_class = pd.DataFrame({'label': labels_unique, 'precision': precision, 'recall': recall, 'f1': f1, 'support': support})
    per_csv = out_dir / 'per_class_metrics_inference.csv'
    per_class.to_csv(per_csv, index=False)
    print('Saved per-class metrics to', per_csv)

    # confusion matrix
    cm = confusion_matrix(all_y, all_preds, labels=labels_unique)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, cmap='Blues', annot=True, fmt='d')
    plt.title('Confusion matrix (validation) - 10 classes')
    plt.xlabel('predicted')
    plt.ylabel('true')
    cm_path = out_dir / 'confusion_matrix.png'
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print('Saved confusion matrix to', cm_path)

    print('Inference complete. Summary:')
    print(per_class)


if __name__ == '__main__':
    main()
