import os, time
import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import zoom
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

OASIS_ROOT   = './data/neurite-oasis.v1.0'
OUTPUT_DIR   = './outputs_3d'
TARGET_SHAPE = (64, 64, 64)
EPOCHS       = 60
BATCH_SIZE   = 2
LR           = 3e-5
ES_PATIENCE  = 15
CONF_THRESH  = 0.75

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cpu')
print('Using CPU')

LABEL_MAP    = {'NonDemented':0, 'VeryMildDemented':1, 'MildDemented':2}
IDX_TO_CLASS = {v: k for k, v in LABEL_MAP.items()}

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma
    def forward(self, inputs, targets):
        ce   = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1    = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm3d(out_ch)
        self.conv2    = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm3d(out_ch)
        self.relu     = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class ResNet3D(nn.Module):
    def __init__(self, layers=[2,2,2,2], num_classes=3, base_filters=32, dropout_p=0.4):
        super().__init__()
        self.in_ch = base_filters
        self.stem  = nn.Sequential(
            nn.Conv3d(1, base_filters, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_filters, base_filters, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(base_filters*1, layers[0], stride=1)
        self.layer2 = self._make_layer(base_filters*2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_filters*4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_filters*8, layers[3], stride=2)
        self.pool   = nn.AdaptiveAvgPool3d(1)
        self.drop   = nn.Dropout(p=dropout_p)
        self.fc     = nn.Linear(base_filters*8, num_classes)
        self._init_weights()

    def _make_layer(self, out_ch, num_blocks, stride):
        layers = [ResBlock3D(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(1, num_blocks):
            layers.append(ResBlock3D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(self.drop(x))

def load_cdr_map(oasis_root):
    csv_path = os.path.join(oasis_root, 'oasis_cross-sectional.csv')
    if not os.path.exists(csv_path):
        print('CSV not found — all labels set to NonDemented')
        return {}
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().upper() for c in df.columns]
    print(f'CSV loaded — {len(df)} subjects found')
    return dict(zip(df['ID'].str.strip(), df['CDR'].fillna(0.0)))

def cdr_to_label(cdr):
    cdr = float(cdr)
    if cdr == 0.0: return 0
    if cdr == 0.5: return 1
    return 2

def build_records(oasis_root):
    cdr_map = load_cdr_map(oasis_root)
    records = []
    dirs    = sorted([d for d in os.listdir(oasis_root) if d.startswith('OASIS_OAS1_')])
    for d in dirs:
        nii = os.path.join(oasis_root, d, 'aligned_norm.nii.gz')
        if not os.path.exists(nii):
            continue
        key = d.replace('OASIS_', '')
        cdr = cdr_map.get(key, 0.0)
        records.append({'subject_id': d, 'nii_path': nii, 'label': cdr_to_label(cdr)})
    print(f'{len(records)} subjects loaded')
    counts = defaultdict(int)
    for r in records: counts[IDX_TO_CLASS[r['label']]] += 1
    for cls, cnt in sorted(counts.items()): print(f'   {cls:<22}: {cnt}')
    return records

def split_subjects(records, seed=42):
    ids  = [r['subject_id'] for r in records]
    lbls = [r['label']      for r in records]
    tr_ids, tmp_ids, _, tmp_lbls = train_test_split(
        ids, lbls, test_size=0.30, stratify=lbls, random_state=seed)
    va_ids, te_ids = train_test_split(
        tmp_ids, test_size=0.50, stratify=tmp_lbls, random_state=seed)
    id_map = {r['subject_id']: r for r in records}
    assert not set(tr_ids) & set(va_ids), 'Train/Val overlap!'
    assert not set(tr_ids) & set(te_ids), 'Train/Test overlap!'
    print(f'Zero subject overlap verified')
    print(f'Train:{len(tr_ids)} | Val:{len(va_ids)} | Test:{len(te_ids)}')
    return [id_map[i] for i in tr_ids], [id_map[i] for i in va_ids], [id_map[i] for i in te_ids]

def load_volume(nii_path):
    vol = nib.load(nii_path).get_fdata(dtype=np.float32)
    if vol.ndim == 4: vol = vol[..., 0]
    factors = [TARGET_SHAPE[i] / vol.shape[i] for i in range(3)]
    vol = zoom(vol, factors, order=1)
    vol = np.clip(vol, 0.0, 1.0) if vol.max() <= 1.0 else \
          (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    return vol.astype(np.float32)

class OASISDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        r   = self.records[idx]
        vol = load_volume(r['nii_path'])
        if self.augment:
            if np.random.rand() < 0.5:
                vol = np.flip(vol, axis=2).copy()
            vol = np.clip(vol * np.random.uniform(0.9, 1.1), 0, 1)
            vol = np.clip(vol + np.random.normal(0, 0.01, vol.shape), 0, 1).astype(np.float32)
        return torch.from_numpy(vol).unsqueeze(0), torch.tensor(r['label'], dtype=torch.long)
    def class_weights(self):
        counts = defaultdict(int)
        for r in self.records: counts[r['label']] += 1
        total = len(self.records)
        w = torch.tensor([
            1.0,
            total / counts.get(1, 1),
            total / counts.get(2, 1)
        ], dtype=torch.float32)
        w = w / w.sum() * 3
        print(f'Class weights: {w.numpy().round(3)}')
        return w

if __name__ == '__main__':

    test_vol = torch.randn(1, 1, 64, 64, 64)
    test_mod = ResNet3D(num_classes=3)
    print(f'Model output shape : {test_mod(test_vol).shape}')
    print(f'Trainable params   : {sum(p.numel() for p in test_mod.parameters()):,}')
    del test_mod, test_vol

    all_records = build_records(OASIS_ROOT)
    train_rec, val_rec, test_rec = split_subjects(all_records)

    train_ds = OASISDataset(train_rec, augment=True)
    val_ds   = OASISDataset(val_rec,   augment=False)
    test_ds  = OASISDataset(test_rec,  augment=False)

    class_weights = train_ds.class_weights()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f'Batches → Train:{len(train_loader)} | Val:{len(val_loader)} | Test:{len(test_loader)}')

    model     = ResNet3D(num_classes=3).to(device)
    criterion = FocalLoss(weight=class_weights.to(device), gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5)

    history      = {'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[],'train_f1':[],'val_f1':[]}
    best_val_f1  = 0.0
    es_counter   = 0
    best_path    = os.path.join(OUTPUT_DIR, 'resnet3d_oasis.pth')

    print(f'Device:{device} | Epochs:{EPOCHS} | Batch:{BATCH_SIZE} | LR:{LR}')
    print(f'Output: {best_path}')

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        tr_preds, tr_labels = [], []

        for i, (vols, labels) in enumerate(train_loader):
            vols, labels = vols.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(vols)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            preds = logits.argmax(1)
            tr_loss    += loss.item() * vols.size(0)
            tr_correct += (preds == labels).sum().item()
            tr_total   += vols.size(0)
            tr_preds.extend(preds.cpu().numpy())
            tr_labels.extend(labels.cpu().numpy())
            if (i+1) % 10 == 0:
                print(f'  Ep{epoch:02d} [{i+1}/{len(train_loader)}] loss={tr_loss/tr_total:.4f}', end='\r')

        tr_loss /= tr_total
        tr_acc   = tr_correct / tr_total
        tr_f1    = f1_score(tr_labels, tr_preds, average='macro', zero_division=0)

        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        va_preds, va_labels = [], []

        with torch.no_grad():
            for vols, labels in val_loader:
                vols, labels = vols.to(device), labels.to(device)
                logits = model(vols)
                loss   = criterion(logits, labels)
                preds  = logits.argmax(1)
                va_loss    += loss.item() * vols.size(0)
                va_correct += (preds == labels).sum().item()
                va_total   += vols.size(0)
                va_preds.extend(preds.cpu().numpy())
                va_labels.extend(labels.cpu().numpy())

        va_loss /= va_total
        va_acc   = va_correct / va_total
        va_f1    = f1_score(va_labels, va_preds, average='macro', zero_division=0)

        history['train_loss'].append(tr_loss); history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc);   history['val_acc'].append(va_acc)
        history['train_f1'].append(tr_f1);     history['val_f1'].append(va_f1)

        scheduler.step(va_f1)

        print(f'Epoch {epoch:02d}/{EPOCHS} | '
              f'Train Loss:{tr_loss:.4f} Acc:{tr_acc:.4f} F1:{tr_f1:.4f} | '
              f'Val Loss:{va_loss:.4f} Acc:{va_acc:.4f} F1:{va_f1:.4f} | '
              f'{time.time()-t0:.1f}s')

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_acc':          va_acc,
                'val_f1':           va_f1,
                'num_classes':      3
            }, best_path)
            print(f'  Best model saved (val_f1={va_f1:.4f})')
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= ES_PATIENCE:
                print(f'Early stopping at epoch {epoch}. Best F1={best_val_f1:.4f}')
                break

    print(f'Training complete! Best val_f1 = {best_val_f1:.4f}')

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    te_preds, te_labels, te_confs = [], [], []
    with torch.no_grad():
        for vols, labels in test_loader:
            probs = torch.softmax(model(vols.to(device)), dim=1)
            conf, pred = probs.max(1)
            te_preds.extend(pred.cpu().numpy())
            te_labels.extend(labels.numpy())
            te_confs.extend(conf.cpu().numpy())

    te_acc = sum(p==l for p,l in zip(te_preds,te_labels)) / len(te_labels)
    te_f1  = f1_score(te_labels, te_preds, average='macro', zero_division=0)

    print(f'Accuracy    : {te_acc*100:.2f}%')
    print(f'Macro F1    : {te_f1:.4f}')
    print(f'Mean Conf   : {np.mean(te_confs):.4f}')
    print(f'Uncertain   : {sum(c<CONF_THRESH for c in te_confs)/len(te_confs)*100:.1f}%')
    print(classification_report(
        te_labels, te_preds,
        target_names=[IDX_TO_CLASS[i] for i in range(3)],
        zero_division=0
    ))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in zip(axes, ['loss','acc','f1'], ['Loss','Accuracy','Macro F1']):
        ax.plot(history[f'train_{metric}'], label='Train', color='#4A90D9')
        ax.plot(history[f'val_{metric}'],   label='Val',   color='#E05C5C')
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3); ax.set_xlabel('Epoch')
    fig.suptitle('ResNet3D-18 Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)

    cm = confusion_matrix(te_labels, te_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[IDX_TO_CLASS[i] for i in range(3)],
                yticklabels=[IDX_TO_CLASS[i] for i in range(3)], ax=ax)
    ax.set_title('Confusion Matrix — ResNet3D-18')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
    plt.show()

    print(f'Outputs saved to : {OUTPUT_DIR}/')
    print(f'Model saved at   : {best_path}')
    print(f'Copy to          : models/resnet3d_oasis.pth')