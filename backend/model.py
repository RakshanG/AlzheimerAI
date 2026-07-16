import os
import io
import torch
import torch.nn as nn
import numpy as np
import tempfile
import base64
from torchvision import models, transforms
from PIL import Image

try:
    import nibabel as nib
    from scipy.ndimage import zoom
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

MODEL_2D_PATH = os.path.join(os.path.dirname(__file__), "models", "efficientnet_b3_merged.pth")
MODEL_3D_PATH = os.path.join(os.path.dirname(__file__), "models", "resnet3d_oasis.pth")

LABEL_MAP_2D = {0: "Non Demented", 1: "Very Mild Demented", 2: "Mild Demented"}
LABEL_MAP_3D = {0: "Non Demented", 1: "Very Mild Demented", 2: "Mild Demented"}

transform_2d = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def build_efficientnet(num_classes=3):
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )
    return model

def load_model_2d():
    device     = torch.device("cpu")
    checkpoint = torch.load(MODEL_2D_PATH, map_location=device, weights_only=False)
    model      = build_efficientnet(num_classes=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device

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

    def _make_layer(self, out_ch, num_blocks, stride):
        layers = [ResBlock3D(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(1, num_blocks):
            layers.append(ResBlock3D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(self.drop(x))

def load_model_3d():
    if not os.path.exists(MODEL_3D_PATH):
        return None, None
    device     = torch.device("cpu")
    checkpoint = torch.load(MODEL_3D_PATH, map_location=device, weights_only=False)
    model      = ResNet3D(num_classes=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device

_model_2d = None
_model_3d = None
_device   = torch.device("cpu")

def load_model():
    global _model_2d, _model_3d, _device
    _model_2d, _device = load_model_2d()
    print("[Model] 2D EfficientNet-B3 loaded")
    _model_3d, _ = load_model_3d()
    if _model_3d:
        print("[Model] 3D ResNet-18 loaded")
    else:
        print("[Model] 3D model not found — 2D only mode active")
    return _model_2d, 1.0, _device

def predict_2d(image: Image.Image) -> dict:
    tensor = transform_2d(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        probs = torch.softmax(_model_2d(tensor), dim=1)
        conf, pred = probs.max(1)
    c = conf.item(); p = pred.item()
    all_probs = probs.squeeze().tolist()
    return {
        "model_type":      "2D",
        "model_name":      "EfficientNet-B3",
        "predicted_class": p,
        "predicted_label": LABEL_MAP_2D[p],
        "confidence":      round(c * 100, 2),
        "probabilities":   {LABEL_MAP_2D[i]: round(all_probs[i] * 100, 2) for i in range(3)},
        "uncertainty_flag": c < 0.70,
        "slices_3d":       None
    }

def generate_gradcam_3d(vol_np: np.ndarray) -> np.ndarray:
    model  = _model_3d
    tensor = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0).float()

    gradients  = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fwd = model.layer4.register_forward_hook(forward_hook)
    bwd = model.layer4.register_full_backward_hook(
        lambda m, gi, go: gradients.append(go[0])
    )

    fwd.remove()
    bwd.remove()

    fwd = model.layer4.register_forward_hook(forward_hook)

    model.zero_grad()
    output = model(tensor)
    pred_class = output.argmax(1).item()
    score = output[0, pred_class]
    score.backward()

    fwd.remove()

    if not activations or not gradients:
        return np.zeros_like(vol_np)

    grads   = gradients[0].squeeze()
    acts    = activations[0].squeeze()
    weights = grads.mean(dim=(1, 2, 3))

    cam = torch.zeros(acts.shape[1:])
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = torch.relu(cam).detach().numpy()
    cam = zoom(cam, [vol_np.shape[i] / cam.shape[i] for i in range(3)], order=1)
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    return cam.astype(np.float32)

def slice_to_b64(mri_slice: np.ndarray, cam_slice: np.ndarray, title: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    fig.patch.set_facecolor('#0f172a')

    axes[0].imshow(mri_slice.T, cmap='gray', origin='lower')
    axes[0].set_title('MRI Slice', color='white', fontsize=9)
    axes[0].axis('off')

    axes[1].imshow(mri_slice.T, cmap='gray', origin='lower')
    axes[1].imshow(cam_slice.T, cmap='jet', alpha=0.45, origin='lower')
    axes[1].set_title('Activation Map', color='white', fontsize=9)
    axes[1].axis('off')

    fig.suptitle(title, color='#a78bfa', fontsize=10, fontweight='bold')
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_3d_slices(vol_np: np.ndarray, cam_np: np.ndarray) -> dict:
    d, h, w = vol_np.shape
    return {
        "axial":    slice_to_b64(vol_np[d//2, :, :], cam_np[d//2, :, :], 'Axial View'),
        "sagittal": slice_to_b64(vol_np[:, h//2, :], cam_np[:, h//2, :], 'Sagittal View'),
        "coronal":  slice_to_b64(vol_np[:, :, w//2], cam_np[:, :, w//2], 'Coronal View')
    }

def predict_3d(nii_bytes: bytes, target_shape=(64, 64, 64)) -> dict:
    if not NIBABEL_AVAILABLE:
        raise RuntimeError("nibabel/scipy not installed. Run: pip3 install nibabel scipy")
    if _model_3d is None:
        raise RuntimeError("3D model not found. Place resnet3d_oasis.pth in models/ folder.")

    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
        tmp.write(nii_bytes)
        tmp_path = tmp.name

    try:
        img = nib.load(tmp_path)
        vol = img.get_fdata(dtype=np.float32)
    finally:
        os.remove(tmp_path)

    if vol.ndim == 4: vol = vol[..., 0]
    zoom_factors = [target_shape[i] / vol.shape[i] for i in range(3)]
    vol = zoom(vol, zoom_factors, order=1)
    vol = np.clip(vol, 0.0, 1.0) if vol.max() <= 1.0 else \
          (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    vol = vol.astype(np.float32)

    tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(_model_3d(tensor), dim=1)
        conf, pred = probs.max(1)

    c = conf.item(); p = pred.item()
    all_probs = probs.squeeze().tolist()

    try:
        cam     = generate_gradcam_3d(vol)
        slices  = generate_3d_slices(vol, cam)
    except Exception as e:
        print(f"[GradCAM3D] Failed: {e}")
        slices = None

    return {
        "model_type":      "3D",
        "model_name":      "ResNet3D-18",
        "predicted_class": p,
        "predicted_label": LABEL_MAP_3D[p],
        "confidence":      round(c * 100, 2),
        "probabilities":   {LABEL_MAP_3D[i]: round(all_probs[i] * 100, 2) for i in range(3)},
        "uncertainty_flag": c < 0.70,
        "slices_3d":       slices
    }

def predict_auto(file_bytes: bytes, filename: str) -> dict:
    fname = filename.lower()
    if fname.endswith(".nii") or fname.endswith(".nii.gz"):
        return predict_3d(file_bytes)
    else:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return predict_2d(image)

def predict(image: Image.Image) -> dict:
    return predict_2d(image)

if __name__ == "__main__":
    print("Loading models...")
    load_model()
    print("Done")