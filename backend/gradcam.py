import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

def denormalize(tensor):
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def generate_gradcam(model, image: Image.Image, target_class: int):
    gradients  = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    target_layer = model.features[-1]
    fwd_handle   = target_layer.register_forward_hook(forward_hook)
    bwd_handle   = target_layer.register_backward_hook(backward_hook)

    image_tensor = transform(image).unsqueeze(0)
    output       = model(image_tensor)
    model.zero_grad()
    output[0, target_class].backward()

    fwd_handle.remove()
    bwd_handle.remove()

    grad = gradients[0].squeeze(0)
    act  = activations[0].squeeze(0)

    weights = grad.mean(dim=(1, 2))
    cam     = (weights[:, None, None] * act).sum(dim=0)
    cam     = F.relu(cam)
    cam     = cam - cam.min()
    cam     = cam / (cam.max() + 1e-8)
    cam     = cam.cpu().numpy()
    cam     = cv2.resize(cam, (224, 224))

    original_img = denormalize(transform(image))
    heatmap      = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap      = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay      = 0.5 * original_img + 0.5 * heatmap
    overlay      = np.clip(overlay, 0, 1)
    overlay_uint8 = (overlay * 255).astype(np.uint8)

    return overlay_uint8

if __name__ == "__main__":
    print("gradcam.py loaded successfully")