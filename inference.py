import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "dinov3"))
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

# ========================
# Configurações
# ========================
checkpoint_path = "checkpoints/dinov3_vitl16_pretrain.pth"
images_path = "images"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# Carregar modelo DINOv3
# ========================
# Importa direto da lib do dinov3
import models.vision_transformer as vits

# ViT-L/16 (ajuste se usar outro backbone)
model = vits.__dict__["vit_large"](patch_size=16)
state_dict = torch.load(checkpoint_path, map_location="cpu")

model.load_state_dict(state_dict, strict=False)
model.eval()
model.to(device)

print("Modelo carregado com sucesso no", device)

# ========================
# Pré-processamento
# ========================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# Exemplo: baixar uma imagem da web
url = "https://pytorch.org/assets/images/deeplab1.png"
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB")

x = transform(img).unsqueeze(0).to(device)

# ========================
# Inferência
# ========================
with torch.no_grad():
    output = model(x)

print("Saída do modelo:", output.shape)
print("Vetor de features (primeiros 5 valores):", output[0, :5].cpu().numpy())