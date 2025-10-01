import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "dinov3"))

import torch
from torchvision import transforms
from PIL import Image

# ========================
# Configurações
# ========================
checkpoint_path = "checkpoints/dinov3_vitl16_pretrain.pth"
images_path = "images"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# Carregar modelo DINOv3
# ========================
import models.vision_transformer as vits

model = vits.__dict__["vit_large"](patch_size=16)
state_dict = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval().to(device)

print(f"✅ Modelo carregado no dispositivo: {device}")

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
print("✅ Pré-processamento definido.")

# ========================
# Inferência em todas as imagens
# ========================
results = []
image_files = [f for f in os.listdir(images_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if not image_files:
    print("⚠️ Nenhuma imagem encontrada em", images_path)
else:
    for img_name in image_files:
        img_path = os.path.join(images_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(x)

            feat_vector = output[0].cpu().numpy()
            results.append((img_name, feat_vector[:5]))  # salva só os 5 primeiros valores
            print(f"✅ {img_name} -> Saída: {output.shape}")
        except Exception as e:
            print(f"❌ Erro ao processar {img_name}: {e}")

    # Salvar resultados
    with open("inference_results.txt", "w") as f:
        for name, vec in results:
            f.write(f"{name}: {vec}\n")

    print("✅ Inferência concluída. Resultados salvos em inference_results.txt")