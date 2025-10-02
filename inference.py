import os
import sys
import time
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

# --- Ajuste do path para importar o SAM-HQ ---
project_root = os.path.abspath("train/segment_anything_training")
sys.path.append(project_root)

from segment_anything import sam_model_registry

# --- Dispositivo ---
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"[INFO] Usando device: {DEVICE}")

# --- Função para carregar imagem e transformá-la ---
def load_image(path, target_size):
    image = Image.open(path).convert("RGB")
    image_np = np.array(image)
    orig_size = image_np.shape[:2]  # (H, W)

    # resize proporcional
    scale = target_size / max(orig_size)
    new_w = int(orig_size[1] * scale)
    new_h = int(orig_size[0] * scale)
    image_resized = Image.fromarray(image_np).resize((new_w, new_h))
    image_resized_np = np.array(image_resized)

    torch_img = torch.as_tensor(image_resized_np, device=DEVICE).permute(2, 0, 1).contiguous()[None, :, :, :]
    return torch_img, orig_size, image_np

# --- Função para salvar máscara + overlay ---
def save_mask(mask_np, image_np, save_mask_path, save_overlay_path):
    os.makedirs(os.path.dirname(save_mask_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_overlay_path), exist_ok=True)

    plt.imsave(save_mask_path, mask_np, cmap="gray")

    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)
    plt.imshow(mask_np, alpha=0.5, cmap="jet")
    plt.axis("off")
    plt.savefig(save_overlay_path, bbox_inches="tight", pad_inches=0)
    plt.close()

# --- Main de execução ---
if __name__ == "__main__":
    TARGET_SIZE = 2048
    INPUT_DIR = "SAMfinos"
    CHECKPOINT_PATH = "checkpoints/sam_hq_vit_h.pth"

    print(f"[INFO] Carregando modelo ViT-H do checkpoint {CHECKPOINT_PATH}")
    sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam.eval()
    print("<All keys matched successfully>")

    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/vis_masks", exist_ok=True)
    os.makedirs("results/vis_overlay", exist_ok=True)

    log_path = os.path.join("results", "logs", "inference_log.txt")
    log_file = open(log_path, "w")

    image_files = []
    if os.path.isdir(INPUT_DIR):
        for fname in os.listdir(INPUT_DIR):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                image_files.append(os.path.join(INPUT_DIR, fname))
    else:
        raise ValueError(f"Pasta de input '{INPUT_DIR}' não encontrada.")

    total = len(image_files)
    print(f"[INFO] Encontradas {total} imagens em {INPUT_DIR}")

    for idx, img_path in enumerate(image_files, 1):
        t0 = time.time()
        torch_img, orig_size, orig_np = load_image(img_path, TARGET_SIZE)

        # Aqui passa também o tamanho original!
        batched = [{
            "image": torch_img[0],
            "original_size": orig_size
        }]

        with torch.no_grad():
            outputs = sam(batched, multimask_output=False)

        out = outputs[0]
        masks = out["masks"]
        iou_pred = out["iou_predictions"]

        elapsed = time.time() - t0

        log_str = (
            f"Image: {img_path}\n"
            f"Input tensor shape: {torch_img.shape}\n"
            f"Final masks shape: {masks.shape}\n"
            f"IoU predictions shape: {iou_pred.shape}\n"
            f"Time (s): {elapsed:.3f}\n"
            "---------------------------\n"
        )
        log_file.write(log_str)

        progress = (idx / total) * 100
        print(f"[{idx}/{total}] {img_path} processada ({progress:.1f}%)")

        mask0 = masks[0][0].cpu().numpy()
        basename = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join("results", "vis_masks", f"{basename}_mask.png")
        overlay_path = os.path.join("results", "vis_overlay", f"{basename}_overlay.png")
        save_mask(mask0, orig_np, mask_path, overlay_path)

    log_file.close()
    print("[INFO] Inferência concluída. Resultados em pasta results/")