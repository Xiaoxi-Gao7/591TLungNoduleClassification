import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from data_utils import get_loaders
from model_factory_v4 import get_model_v4
from torchvision import transforms as T

DATA_DIR = "LIDC-IDRI-slices"
MODEL_PATH = "Checkpoints_V4_Final_Sprint/best_model_v4.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_tta(imgs):
    """ TTA 变"""
    # imgs is a tensor [B, 3, H, W]
    tta_variants = [
        imgs,                                # 1. identity
        torch.flip(imgs, dims=[3]),          # 2. horizontal flip
        torch.flip(imgs, dims=[2]),          # 3. vertical flip
        torch.rot90(imgs, k=1, dims=[2, 3]), # 4. rotate 90
        torch.rot90(imgs, k=3, dims=[2, 3])  # 5. rotate 270 
    ]
    return tta_variants

def evaluate_with_tta():
    # 1. load dataset (V4)
    _, val_loader = get_loaders(DATA_DIR, batch_size=8, mode="V4")
    
    # 2. load model
    model = get_model_v4().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    y_true, y_pred_tta = [], []

    print(" TTA (5x Voting)... ")
    
    with torch.no_grad():
        for (imgs_l, imgs_g), labels in val_loader:
            imgs_l, imgs_g = imgs_l.to(DEVICE), imgs_g.to(DEVICE)
            
            # TTA 
            tta_l = apply_tta(imgs_l)
            tta_g = apply_tta(imgs_g)
            
            # aggregate the predicted probabilities
            batch_probs = torch.zeros((imgs_l.size(0), 2)).to(DEVICE)
            
            for i in range(len(tta_l)):
                outputs = model(tta_l[i], tta_g[i])
                batch_probs += torch.softmax(outputs, dim=1)
            
            # average
            avg_probs = batch_probs / len(tta_l)
            _, preds = torch.max(avg_probs, 1)
            
            y_true.extend(labels.numpy())
            y_pred_tta.extend(preds.cpu().numpy())

    # 3. results
    final_acc = accuracy_score(y_true, y_pred_tta)
    print(f"\n✨ TTA finish！")
    print(f"🔥 TTA  Acc: {final_acc*100:.2f}%")
    print(classification_report(y_true, y_pred_tta, target_names=['Non-Nodule', 'Nodule']))

if __name__ == "__main__":
    evaluate_with_tta()