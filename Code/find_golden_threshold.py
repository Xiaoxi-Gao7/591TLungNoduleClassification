import torch
import numpy as np
from sklearn.metrics import accuracy_score
from data_utils import get_loaders
from model_factory_v4 import get_model_v4

# V4
DATA_DIR = "LIDC-IDRI-slices"
MODEL_PATH = "Checkpoints_V4_Final_Sprint/best_model_v4.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_optimal():
    _, val_loader = get_loaders(DATA_DIR, batch_size=8, mode="V4")
    model = get_model_v4().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    y_true, y_probs = [], []

    print("finding the golden threshold...")
    with torch.no_grad():
        for (imgs_l, imgs_g), labels in val_loader:
            imgs_l, imgs_g = imgs_l.to(DEVICE), imgs_g.to(DEVICE)
            outputs = model(imgs_l, imgs_g)
            probs = torch.softmax(outputs, dim=1)[:, 1] 
            
            y_true.extend(labels.numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    
    best_threshold = 0.5
    max_acc = 0.0
    

    for th in np.arange(0.3, 0.7, 0.01):
        preds = (y_probs >= th).astype(int)
        acc = accuracy_score(y_true, preds)
        
        if acc > max_acc:
            max_acc = acc
            best_threshold = th
        
        if round(th*100) % 5 == 0:
            print(f"Threshold: {th:.2f} | Accuracy: {acc*100:.2f}%")

    print("\n" + "="*30)
    print(f"✨ find the golden threshold: {best_threshold:.2f}")
    print(f"🔥 final Accuracy: {max_acc*100:.2f}%")
    print("="*30)
    

if __name__ == "__main__":
    find_optimal()