import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from data_utils import get_loaders
from model_factory_v4 import get_model_v4


DATA_DIR = "LIDC-IDRI-slices"
MODEL_PATH = "Checkpoints_V4_Final_Sprint/best_model_v4.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_and_evaluate():
    _, val_loader = get_loaders(DATA_DIR, batch_size=8, mode="V4")
    
    model = get_model_v4().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for (imgs_l, imgs_g), labels in val_loader:
            imgs_l, imgs_g, labels = imgs_l.to(DEVICE), imgs_g.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs_l, imgs_g)
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    # --- confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Non-Nodule', 'Nodule'], 
                yticklabels=['Non-Nodule', 'Nodule'])
    plt.title('V4 Confusion Matrix')
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.savefig('v4_confusion_matrix.png')
    print("confusion matrix is already saved: v4_confusion_matrix.png")

    # ---  ROC ---
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('V4 ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('v4_roc_curve.png')
    print(f" ROC : v4_roc_curve.png (AUC: {roc_auc:.4f})")

    # ---  report
    print("\n report:")
    print(classification_report(y_true, y_pred, target_names=['Non-Nodule', 'Nodule']))

if __name__ == "__main__":
    plot_and_evaluate()