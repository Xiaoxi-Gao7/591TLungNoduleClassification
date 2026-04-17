import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm
from data_utils import get_loaders 
from model_factory_v3 import get_model_v3 

# 保持 FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt)**self.gamma * ce_loss).mean()

# 配置：调高一点学习率，因为 EfficientNet 需要稍微多一点“动力”
DATA_DIR, BATCH_SIZE, LR, EPOCHS = "LIDC-IDRI-slices", 16, 4e-5, 25 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    train_loader, val_loader = get_loaders(DATA_DIR, BATCH_SIZE)
    model = get_model_v3().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3) # 换成 AdamW
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
    criterion = FocalLoss()

    
    SAVE_DIR = "Checkpoints_V3_EfficientNet"
    os.makedirs(SAVE_DIR, exist_ok=True)
    best_path = f"{SAVE_DIR}/best_model_eff.pth"
    log_path = f"{SAVE_DIR}/v3_sprint_log.csv"
    
    best_acc, history = 0.0, []
    patience, early_stop_counter = 6, 0 # more patient

    print(" V3 starts：EfficientNet-B0 + AdamW + WarmRestarts")

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs); loss = criterion(out, labels)
            loss.backward(); optimizer.step()
            _, pred = torch.max(out, 1); total += labels.size(0); correct += (pred == labels).sum().item()
            pbar.set_postfix(acc=f"{100.*correct/total:.2f}%")

        
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                v_out = model(imgs); _, v_pred = torch.max(v_out, 1)
                v_total += labels.size(0); v_correct += (v_pred == labels).sum().item()
        
        val_acc = 100 * v_correct / v_total
        print(f"📊 Epoch {epoch+1} | Train: {100.*correct/total:.2f}% | Val: {val_acc:.2f}%")
        
        history.append({"epoch": epoch+1, "val_acc": val_acc})
        pd.DataFrame(history).to_csv(log_path, index=False)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"💾 find better model！the highest record now: {val_acc:.2f}%")
            early_stop_counter = 0 
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            print("🛑 early stop. V3 ends")
            break
        scheduler.step()

if __name__ == "__main__":
    train()