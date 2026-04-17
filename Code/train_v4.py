import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm
from data_utils import get_loaders  
from model_factory_v4 import get_model_v4

# maintain FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt)**self.gamma * ce_loss).mean()

# modify parameters
DATA_DIR, BATCH_SIZE, LR, EPOCHS = "LIDC-IDRI-slices", 8, 3e-5, 25 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    
    train_loader, val_loader = get_loaders(DATA_DIR, BATCH_SIZE, mode="V4")
    model = get_model_v4().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = FocalLoss()

    SAVE_DIR = "Checkpoints_V4_Final_Sprint"
    os.makedirs(SAVE_DIR, exist_ok=True)
    best_path, log_path = f"{SAVE_DIR}/best_model_v4.pth", f"{SAVE_DIR}/v4_log.csv"
    
    best_acc, history = 0.0, []
    patience, early_stop_counter = 6, 0 

    print("🔥 V4 starts！")

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        
        for (imgs_l, imgs_g), labels in pbar:
            imgs_l, imgs_g, labels = imgs_l.to(DEVICE), imgs_g.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(imgs_l, imgs_g) # duel path
            loss = criterion(out, labels)
            loss.backward(); optimizer.step()
            
            _, pred = torch.max(out, 1); total += labels.size(0); correct += (pred == labels).sum().item()
            pbar.set_postfix(acc=f"{100.*correct/total:.2f}%")

        
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for (v_l, v_g), v_labels in val_loader:
                v_l, v_g, v_labels = v_l.to(DEVICE), v_g.to(DEVICE), v_labels.to(DEVICE)
                v_out = model(v_l, v_g)
                _, v_pred = torch.max(v_out, 1)
                v_total += v_labels.size(0); v_correct += (v_pred == v_labels).sum().item()
        
        val_acc = 100 * v_correct / v_total
        print(f"📊 Epoch {epoch+1} | Train: {100.*correct/total:.2f}% | Val: {val_acc:.2f}%")
        
        history.append({"epoch": epoch+1, "train_acc": 100.*correct/total, "val_acc": val_acc})
        pd.DataFrame(history).to_csv(log_path, index=False)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"💾 record update！The best Acc now: {val_acc:.2f}%")
            early_stop_counter = 0 
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            print("🛑 early stop. V4 ends.")
            break
        scheduler.step()

if __name__ == "__main__":
    train() 