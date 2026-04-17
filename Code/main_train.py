import os
import torch
import torch.nn as nn
import gc
import pandas as pd
from tqdm.auto import tqdm
from data_utils import get_loaders
from model_factory import get_model

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt)**self.gamma * ce_loss).mean()

# modify the learning rate and patience
DATA_DIR, BATCH_SIZE, LR, EPOCHS = "LIDC-IDRI-slices", 16, 2e-5, 20 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    gc.collect(); torch.cuda.empty_cache()
    train_loader, val_loader = get_loaders(DATA_DIR, BATCH_SIZE)
    model = get_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = FocalLoss()

    SAVE_DIR = "Checkpoints_Ultimate_80"
    os.makedirs(SAVE_DIR, exist_ok=True)
    best_path, latest_path = f"{SAVE_DIR}/best_model.pth", f"{SAVE_DIR}/latest_checkpoint.pth"
    log_path = f"{SAVE_DIR}/ultimate_log.csv"
    
    start_epoch, best_acc, history = 0, 0.0, []
    patience, early_stop_counter = 5, 0 # 延长耐心

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        correct, total, total_loss = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs); loss = criterion(out, labels)
            loss.backward(); optimizer.step()
            _, pred = torch.max(out, 1); total += labels.size(0); correct += (pred == labels).sum().item()
            total_loss += loss.item()
            pbar.set_postfix(acc=f"{100.*correct/total:.2f}%")

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                v_out = model(imgs); _, v_pred = torch.max(v_out, 1)
                v_total += labels.size(0); v_correct += (v_pred == labels).sum().item()
        
        val_acc = 100 * v_correct / v_total
        print(f"\n📊 Epoch {epoch+1} : Val Acc: {val_acc:.2f}%")
        
        history.append({"epoch": epoch+1, "train_acc": 100.*correct/total, "val_acc": val_acc})
        pd.DataFrame(history).to_csv(log_path, index=False)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"new record！the highest record now: {val_acc:.2f}%")
            early_stop_counter = 0 
        else:
            early_stop_counter += 1
            print(f" warning: validation dataset have not been up for {early_stop_counter} round.")
            
        if early_stop_counter >= patience:
            print(f"🛑 early stop, save the best model and end.")
            break
        
        scheduler.step()
        torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_acc': max(val_acc, best_acc)}, latest_path)

if __name__ == "__main__":
    train()
    
    
  