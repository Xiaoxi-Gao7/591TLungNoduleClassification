import os
import glob
import cv2
import torch
import numpy as np
import re
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import GroupShuffleSplit


def get_patient_id(path):
    match = re.search(r"LIDC-IDRI-\d+", path)
    return match.group(0) if match else "unknown"


# (V2/V3) single path
# 128x128 Resize to 224x224

class LIDC_HardMode_Dataset(Dataset):
    def __init__(self, paths, labels, is_train=False):
        self.paths = paths
        self.labels = labels
        self.is_train = is_train
        self.transform = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip() if is_train else T.Lambda(lambda x: x),
            T.RandomVerticalFlip() if is_train else T.Lambda(lambda x: x),
            T.RandomRotation(15) if is_train else T.Lambda(lambda x: x),
            T.ColorJitter(brightness=0.1, contrast=0.1) if is_train else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.paths)

    def get_nodule_center(self, img_path):
        base = os.path.dirname(os.path.dirname(img_path))
        name = os.path.basename(img_path)
        centers = []
        for i in range(4):
            m_path = os.path.join(base, f"mask-{i}", name)
            if os.path.exists(m_path):
                m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
                if m is not None and np.max(m) > 0:
                    M = cv2.moments(m)
                    if M["m00"] != 0:
                        centers.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))
        return np.mean(centers, axis=0).astype(int) if centers else (256, 256)

    def __getitem__(self, idx):
        img_path, label = self.paths[idx], self.labels[idx]
        img_dir, img_name = os.path.dirname(img_path), os.path.basename(img_path)
        try: img_idx = int(img_name.replace('.png', ''))
        except: img_idx = 0
            
        slices = []
        for offset in [-1, 0, 1]:
            target_path = os.path.join(img_dir, f"{str(img_idx + offset).zfill(3)}.png")
            s = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(target_path) else None
            if s is None: s = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            slices.append(s)
        img_25d = cv2.merge(slices)

        h, w = img_25d.shape[:2]
        crop_size = 128
        if label == 1:
            cx, cy = self.get_nodule_center(img_path)
            if self.is_train:
                cx += random.randint(-8, 8)
                cy += random.randint(-8, 8)
            y1 = max(0, min(cy - crop_size//2, h - crop_size))
            x1 = max(0, min(cx - crop_size//2, w - crop_size))
            img_crop = img_25d[y1:y1+crop_size, x1:x1+crop_size]
        else:
            low_y, high_y = h//4, 3*h//4 - crop_size
            low_x, high_x = w//4, 3*w//4 - crop_size
            if high_y > low_y and high_x > low_x:
                start_y, start_x = random.randint(low_y, high_y), random.randint(low_x, high_x)
            else:
                start_y, start_x = max(0, (h - crop_size) // 2), max(0, (w - crop_size) // 2)
            img_crop = img_25d[start_y:start_y+crop_size, start_x:start_x+crop_size]

        img_final = cv2.resize(img_crop, (224, 224))
        return self.transform(img_final), torch.tensor(label).long()


#V4 dual path
class LIDC_V4_MultiScale_Dataset(Dataset):
    def __init__(self, paths, labels, is_train=False):
        self.paths = paths
        self.labels = labels
        self.is_train = is_train
        self.transform = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip() if is_train else T.Lambda(lambda x: x),
            T.RandomVerticalFlip() if is_train else T.Lambda(lambda x: x),
            T.RandomRotation(15) if is_train else T.Lambda(lambda x: x),
            T.ColorJitter(brightness=0.1, contrast=0.1) if is_train else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.paths)

    def get_nodule_center(self, img_path):
        base = os.path.dirname(os.path.dirname(img_path))
        name = os.path.basename(img_path)
        centers = []
        for i in range(4):
            m_path = os.path.join(base, f"mask-{i}", name)
            if os.path.exists(m_path):
                m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
                if m is not None and np.max(m) > 0:
                    M = cv2.moments(m)
                    if M["m00"] != 0:
                        centers.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))
        return np.mean(centers, axis=0).astype(int) if centers else (256, 256)

    def __getitem__(self, idx):
        img_path, label = self.paths[idx], self.labels[idx]
        img_dir, img_name = os.path.dirname(img_path), os.path.basename(img_path)
        
        # 2.5D 
        slices = []
        try: img_idx = int(img_name.replace('.png', ''))
        except: img_idx = 0
        for offset in [-1, 0, 1]:
            target_path = os.path.join(img_dir, f"{str(img_idx + offset).zfill(3)}.png")
            s = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(target_path) else None
            if s is None: s = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            slices.append(s)
        img_25d = cv2.merge(slices)
        h, w = img_25d.shape[:2]

        # pixel
        if label == 1:
            cx, cy = self.get_nodule_center(img_path)
            if self.is_train:
                cx += random.randint(-5, 5)
                cy += random.randint(-5, 5)
        else:
            cx, cy = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)

        # 128x128 
        y1, x1 = max(0, cy-64), max(0, cx-64)
        crop_l = img_25d[y1:min(y1+128, h), x1:min(x1+128, w)]
        crop_l = cv2.resize(crop_l, (128, 128))

        # 256x256 -> Resize to 128 
        y2, x2 = max(0, cy-128), max(0, cx-128)
        crop_g = img_25d[y2:min(y2+256, h), x2:min(x2+256, w)]
        crop_g = cv2.resize(crop_g, (128, 128))

        return (self.transform(crop_l), self.transform(crop_g)), torch.tensor(label).long()



def get_loaders(data_root, batch_size=32, mode="V4"):
    all_pngs = glob.glob(os.path.join(data_root, "**", "images", "*.png"), recursive=True)
    pos_paths, hard_neg_paths, easy_neg_paths = [], [], []
    
    print(f" (Mode: {mode})...")
    for p in all_pngs:
        base, name = os.path.dirname(os.path.dirname(p)), os.path.basename(p)
        votes = 0
        for i in range(4):
            m_path = os.path.join(base, f"mask-{i}", name)
            if os.path.exists(m_path):
                m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
                if m is not None and np.max(m) > 0: votes += 1
        
        if votes >= 3: pos_paths.append(p)
        elif votes == 2: hard_neg_paths.append(p)
        elif votes <= 1: easy_neg_paths.append(p)

    num_total_neg = len(pos_paths) * 2 
    num_hard = min(len(hard_neg_paths), int(num_total_neg * 0.4))
    num_easy = min(len(easy_neg_paths), num_total_neg - num_hard)

    print(f"📦 positive sample {len(pos_paths)} | hard negtive sample(2 vote) {num_hard} | normal negtive sample(0-1) {num_easy}")
    
    sampled_neg = random.sample(hard_neg_paths, num_hard) + random.sample(easy_neg_paths, num_easy)
    final_paths, final_labels = pos_paths + sampled_neg, [1]*len(pos_paths) + [0]*len(sampled_neg)
    
    groups = [get_patient_id(p) for p in final_paths]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(final_paths, final_labels, groups=groups))
    
    tr_p = [final_paths[i] for i in train_idx]; tr_l = [final_labels[i] for i in train_idx]
    val_p = [final_paths[i] for i in val_idx]; val_l = [final_labels[i] for i in val_idx]

    # choose mode kind
    if mode == "V4":
        train_ds = LIDC_V4_MultiScale_Dataset(tr_p, tr_l, True)
        val_ds = LIDC_V4_MultiScale_Dataset(val_p, val_l, False)
    else:
        train_ds = LIDC_HardMode_Dataset(tr_p, tr_l, True)
        val_ds = LIDC_HardMode_Dataset(val_p, val_l, False)

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0))
    

#check the dataset
if __name__ == "__main__":
    
    DATA_ROOT = r"C:\Users\xiaox\Desktop\BMEG_Project\LIDC-IDRI-slices"
    
    train_loader, val_loader = get_loaders(DATA_ROOT, batch_size=32, mode="V4")
    
    print("\n" + "="*40)
    print("Final Report Stats")
    print("="*40)
    print(f"🔹 the amount of slices in train dataset: {len(train_loader.dataset)}")
    print(f"🔹 the amount of slices in validation dataset: {len(val_loader.dataset)}")
    print(f"🔹 whole samples:   {len(train_loader.dataset) + len(val_loader.dataset)}")
    
    # patients
    train_pids = set(get_patient_id(p) for p in train_loader.dataset.paths)
    val_pids = set(get_patient_id(p) for p in val_loader.dataset.paths)
    print(f"👤 number of patients in train dataset:   {len(train_pids)}")
    print(f"👤 number of patients in validation dataset:   {len(val_pids)}")
    print("="*40)