import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB
from collections import defaultdict
import random

def get_dataloaders(batch_size=32):
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    
    full_train_data = GTSRB(root=data_path, split='train', transform=transform)
    test_data = GTSRB(root=data_path, split='test', transform=transform)

    
    track_dict = defaultdict(list)
    
    for idx, (path, target) in enumerate(full_train_data._samples):
      
        filename = os.path.basename(path)
        track_id = filename.split('_')[0] 
        
        
        unique_track_id = f"{target}_{track_id}"
        track_dict[unique_track_id].append(idx)

    
    train_indices = []
    val_indices = []
    
   
    random.seed(42)
    all_tracks = list(track_dict.values())
    random.shuffle(all_tracks)
    
    split_idx = int(len(all_tracks) * 0.8)
    
    for track_images in all_tracks[:split_idx]:
        train_indices.extend(track_images)
    for track_images in all_tracks[split_idx:]:
        val_indices.extend(track_images)

   
    train_subset = Subset(full_train_data, train_indices)
    val_subset = Subset(full_train_data, val_indices)

   
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, 43 
    
if __name__ == "__main__":
    train_loader, val_loader, test_loader, num_classes = get_dataloaders()
    print(f"\nSuccess! Data leakage prevented and dataset successfully split.")
    print(f"Number of images allocated for Training: {len(train_loader.dataset)}")
    print(f"Number of images allocated for Validation: {len(val_loader.dataset)}\n")