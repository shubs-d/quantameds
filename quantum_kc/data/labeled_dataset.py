"""PyTorch Dataset for the CornOrb labeled dataset."""

import os
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from typing import Optional, Tuple, Callable
from .preprocessing import build_tabular_pipeline
from .image_transforms import load_and_stack_maps
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

class CornOrbDataset(Dataset):
    def __init__(
        self, 
        csv_path: str, 
        data_root: str, 
        split: str = 'train', 
        fold: Optional[int] = None, 
        transform: Optional[Callable] = None, 
        tabular_scaler: Optional[MinMaxScaler] = None
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.transform = transform
        
        # Read CSV
        df = pd.read_csv(csv_path, dtype={'patient_code': str})
        # Excel artifact: '3E132' was written as scientific notation '3E+132' in the CSV
        df['patient_code'] = df['patient_code'].astype(str).str.replace('3E+132', '3E132', regex=False)
        
        # Filter by split and fold
        if split == 'test':
            if 'split_80_20' in df.columns:
                df = df[df['split_80_20'] != 'train']
        elif split in ('train', 'val'):
            if 'split_80_20' in df.columns:
                df = df[df['split_80_20'] == 'train']
            if fold is not None and 'fold' in df.columns:
                if split == 'train':
                    df = df[df['fold'] != fold]
                elif split == 'val':
                    df = df[df['fold'] == fold]
        
        self.df = df.reset_index(drop=True)
        
        # Process tabular features
        angular_cols = ['astig_axis_deg', 'kmax_axis_deg']
        skewed_cols = ['astig_value_D']
        
        # Preserve original strings for image path resolution before encoding
        if 'eye' in self.df.columns:
            self.df['eye_str'] = self.df['eye'].copy()
            self.df['eye'] = self.df['eye'].map({'OD': 0, 'OS': 1, 'od': 0, 'os': 1})
        if 'gender' in self.df.columns:
            self.df['gender'] = self.df['gender'].map({'m': 0, 'f': 1, 'M': 0, 'F': 1})
            
        numeric_cols = [
            'kmax_value_D', 'pachy_central_um', 'pachy_thinnest_um', 
            'pachy_thinnest_x', 'pachy_thinnest_y', 'asphericity_anterior', 
            'asphericity_posterior', 'age_years', 'gender', 'eye',
            'astig_axis_deg', 'kmax_axis_deg', 'astig_value_D'
        ]
        
        existing_num_cols = [col for col in numeric_cols if col in self.df.columns]
        tabular_data = self.df[existing_num_cols]
        
        self.tabular_features, self.tabular_scaler = build_tabular_pipeline(
            tabular_data, 
            angular_cols=[c for c in angular_cols if c in existing_num_cols], 
            skewed_cols=[c for c in skewed_cols if c in existing_num_cols], 
            scaler=tabular_scaler
        )
        
        if 'label' in self.df.columns:
            # .copy() avoids PyTorch's non-writable tensor UserWarning
            self.labels = torch.LongTensor(self.df['label'].values.copy())
        else:
            self.labels = torch.zeros(len(self.df), dtype=torch.long)
            logger.warning("No 'label' column found in dataset, defaulting to zeros.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        patient_code = str(row['patient_code'])
        eye = str(row['eye_str']) if 'eye_str' in self.df.columns else ('OD' if row['eye'] == 0 else 'OS')

        # Image path: {data_root}/{patient_code}/{eye}/
        img_dir = os.path.join(self.data_root, patient_code, eye)

        try:
            img = load_and_stack_maps(img_dir, patient_code, eye)
        except Exception as e:
            logger.error(f"Error loading images for {patient_code} {eye}: {e}")
            img = torch.zeros((3, 224, 224))
            return img, torch.FloatTensor(self.tabular_features[idx]), self.labels[idx]

        if self.transform:
            img_tensor = self.transform(img)
        else:
            from torchvision.transforms import ToTensor
            img_tensor = ToTensor()(img)

        tab_tensor = torch.FloatTensor(self.tabular_features[idx])
        label = self.labels[idx]

        return img_tensor, tab_tensor, label
        
    def get_class_weights(self) -> torch.Tensor:
        """Get inverse frequency class weights for loss function."""
        class_counts = np.bincount(self.labels.numpy())
        total = len(self.labels)
        weights = total / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)
        
    def get_sampler(self) -> WeightedRandomSampler:
        """Get WeightedRandomSampler for balanced batching."""
        class_counts = np.bincount(self.labels.numpy())
        class_weights = 1. / class_counts
        sample_weights = class_weights[self.labels.numpy()]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return sampler
