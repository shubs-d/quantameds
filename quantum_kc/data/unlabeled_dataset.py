"""PyTorch Dataset for the unlabeled Orbscan IIz dataset."""

import os
import logging
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Tuple, Callable
from sklearn.preprocessing import MinMaxScaler
from .preprocessing import build_tabular_pipeline

logger = logging.getLogger(__name__)

class OrbscanUnlabeledDataset(Dataset):
    def __init__(
        self, 
        csv_path: str, 
        images_dir: str, 
        transform: Optional[Callable] = None, 
        tabular_scaler: Optional[MinMaxScaler] = None, 
        qc_filter: bool = True
    ):
        super().__init__()
        self.images_dir = images_dir
        self.transform = transform
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # QC Filter
        if qc_filter and 'QC_status' in df.columns:
            df = df[df['QC_status'] == 'PASS']
            logger.info(f"Filtered dataset by QC_status PASS, remaining: {len(df)}")
            
        self.df = df.reset_index(drop=True)
        
        # Store filenames
        self.filenames = self.df['filename'].tolist() if 'filename' in self.df.columns else []
        
        # Process tabular features
        angular_cols = [
            'SimK_Astig_Axis_deg', 'MaxK_Axis_deg', 'MinK_Axis_deg', 
            'Zone3mm_SteepAxis_deg', 'Zone3mm_FlatAxis_deg', 
            'Zone5mm_SteepAxis_deg', 'Zone5mm_FlatAxis_deg', 'Kappa_at_deg'
        ]
        
        skewed_cols = ['SimK_Astig_D', 'Zone3mm_Irreg_D', 'Zone5mm_Irreg_D']
        
        exclude_cols = ['filename', 'patient_id', 'eye', 'QC_status', 'date']
        numeric_cols = [col for col in self.df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(self.df[col])]
        
        tabular_data = self.df[numeric_cols]
        
        self.tabular_features, self.tabular_scaler = build_tabular_pipeline(
            tabular_data, 
            angular_cols=[c for c in angular_cols if c in numeric_cols], 
            skewed_cols=[c for c in skewed_cols if c in numeric_cols], 
            scaler=tabular_scaler
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.filenames:
            filename = f"image_{idx}.png"
        else:
            filename = self.filenames[idx]
            
        if not filename.endswith('.png'):
            filename = f"{filename}.png"
            
        img_path = os.path.join(self.images_dir, filename)
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            img = Image.new('RGB', (224, 224), color=0)
            
        if self.transform:
            img_tensor = self.transform(img)
        else:
            from torchvision.transforms import ToTensor
            img_tensor = ToTensor()(img)
            
        tab_tensor = torch.FloatTensor(self.tabular_features[idx])
        
        return img_tensor, tab_tensor
