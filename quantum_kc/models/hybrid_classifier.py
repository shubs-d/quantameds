import math
import torch
import torch.nn as nn
from typing import Optional

from .tabular_encoder import TabularMLPEncoder
from .quantum_circuit import create_vqc_torch_layer

class HybridQuantumClassifier(nn.Module):
    """
    Stage 2 hybrid model combining a frozen CNN encoder with a VQC.
    """
    def __init__(self, encoder: nn.Module, n_qubits: int = 8, n_layers: int = 2, 
                 n_classes: int = 2, freeze_encoder: bool = True, 
                 dev_name: str = 'lightning.qubit', diff_method: str = 'adjoint'):
        """
        Initializes the HybridQuantumClassifier.

        Args:
            encoder (nn.Module): The image encoder module (e.g., CornealEncoder).
            n_qubits (int): Number of qubits in the VQC.
            n_layers (int): Number of layers in the VQC ansatz.
            n_classes (int): Number of output classes.
            freeze_encoder (bool): Whether to freeze the encoder's parameters.
            dev_name (str): PennyLane device name for the VQC.
            diff_method (str): Differentiation method for the VQC.
        """
        super().__init__()
        self.encoder = encoder
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # Assume encoder output matches the first dim, if not, might need a dynamic way, 
        # but encoder outputs latent_dim. We'll find latent_dim from encoder if possible.
        # Assuming the encoder returns an 8-dim latent vector, but we add a linear scale 
        # layer just in case, mapping whatever to n_qubits.
        # We will assume latent_dim is equal to n_qubits for simplicity, 
        # or we just map n_qubits to n_qubits.
        self.scaling_layer = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.Sigmoid()
        )
        
        self.vqc_layer = create_vqc_torch_layer(
            n_qubits=n_qubits, 
            n_layers=n_layers, 
            dev_name=dev_name, 
            diff_method=diff_method
        )
        
        self.classifier_head = nn.Linear(n_qubits, n_classes)

    def forward(self, x_image: torch.Tensor, x_tabular: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the hybrid model.

        Args:
            x_image (torch.Tensor): Image input.
            x_tabular (torch.Tensor, optional): Tabular input (not heavily used in this class).

        Returns:
            torch.Tensor: Logits of shape (batch, n_classes).
        """
        z = self.encoder(x_image)
        z_scaled = self.scaling_layer(z) * math.pi
        
        if x_tabular is not None:
            # Simple concatenation and linear mixing if x_tabular was pre-encoded
            # but usually MultimodalHybridClassifier handles fusion better.
            pass
            
        q_out = self.vqc_layer(z_scaled)
        logits = self.classifier_head(q_out)
        return logits


class MultimodalHybridClassifier(nn.Module):
    """
    Multimodal hybrid model combining image and tabular pathways with a VQC.
    """
    def __init__(self, encoder: nn.Module, tabular_input_dim: int, 
                 n_qubits: int = 8, n_layers: int = 2, n_classes: int = 2, 
                 freeze_encoder: bool = True, dev_name: str = 'lightning.qubit', 
                 diff_method: str = 'adjoint'):
        """
        Initializes the MultimodalHybridClassifier.

        Args:
            encoder (nn.Module): The image encoder module.
            tabular_input_dim (int): Number of features in tabular input.
            n_qubits (int): Number of qubits in the VQC.
            n_layers (int): Number of layers in the VQC ansatz.
            n_classes (int): Number of output classes.
            freeze_encoder (bool): Whether to freeze the image encoder's parameters.
            dev_name (str): PennyLane device name for the VQC.
            diff_method (str): Differentiation method for the VQC.
        """
        super().__init__()
        self.encoder = encoder
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        self.tabular_encoder = TabularMLPEncoder(input_dim=tabular_input_dim, latent_dim=n_qubits)
        
        # Scaling for image latent
        self.scaling_layer = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.Sigmoid()
        )
        
        self.vqc_layer = create_vqc_torch_layer(
            n_qubits=n_qubits, 
            n_layers=n_layers, 
            dev_name=dev_name, 
            diff_method=diff_method
        )
        
        self.classifier_head = nn.Linear(n_qubits, n_classes)

    def forward(self, x_image: torch.Tensor, x_tabular: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multimodal hybrid model.

        Args:
            x_image (torch.Tensor): Image input.
            x_tabular (torch.Tensor): Tabular input.

        Returns:
            torch.Tensor: Logits of shape (batch, n_classes).
        """
        z_img = self.encoder(x_image)
        z_img_scaled = self.scaling_layer(z_img) * math.pi
        
        z_tab = self.tabular_encoder(x_tabular)
        
        # Average fusion
        z_fused = (z_img_scaled + z_tab) / 2.0
        
        q_out = self.vqc_layer(z_fused)
        logits = self.classifier_head(q_out)
        return logits
