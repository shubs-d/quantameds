import pennylane as qml
import torch

def create_vqc_qnode(n_qubits: int = 8, n_layers: int = 2, dev_name: str = 'lightning.qubit', diff_method: str = 'adjoint') -> qml.QNode:
    """
    Creates a PennyLane QNode for a Variational Quantum Classifier (VQC).
    
    This circuit uses:
    - AngleEmbedding: Maps classical features to qubit rotations efficiently.
    - StronglyEntanglingLayers: Provides trainable entanglement. Shallow depth (e.g., 2 layers) 
      avoids barren plateaus and provides good trainability.
    - Adjoint differentiation is efficient for statevector simulators like lightning.qubit.

    Args:
        n_qubits (int): Number of qubits.
        n_layers (int): Number of layers in the StronglyEntanglingLayers.
        dev_name (str): PennyLane device name.
        diff_method (str): Differentiation method.

    Returns:
        qml.QNode: The quantum node interface for torch.
    """
    dev = qml.device(dev_name, wires=n_qubits)
    
    @qml.qnode(dev, interface='torch', diff_method=diff_method)
    def qnode(inputs, weights):
        # Data encoding: classical data to quantum state
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
        
        # Trainable ansatz
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        
        # Measurement: expectation value in Z basis for each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
    return qnode

def create_vqc_torch_layer(n_qubits: int = 8, n_layers: int = 2, dev_name: str = 'lightning.qubit', diff_method: str = 'adjoint') -> qml.qnn.TorchLayer:
    """
    Creates a PyTorch-compatible TorchLayer for the VQC.

    Args:
        n_qubits (int): Number of qubits.
        n_layers (int): Number of layers in the StronglyEntanglingLayers.
        dev_name (str): PennyLane device name.
        diff_method (str): Differentiation method.

    Returns:
        qml.qnn.TorchLayer: The quantum neural network layer.
    """
    qnode = create_vqc_qnode(
        n_qubits=n_qubits, 
        n_layers=n_layers, 
        dev_name=dev_name, 
        diff_method=diff_method
    )
    
    weight_shapes = {'weights': (n_layers, n_qubits, 3)}
    return qml.qnn.TorchLayer(qnode, weight_shapes)
