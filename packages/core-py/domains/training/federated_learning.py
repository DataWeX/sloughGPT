"""
Federated Learning Module for SloughGPT
=======================================

Enables on-device model training with weight synchronization:

1. Clients train local model on their conversations
2. Clients send weight deltas (not full weights) to server
3. Server aggregates deltas from multiple clients
4. Server fine-tunes base model with aggregated knowledge
5. Updated model weights distributed back to clients

This creates a continuous learning loop where the model
improves from all users' conversations without sharing raw data.
"""

from domains.training.slonet_compat import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
import time


@dataclass
class WeightDelta:
    """Represents a weight update from a client."""
    client_id: str
    timestamp: datetime
    layer_name: str
    old_weights: List[float]
    new_weights: List[float]
    learning_rate: float = 0.001
    training_samples: int = 0  # Number of examples trained on
    loss: Optional[float] = None
    
    @property
    def delta(self) -> List[float]:
        """Compute the weight change."""
        return [new - old for old, new in zip(self.old_weights, self.new_weights)]
    
    @property
    def delta_magnitude(self) -> float:
        """L2 norm of the weight change."""
        d = self.delta
        return np.sqrt(sum(x * x for x in d))
    
    def to_dict(self) -> Dict:
        return {
            'client_id': self.client_id,
            'timestamp': self.timestamp.isoformat(),
            'layer_name': self.layer_name,
            'delta': self.delta,
            'learning_rate': self.learning_rate,
            'training_samples': self.training_samples,
            'loss': self.loss,
            'delta_magnitude': self.delta_magnitude,
        }


@dataclass 
class ClientUpdate:
    """A complete update from a client containing multiple layer deltas."""
    client_id: str
    timestamp: datetime
    model_version: str  # Version of base model client is using
    layer_deltas: List[WeightDelta]
    total_training_samples: int = 0
    client_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def aggregated_delta(self) -> Dict[str, List[float]]:
        """Aggregate deltas across all layers."""
        aggregated = {}
        for delta in self.layer_deltas:
            aggregated[delta.layer_name] = delta.delta
        return aggregated
    
    def to_dict(self) -> Dict:
        return {
            'client_id': self.client_id,
            'timestamp': self.timestamp.isoformat(),
            'model_version': self.model_version,
            'layer_deltas': [d.to_dict() for d in self.layer_deltas],
            'total_training_samples': self.total_training_samples,
            'client_metadata': self.client_metadata,
        }


class FederatedAggregator:
    """
    Aggregates weight updates from multiple clients using FedAvg.
    
    FedAvg (Federated Averaging):
    1. Clients compute gradients on local data
    2. Server receives weighted gradients
    3. Server computes weighted average
    4. Server updates global model
    """
    
    def __init__(self, model, aggregation_strategy='fedavg'):
        self.model = model
        self.aggregation_strategy = aggregation_strategy
        
        # Client updates storage
        self.pending_updates: List[ClientUpdate] = []
        
        # Aggregated weights (averaged across clients)
        self.global_weights: Dict[str, torch.Tensor] = {}
        
        # Version tracking
        self.global_model_version = 1
        self.last_aggregation = None
        
        # Client registry
        self.registered_clients: Dict[str, Dict] = {}
        
        # Metrics
        self.metrics = {
            'total_updates_received': 0,
            'updates_by_client': {},
            'aggregation_rounds': 0,
            'avg_client_contribution': 0,
        }
    
    def register_client(self, client_id: str, metadata: Dict = None) -> str:
        """Register a new client and return registration token."""
        token = hashlib.sha256(f"{client_id}{time.time()}".encode()).hexdigest()[:16]
        self.registered_clients[client_id] = {
            'token': token,
            'registered_at': datetime.now(),
            'last_update': None,
            'metadata': metadata or {},
            'update_count': 0,
        }
        logger.info(f"Registered client: {client_id}")
        return token
    
    def receive_update(self, update: ClientUpdate) -> bool:
        """Receive and validate a weight update from a client."""
        # Validate client is registered
        if update.client_id not in self.registered_clients:
            logger.warning(f"Update from unregistered client: {update.client_id}")
            return False
        
        # Store update
        self.pending_updates.append(update)
        self.registered_clients[update.client_id]['last_update'] = datetime.now()
        self.registered_clients[update.client_id]['update_count'] += 1
        self.metrics['total_updates_received'] += 1
        
        logger.info(f"Received update from {update.client_id}: {len(update.layer_deltas)} layers")
        return True
    
    def aggregate(self, min_updates: int = 1) -> Optional[Dict[str, torch.Tensor]]:
        """
        Aggregate pending updates using FedAvg.
        
        Args:
            min_updates: Minimum number of updates required to aggregate
            
        Returns:
            New global weights if successful, None otherwise
        """
        if len(self.pending_updates) < min_updates:
            logger.info(f"Not enough updates: {len(self.pending_updates)}/{min_updates}")
            return None
        
        logger.info(f"Aggregating {len(self.pending_updates)} client updates...")
        
        # Collect all layer names
        layer_names = set()
        for update in self.pending_updates:
            for delta in update.layer_deltas:
                layer_names.add(delta.layer_name)
        
        # Aggregate each layer
        aggregated = {}
        
        for layer_name in layer_names:
            # Get all deltas for this layer
            layer_deltas = []
            weights = []
            
            for update in self.pending_updates:
                for delta in update.layer_deltas:
                    if delta.layer_name == layer_name:
                        # Weight by number of training samples
                        weight = update.total_training_samples or 1
                        layer_deltas.append({
                            'delta': delta.delta,
                            'weight': weight,
                        })
                        weights.append(weight)
            
            # Compute weighted average
            total_weight = sum(w['weight'] for w in layer_deltas)
            
            if total_weight == 0:
                continue
                
            # Weighted average of deltas
            avg_delta = [
                sum(d['delta'][i] * d['weight'] for d in layer_deltas) / total_weight
                for i in range(len(layer_deltas[0]['delta']))
            ]
            
            aggregated[layer_name] = avg_delta
        
        # Apply aggregated delta to global model
        self._apply_delta_to_model(aggregated)
        
        # Update metrics
        self.metrics['aggregation_rounds'] += 1
        self.metrics['avg_client_contribution'] = len(self.pending_updates)
        
        # Clear pending updates
        self.pending_updates = []
        
        # Increment version
        self.global_model_version += 1
        self.last_aggregation = datetime.now()
        
        logger.info(f"Aggregation complete. New version: {self.global_model_version}")
        
        return self.global_weights.copy()
    
    def _apply_delta_to_model(self, delta: Dict[str, List[float]]):
        """Apply aggregated delta to global model weights."""
        # This is a simplified version
        # In reality, we'd load the model and update specific layers
        self.global_weights = delta
    
    def get_model_update(self, current_version: int) -> Optional[Dict]:
        """
        Get model update for a client.
        
        Returns the full model weights if newer version available.
        """
        if current_version < self.global_model_version:
            return {
                'version': self.global_model_version,
                'weights': self.global_weights,
                'is_full_update': True,  # Send full weights, not just delta
            }
        return None
    
    def get_status(self) -> Dict:
        """Get aggregator status."""
        return {
            'global_model_version': self.global_model_version,
            'pending_updates': len(self.pending_updates),
            'registered_clients': len(self.registered_clients),
            'last_aggregation': self.last_aggregation.isoformat() if self.last_aggregation else None,
            'metrics': self.metrics,
        }


class WeightSerializer:
    """Serialize/deserialize model weights for transmission."""
    
    @staticmethod
    def serialize_weights(state_dict: Dict[str, torch.Tensor]) -> Dict:
        """Convert PyTorch tensors to JSON-serializable format."""
        serialized = {}
        for name, tensor in state_dict.items():
            serialized[name] = {
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'data': tensor.cpu().numpy().flatten().tolist(),
            }
        return serialized
    
    @staticmethod
    def deserialize_weights(serialized: Dict) -> Dict[str, torch.Tensor]:
        """Convert serialized weights back to PyTorch tensors."""
        weights = {}
        for name, data in serialized.items():
            arr = np.array(data['data'], dtype=np.float32)
            weights[name] = torch.from_numpy(arr).reshape(data['shape'])
        return weights
    
    @staticmethod
    def compute_delta(old: torch.Tensor, new: torch.Tensor) -> List[float]:
        """Compute weight delta between two tensors."""
        return (new - old).cpu().numpy().flatten().tolist()
    
    @staticmethod
    def apply_delta(tensor: torch.Tensor, delta: List[float], lr: float = 1.0) -> torch.Tensor:
        """Apply a delta to a tensor."""
        delta_tensor = torch.tensor(delta, dtype=tensor.dtype).reshape(tensor.shape)
        return tensor + lr * delta_tensor


# Logging setup
import logging
logger = logging.getLogger("federated")
