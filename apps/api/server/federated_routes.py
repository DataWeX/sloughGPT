"""
Federated Learning API Endpoints
================================

Adds these endpoints to sloughgpt server:

POST /federated/register
  - Register a new client device

POST /federated/update
  - Receive weight updates from clients
  
GET /federated/model
  - Get latest model weights (polled by clients)

GET /federated/status
  - Get aggregator status

POST /federated/aggregate
  - Trigger manual aggregation (admin only)
"""

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime
import hashlib
import json

router = APIRouter(prefix="/federated", tags=["federated"])


# In-memory storage (in production, use Redis/DB)
class Storage:
    clients = {}
    updates = []
    global_weights = None
    global_version = 1
    
    @classmethod
    def reset(cls):
        cls.clients = {}
        cls.updates = []
        cls.global_weights = None
        cls.global_version = 1


# ============ Request/Response Models ============

class ClientRegistration(BaseModel):
    client_id: str
    device_info: Optional[Dict[str, Any]] = None
    current_model_version: Optional[int] = None


class ClientRegistrationResponse(BaseModel):
    client_id: str
    token: str
    registered: bool


class LayerDelta(BaseModel):
    layer_name: str
    old_weights: List[float]
    new_weights: List[float]
    learning_rate: Optional[float] = 0.001
    training_samples: Optional[int] = 0
    loss: Optional[float] = None


class WeightUpdate(BaseModel):
    client_id: str
    token: str
    model_version: int
    layer_deltas: List[LayerDelta]
    total_training_samples: Optional[int] = 0
    metadata: Optional[Dict[str, Any]] = None


class WeightUpdateResponse(BaseModel):
    received: bool
    update_id: str
    pending_updates: int


class ModelUpdateResponse(BaseModel):
    version: int
    weights: Dict[str, Any]
    is_update_available: bool


class AggregatorStatus(BaseModel):
    global_version: int
    pending_updates: int
    registered_clients: int
    last_aggregation: Optional[str]


# ============ Endpoints ============

@router.post("/register", response_model=ClientRegistrationResponse)
async def register_client(registration: ClientRegistration):
    """Register a new client device for federated learning."""
    
    client_id = registration.client_id
    
    if client_id in Storage.clients:
        # Return existing token
        return ClientRegistrationResponse(
            client_id=client_id,
            token=Storage.clients[client_id]['token'],
            registered=True
        )
    
    # Generate token
    token = hashlib.sha256(
        f"{client_id}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:16]
    
    Storage.clients[client_id] = {
        'token': token,
        'registered_at': datetime.now(),
        'device_info': registration.device_info,
        'last_update': None,
        'update_count': 0,
    }
    
    return ClientRegistrationResponse(
        client_id=client_id,
        token=token,
        registered=True
    )


@router.post("/update", response_model=WeightUpdateResponse)
async def receive_update(update: WeightUpdate):
    """Receive weight update from a client."""
    
    # Validate client
    if update.client_id not in Storage.clients:
        raise HTTPException(status_code=401, detail="Unregistered client")
    
    if Storage.clients[update.client_id]['token'] != update.token:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Store update
    update_id = hashlib.sha256(
        f"{update.client_id}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]
    
    Storage.updates.append({
        'id': update_id,
        'client_id': update.client_id,
        'model_version': update.model_version,
        'layer_deltas': update.layer_deltas,
        'total_training_samples': update.total_training_samples,
        'metadata': update.metadata,
        'received_at': datetime.now(),
    })
    
    # Update client status
    Storage.clients[update.client_id]['last_update'] = datetime.now()
    Storage.clients[update.client_id]['update_count'] += 1
    
    return WeightUpdateResponse(
        received=True,
        update_id=update_id,
        pending_updates=len(Storage.updates)
    )


@router.get("/model", response_model=ModelUpdateResponse)
async def get_model_update(
    client_id: str,
    current_version: int = 1
):
    """Get model update if available."""
    
    is_update_available = current_version < Storage.global_version
    
    if not is_update_available:
        return ModelUpdateResponse(
            version=Storage.global_version,
            weights={},
            is_update_available=False
        )
    
    return ModelUpdateResponse(
        version=Storage.global_version,
        weights=Storage.global_weights or {},
        is_update_available=True
    )


@router.get("/status", response_model=AggregatorStatus)
async def get_status():
    """Get federated learning aggregator status."""
    
    last_agg = None
    if Storage.updates:
        # Find most recent aggregation time
        for u in Storage.updates:
            if 'aggregated_at' in u:
                if last_agg is None or u['aggregated_at'] > last_agg:
                    last_agg = u['aggregated_at']
    
    return AggregatorStatus(
        global_version=Storage.global_version,
        pending_updates=len(Storage.updates),
        registered_clients=len(Storage.clients),
        last_aggregation=last_agg.isoformat() if last_agg else None
    )


@router.post("/aggregate")
async def trigger_aggregation():
    """Trigger weight aggregation (admin endpoint)."""
    
    if len(Storage.updates) == 0:
        return {"message": "No pending updates to aggregate"}
    
    # Simple FedAvg implementation
    # Aggregate weight deltas across all clients
    aggregated = {}
    
    for update in Storage.updates:
        weight = update.get('total_training_samples', 1) or 1
        
        for layer_delta in update['layer_deltas']:
            layer_name = layer_delta.layer_name
            
            # Compute delta
            old = layer_delta.old_weights
            new = layer_delta.new_weights
            delta = [n - o for o, n in zip(old, new)]
            
            if layer_name not in aggregated:
                aggregated[layer_name] = {'sum': delta, 'total_weight': weight}
            else:
                # Weighted average
                current = aggregated[layer_name]
                current['total_weight'] += weight
                current['sum'] = [
                    current['sum'][i] * (current['total_weight'] - weight) / current['total_weight']
                    + delta[i] * weight / current['total_weight']
                    for i in range(len(delta))
                ]
    
    # Update global model
    Storage.global_weights = {k: v['sum'] for k, v in aggregated.items()}
    Storage.global_version += 1
    
    # Mark updates as aggregated
    for u in Storage.updates:
        u['aggregated_at'] = datetime.now()
    Storage.updates = []
    
    return {
        "message": f"Aggregated {len(aggregated)} layers",
        "new_version": Storage.global_version,
        "layers_updated": list(aggregated.keys())
    }


@router.delete("/reset")
async def reset_federated():
    """Reset federated learning state (admin endpoint)."""
    Storage.reset()
    return {"message": "Federated learning state reset"}
