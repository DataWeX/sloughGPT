"""
SloughGPT Unit Tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

try:
    from domains.models import SloughGPTModel
except (ImportError, ModuleNotFoundError):
    pytest.skip("domains.models not available", allow_module_level=True)


class TestSloughGPTModel:
    """Tests for SloughGPTModel."""
    
    def test_model_creation(self):
        """Test SloughGPTModel can be created."""
        from domains.models import SloughGPTModel
        
        model = SloughGPTModel(
            vocab_size=100,
            n_embed=64,
            n_layer=2,
            n_head=2,
            block_size=32
        )
        
        assert model is not None
        assert model.num_parameters() > 0
    
    def test_forward_pass(self):
        """Test forward pass."""
        from domains.models import SloughGPTModel
        
        model = SloughGPTModel(
            vocab_size=100,
            n_embed=64,
            n_layer=2,
            n_head=2,
            block_size=32
        )
        
        x = torch.randint(0, 100, (2, 10))
        logits, loss = model(x)
        
        assert logits.shape == (2, 10, 100)
    
    def test_generation(self):
        """Test text generation."""
        from domains.models import SloughGPTModel
        
        model = SloughGPTModel(
            vocab_size=100,
            n_embed=64,
            n_layer=2,
            n_head=2,
            block_size=32
        )
        
        idx = torch.tensor([[1]])
        output = model.generate(idx, max_new_tokens=10)
        
        assert output.shape[1] == 11


class TestSloughGPTModel:
    """Tests for SloughGPTModel - OUR architecture."""

    def test_model_creation(self):
        """Test SloughGPTModel can be created."""
        from domains.models import SloughGPTModel

        model = SloughGPTModel(
            vocab_size=100,
            n_embed=64,
            n_layer=2,
            n_head=4,
            block_size=32,
            dropout=0.0,
        )

        assert model is not None
        assert model.num_parameters() > 0
        assert model.config().get("model_type") == "sloughgpt"

    def test_forward_pass(self):
        """Test forward pass."""
        from domains.models import SloughGPTModel

        model = SloughGPTModel(
            vocab_size=100,
            n_embed=64,
            n_layer=2,
            n_head=4,
            block_size=32,
            dropout=0.0,
        )

        x = torch.randint(0, 100, (2, 10))
        logits, loss = model(x, targets=torch.randint(0, 100, (2, 10)))

        assert logits.shape == (2, 10, 100)
        assert loss is not None

    def test_generation(self):
        """Test text generation."""
        from domains.models import SloughGPTModel

        model = SloughGPTModel(
            vocab_size=100,
            n_embed=64,
            n_layer=2,
            n_head=4,
            block_size=32,
            dropout=0.0,
        )
        model.eval()

        idx = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=10)

        assert output.shape[1] >= 4  # At least input + 1 generated

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing."""
        from domains.models import SloughGPTModel

        model = SloughGPTModel(
            vocab_size=50,
            n_embed=32,
            n_layer=2,
            n_head=2,
            block_size=16,
            dropout=0.0,
        )

        model.apply_gradient_checkpointing()
        assert any(b.use_checkpoint for b in model.blocks)


class TestLoRA:
    """Tests for LoRA."""
    
    def test_apply_lora(self):
        """Test LoRA can be applied."""
        from domains.models import SloughGPTModel
        from domains.training.lora import apply_lora_to_model, LoRAConfig
        
        model = SloughGPTModel(
            vocab_size=100,
            n_embed=64,
            n_layer=2,
            n_head=2,
            block_size=32
        )
        
        lora_config = LoRAConfig(rank=4, alpha=16)
        model_lora = apply_lora_to_model(model, config=lora_config)
        
        assert model_lora is not None


class TestQuantization:
    """Tests for quantization."""
    
    def test_dynamic_quantization(self):
        """Test dynamic quantization (SloNet skips — PyTorch-specific)."""
        from domains.models import SloughGPTModel
        from domains.training.efficient_inference import Quantizer
        
        model = SloughGPTModel(
            vocab_size=50,
            n_embed=32,
            n_layer=2,
            n_head=2,
            block_size=16
        )
        
        # SloNet is pure NumPy — no PyTorch nn.Module internals.
        # Dynamic quantization requires torch.nn.Module._modules dict.
        if not hasattr(model, "_modules"):
            pytest.skip("SloNet model does not support PyTorch dynamic quantization")
        
        quantized = Quantizer.dynamic_quantize(model)
        
        assert quantized is not None


class TestPersonality:
    """Tests for personality system."""
    
    def test_list_personalities(self):
        """Test listing personalities."""
        from domains.ai_personality import list_personalities
        
        personalities = list_personalities()
        
        assert len(personalities) > 0
        assert any(p['name'] == 'Helpful' for p in personalities)
    
    def test_personality_manager(self):
        """Test personality manager."""
        from domains.ai_personality import PersonalityManager, PersonalityType
        
        manager = PersonalityManager()
        
        assert manager.current is not None
        
        manager.set_personality(PersonalityType.CREATIVE)
        
        assert manager.current.name == 'Creative'


class TestDatabase:
    """Tests for database."""
    
    def test_database_manager(self):
        """Test database manager."""
        from domains.infrastructure.db_manager import DatabaseManager
        
        db = DatabaseManager()
        
        assert db is not None
    
    def test_conversation_operations(self):
        """Test conversation CRUD (file-based fallback)."""
        from domains.infrastructure.db_manager import DatabaseManager
        
        db = DatabaseManager()
        db._connected = False  # use file-based fallback
        
        # Create conversation
        conv = db.create_conversation(name="test")
        
        assert conv['name'] == "test"
        
        # Get conversation
        retrieved = db.get_conversation(conv['id'])
        
        assert retrieved['id'] == conv['id']
        
        # Delete
        deleted = db.delete_conversation(conv['id'])
        
        assert deleted == True


class TestAPI:
    """Tests for API endpoints (current app)."""
    
    def test_health_endpoint(self):
        """Test health endpoint."""
        from apps.api.server.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        response = client.get("/health")
        
        assert response.status_code == 200
    
    def test_souls_endpoint(self):
        """Test souls endpoint (mapped from old /personalities)."""
        from apps.api.server.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        response = client.get("/souls")
        
        assert response.status_code == 200
        assert 'souls' in response.json()
        assert isinstance(response.json()['souls'], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
