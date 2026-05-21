"""
Direct Model Loader - Load .pt checkpoints into chat
"""
from pathlib import Path
from domains.training.slonet_compat import torch
from typing import Optional, Dict, Any


def load_pt_checkpoint(path: str) -> tuple:
    """Load .pt checkpoint, return (model, tokenizer, info)."""
    import torch
    
    pt_path = Path(path)
    if not pt_path.exists():
        return None, None, {"error": f"File not found: {path}"}
    
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    
    model = ckpt.get('model')
    tokenizer = ckpt.get('tokenizer', {})
    total_steps = ckpt.get('total_steps', 0)
    training_log = ckpt.get('training_log', [])
    
    # Get vocab
    stoi = tokenizer.get('stoi', {})
    itos = tokenizer.get('itos', {})
    
    # Build encode/decode
    chars = list(stoi.keys()) if stoi else list(range(len(itos)))
    
    def encode(text):
        return [stoi.get(c, 0) for c in text[:200]]
    
    def decode(tokens):
        return ''.join([itos.get(t, '?') for t in tokens])
    
    info = {
        "steps": total_steps,
        "vocab_size": len(chars),
        "train_loss": training_log[-1] if training_log else None,
    }
    
    return {"encode": encode, "decode": decode, "model": model}, tokenizer, info


class DirectModelLoader:
    """Direct loader for trained .pt checkpoints."""
    
    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.current_info = {}
        self.checkpoint_dir = Path("models/auto-training")
    
    def load(self, checkpoint_name: str) -> Dict[str, Any]:
        """Load checkpoint by name."""
        # Try .pt first
        for ext in (".pt",):
            path = self.checkpoint_dir / (checkpoint_name + ext if not checkpoint_name.endswith(ext) else checkpoint_name)
            if not path.exists():
                path = self.checkpoint_dir / checkpoint_name
            
            if path.exists() and path.suffix == ".pt":
                return self._load_pt(path)
        
        return {"error": f"Checkpoint not found: {checkpoint_name}"}
    
    def _load_pt(self, path) -> Dict[str, Any]:
        """Load .pt file."""
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            
            model = ckpt.get('model')
            tokenizer = ckpt.get('tokenizer', {})
            
            stoi = tokenizer.get('stoi', {})
            itos = tokenizer.get('itos', {})
            
            def encode(text):
                return [stoi.get(c, 0) for c in text[:200]]
            
            def decode(tokens):
                return ''.join([itos.get(t, '?') for t in tokens])
            
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_info = {
                "name": path.name,
                "steps": ckpt.get('total_steps', 0),
                "vocab_size": len(stoi),
                "loaded": True,
            }
            
            return {"status": "loaded", "info": self.current_info}
            
        except Exception as e:
            return {"error": str(e)}
    
    def generate(self, text: str, max_tokens: int = 30, temperature: float = 0.8) -> str:
        """Generate response using loaded checkpoint."""
        if not self.current_model:
            return "[No model loaded]"
        
        # Encode
        tokens = [self.current_tokenizer.get('stoi', {}).get(c, 0) for c in text[:200]]
        
        # For now, return - actual generation needs full model forward
        # This is a placeholder - real inference would use model.generate()
        return f"[Using checkpoint: {self.current_info.get('name', 'unknown')}]"


# Global loader
_loader: Optional[DirectModelLoader] = None


def get_direct_loader() -> DirectModelLoader:
    """Get direct model loader."""
    global _loader
    if _loader is None:
        _loader = DirectModelLoader()
    return _loader


__all__ = ["DirectModelLoader", "get_direct_loader", "load_pt_checkpoint"]