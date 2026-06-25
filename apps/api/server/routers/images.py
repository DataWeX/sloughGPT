"""Image Generation Router - text-to-image generation with style selection."""

import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger("man.routers.images")
router = APIRouter(prefix="/images", tags=["images"])


# ── Styles ────────────────────────────────────────────────────────────────

STYLES = {
    "realistic": "Realistic",
    "cartoon": "Cartoon",
    "watercolor": "Watercolor",
    "sketch": "Sketch",
    "fantasy": "Fantasy",
}


# ── Schema ────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    style: Literal["realistic", "cartoon", "watercolor", "sketch", "fantasy"] = "realistic"


class GenerateResponse(BaseModel):
    image: str  # base64 data URL
    style: str
    prompt: str
    id: str


# ── Procedural Image Generators (Pillow-based, no external deps) ──────────


def _generate_gradient_image(prompt: str, width: int = 512, height: int = 512) -> bytes:
    """Generate a gradient-based image based on prompt keywords."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise HTTPException(status_code=500, detail="Pillow library required for image generation")
    
    # Extract keywords and map to colors
    prompt_lower = prompt.lower()
    colors = []
    
    if any(k in prompt_lower for k in ["cabin", "nature", "forest", "tree", "green"]):
        colors = ["#2d5a27", "#4a7c3e", "#6b9e5a"]  # forest greens
    elif any(k in prompt_lower for k in ["sunset", "evening", "sky", "orange", "red"]):
        colors = ["#d35400", "#e67e22", "#f39c12", "#f1c40f"]  # sunset orange
    elif any(k in prompt_lower for k in ["ocean", "water", "blue", "sea"]):
        colors = ["#1e3799", "#3a86ff", "#83b6ed", "#caf0f8"]  # ocean blue
    elif any(k in prompt_lower for k in ["mountain", "sky", "cloud", "white", "bright"]):
        colors = ["#3a86ff", "#83b6ed", "#caf0f8", "#ffffff"]  # sky white
    elif any(k in prompt_lower for k in ["fire", "flame", "heat", "red", "burn"]):
        colors = ["#b51700", "#e85d04", "#faa330", "#ffdd59"]  # fire
    else:
        colors = ["#6a11cb", "#2575fc"]  # default gradient
    
    # Convert hex to RGB
    def _c(c): return int(c.lstrip('#')[:2], 16)
    rgb_colors = [(_c(c), _c(c), _c(c)) for c in colors]
    # Actually parse properly
    rgb_colors = []
    for c in colors:
        h = c.lstrip('#')
        rgb_colors.append((int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)))

    # Create gradient image
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    
    for y in range(height):
        t = y / height
        idx = t * (len(rgb_colors) - 1)
        i = int(idx)
        f = idx - i
        if i >= len(rgb_colors) - 1:
            r, g, b = rgb_colors[-1]
        else:
            r = int(rgb_colors[i][0] * (1-f) + rgb_colors[i+1][0] * f)
            g = int(rgb_colors[i][1] * (1-f) + rgb_colors[i+1][1] * f)
            b = int(rgb_colors[i][2] * (1-f) + rgb_colors[i+1][2] * f)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # Add subtle pattern
    draw2 = ImageDraw.Draw(img)
    for i in range(0, width, 20):
        for j in range(0, height, 20):
            alpha = 30 if (i + j) % 40 == 0 else 15
            draw2.rectangle([i, j, i+10, j+10], fill=(255, 255, 255, alpha))
    
    # Save to bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def _generate_cartoon_image(prompt: str, width: int = 512, height: int = 512) -> bytes:
    """Generate a cartoon-style image with bold colors and outlines."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise HTTPException(status_code=500, detail="Pillow library required")
    
    img = Image.new("RGB", (width, height), color="#f0f0f0")
    draw = ImageDraw.Draw(img)
    
    # Draw simple shapes
    rand_x = (width // 4) + (hash(prompt) % (width // 2))
    rand_y = (height // 4) + (hash(prompt + "y") % (height // 2))
    size = min(width, height) // 3
    
    # Main circle
    draw.ellipse([rand_x, rand_y, rand_x + size, rand_y + size], fill="#ff6b6b", outline="#333", width=4)
    
    # Second circle
    draw.ellipse([rand_x + size//2, rand_y + size//2, rand_x + size*1.5, rand_y + size*1.5], 
                 fill="#4ecdc4", outline="#333", width=4)
    
    # Text label (simple)
    try:
        font = ImageFont.load_default()
    except:
        font = None
    draw.text((width//2 - 50, height - 50), prompt[:20], fill="#333", font=font)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def _generate_watercolor_image(prompt: str, width: int = 512, height: int = 512) -> bytes:
    """Generate a watercolor-style image with soft, blended colors."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise HTTPException(status_code=500, detail="Pillow library required")
    
    img = Image.new("RGBA", (width, height), color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Random soft circles
    import random
    random.seed(hash(prompt))
    
    colors = [
        (255, 200, 200, 80),  # soft red
        (200, 255, 200, 80),  # soft green
        (200, 200, 255, 80),  # soft blue
        (255, 255, 200, 80),  # soft yellow
        (255, 200, 255, 80),  # soft pink
    ]
    
    for _ in range(5):
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(50, 200)
        color = random.choice(colors)
        draw.ellipse([x, y, x + size, y + size], fill=color)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def _generate_sketch_image(prompt: str, width: int = 512, height: int = 512) -> bytes:
    """Generate a sketch-style image with hand-drawn appearance."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise HTTPException(status_code=500, detail="Pillow library required")
    
    img = Image.new("RGB", (width, height), color="#fdfbf7")
    draw = ImageDraw.Draw(img)
    
    # Draw irregular lines
    import random
    random.seed(hash(prompt + "sketch"))
    
    color = (50, 50, 50)
    
    # Crosshatching pattern
    for i in range(0, width, 10):
        y_start = random.randint(0, height // 4)
        draw.line([i, y_start, i, y_start + random.randint(50, 200)], fill=color, width=random.randint(1, 2))
    
    for j in range(0, height, 10):
        x_start = random.randint(0, width // 4)
        draw.line([x_start, j, x_start + random.randint(50, 200), j], fill=color, width=random.randint(1, 2))
    
    # Add a simple shape in center
    cx, cy = width // 2, height // 2
    draw.arc([cx - 80, cy - 80, cx + 80, cy + 80], 0, 270, fill=color, width=3)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def _generate_fantasy_image(prompt: str, width: int = 512, height: int = 512) -> bytes:
    """Generate a fantasy-style image with magical colors and glow."""
    try:
        from PIL import Image, ImageDraw, ImageFilter
    except ImportError:
        raise HTTPException(status_code=500, detail="Pillow library required")
    
    img = Image.new("RGB", (width, height), color="#1a1a2e")
    draw = ImageDraw.Draw(img)
    
    # Starfield
    import random
    random.seed(hash(prompt + "fantasy"))
    
    for _ in range(50):
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(1, 3)
        draw.ellipse([x, y, x + size, y + size], fill="#fffff0")
    
    # Add magical circles
    colors = ["#ff00ff", "#00ffff", "#ff00ff", "#ffff00"]
    for i, color in enumerate(colors):
        cx = width // 2 + random.randint(-100, 100)
        cy = height // 2 + random.randint(-100, 100)
        size = 100 - i * 15
        alpha = 60 - i * 10
        # Draw with pseudo-alpha
        draw.ellipse([cx - size, cy - size, cx + size, cy + size], 
                     fill=(*hex_to_rgb(color), alpha)) if False else None
        # Actually just draw without alpha (PIL RGB mode)
        draw.ellipse([cx - size, cy - size, cx + size, cy + size], 
                     fill=hex_to_rgb(color))
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


# ── Generation Functions ───────────────────────────────────────────────────

def _generate_image(prompt: str, style: str) -> bytes:
    """Generate an image based on prompt and style using procedural methods."""
    generators = {
        "realistic": _generate_gradient_image,
        "cartoon": _generate_cartoon_image,
        "watercolor": _generate_watercolor_image,
        "sketch": _generate_sketch_image,
        "fantasy": _generate_fantasy_image,
    }
    
    generator = generators.get(style, _generate_gradient_image)
    return generator(prompt)


def _save_image(image_bytes: bytes, style: str) -> str:
    """Save image to gallery and return relative path."""
    gallery_dir = Path(__file__).resolve().parents[4] / "data" / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    
    import uuid
    filename = f"generated_{uuid.uuid4().hex[:8]}.png"
    filepath = gallery_dir / filename
    
    filepath.write_bytes(image_bytes)
    
    return f"/data/gallery/{filename}"


# ── Endpoint ──────────────────────────────────────────────────────────────

@router.post("/generate", response_model=GenerateResponse)
async def generate_image(
    request: GenerateRequest,
    background_tasks: BackgroundTasks = None,
):
    """Generate an image from text description.
    
    Creates an AI-generated image using procedural generation (no external APIs required).
    Returns a base64-encoded PNG image.
    
    Args:
        request: GenerationRequest with prompt and style
        
    Returns:
        GenerateResponse with image data, style, prompt, and ID
    """
    try:
        # Generate image
        image_bytes = _generate_image(request.prompt, request.style)
        
        # Save to gallery
        image_path = _save_image(image_bytes, request.style)
        
        # Convert to base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{base64_image}"
        
        return GenerateResponse(
            image=data_url,
            style=request.style,
            prompt=request.prompt,
            id=image_path.split("/")[-1].replace(".png", ""),
        )
        
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {e}")


@router.get("/gallery")
async def list_gallery():
    """List all generated images in the gallery."""
    gallery_dir = Path(__file__).resolve().parents[4] / "data" / "gallery"
    
    if not gallery_dir.exists():
        return {"images": []}
    
    images = []
    for filepath in sorted(gallery_dir.glob("generated_*.png"), key=lambda x: x.stat().st_mtime, reverse=True):
        images.append({
            "id": filepath.stem,
            "path": f"/data/gallery/{filepath.name}",
            "created": int(filepath.stat().st_mtime),
        })
    
    return {"images": images[:50]}  # Last 50 images


@router.get("/styles")
async def list_styles():
    """List available image generation styles."""
    return {"styles": list(STYLES.items())}
