"""
Generate a minimal VLM test dataset with synthetic images + captions.

Creates datasets/vlm-demo/ with 5 colored-shape PNGs and a corpus.jsonl.
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "vlm-demo"
IMAGES_DIR = OUT_DIR / "images"

SHAPES = [
    ("red_square", "A red square centered on a black background"),
    ("blue_circle", "A blue circle drawn on a dark background"),
    ("green_triangle", "A green triangle pointing upward on a black background"),
    ("yellow_star", "A yellow five-pointed star on a dark background"),
    ("cyan_diamond", "A cyan diamond shape centered on a black background"),
    ("magenta_hexagon", "A magenta hexagon on a dark background"),
    ("white_cross", "A white plus sign centered on a black background"),
    ("orange_ellipse", "An orange ellipse stretched horizontally on a dark background"),
]


def draw_shape(name: str, color: tuple[int, int, int], draw_func):
    """Draw a shape onto a 64x64 RGB image."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (64, 64), (10, 10, 10))
    draw = ImageDraw.Draw(img)
    draw_func(draw, color)
    path = IMAGES_DIR / f"{name}.png"
    img.save(path)
    return path


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    from PIL import Image, ImageDraw

    entries = []

    for stem, caption in SHAPES:
        if stem == "red_square":
            def _draw(d, c):
                d.rectangle([12, 12, 52, 52], fill=c)
            path = draw_shape(stem, (200, 30, 30), _draw)

        elif stem == "blue_circle":
            def _draw(d, c):
                d.ellipse([8, 8, 56, 56], fill=c)
            path = draw_shape(stem, (30, 100, 220), _draw)

        elif stem == "green_triangle":
            def _draw(d, c):
                d.polygon([(32, 4), (4, 56), (60, 56)], fill=c)
            path = draw_shape(stem, (30, 180, 50), _draw)

        elif stem == "yellow_star":
            def _draw(d, c):
                pts = []
                for i in range(10):
                    r = 26 if i % 2 == 0 else 12
                    angle = -90 + i * 36
                    import math
                    x = 32 + r * math.cos(math.radians(angle))
                    y = 32 + r * math.sin(math.radians(angle))
                    pts.append((x, y))
                d.polygon(pts, fill=c)
            path = draw_shape(stem, (220, 200, 20), _draw)

        elif stem == "cyan_diamond":
            def _draw(d, c):
                d.polygon([(32, 4), (60, 32), (32, 60), (4, 32)], fill=c)
            path = draw_shape(stem, (20, 200, 200), _draw)

        elif stem == "magenta_hexagon":
            def _draw(d, c):
                import math
                pts = []
                for i in range(6):
                    angle = -90 + i * 60
                    x = 32 + 24 * math.cos(math.radians(angle))
                    y = 32 + 24 * math.sin(math.radians(angle))
                    pts.append((x, y))
                d.polygon(pts, fill=c)
            path = draw_shape(stem, (200, 30, 200), _draw)

        elif stem == "white_cross":
            def _draw(d, c):
                d.rectangle([24, 4, 40, 60], fill=c)
                d.rectangle([4, 24, 60, 40], fill=c)
            path = draw_shape(stem, (220, 220, 220), _draw)

        elif stem == "orange_ellipse":
            def _draw(d, c):
                d.ellipse([4, 16, 60, 48], fill=c)
            path = draw_shape(stem, (220, 120, 20), _draw)

        rel_path = f"datasets/vlm-demo/images/{path.name}"
        entries.append({
            "image_path": rel_path,
            "caption": caption,
        })

    jsonl_path = OUT_DIR / "corpus.jsonl"
    with open(jsonl_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    print(f"Created {len(entries)} image-text pairs in {OUT_DIR}")
    print(f"  Images: {IMAGES_DIR}")
    print(f"  JSONL:  {jsonl_path}")


if __name__ == "__main__":
    main()
