"""
Images command group — image generation and gallery.
"""

from core.framework import click
from core.helpers import ns as _ns, api_get, api_post, output_json


def register(cli):
    """Register images commands with the CLI group."""

    @cli.group(help="Image generation and gallery")
    def images():
        pass

    @images.command("generate", help="Generate an image from text")
    @click.argument("prompt")
    @click.option("--style", type=click.Choice(["realistic", "cartoon", "watercolor", "sketch", "fantasy"]),
                  default="realistic", help="Image style")
    @click.option("--output", "-o", help="Save to file path")
    @click.pass_context
    def images_generate(ctx, prompt, style, output):
        r = api_post(ctx, "/images/generate",
                     json={"prompt": prompt, "style": style})
        if r.status_code != 200:
            log.error(f"Generate failed: {r.text}")
            return
        data = r.json()
        if output_json(ctx, data):
            return
        img_data = data.get("data", data)
        img_id = img_data.get("id", "?")
        log.success(f"Generated image: {img_id} (style={style})")
        if output:
            import base64
            b64 = img_data.get("image", "")
            if b64 and "," in b64:
                b64 = b64.split(",", 1)[1]
            with open(output, "wb") as f:
                f.write(base64.b64decode(b64))
            log.info(f"Saved to: {output}")

    @images.command("gallery", help="List generated images")
    @click.option("--limit", "-n", default=10, type=int)
    @click.pass_context
    def images_gallery(ctx, limit):
        r = api_get(ctx, f"/images/gallery?limit={limit}")
        if r.status_code != 200:
            log.error(f"Gallery failed: {r.text}")
            return
        data = r.json()
        images_list = data.get("data", data).get("images", [])
        if output_json(ctx, {"images": images_list}):
            return
        log.header("Image Gallery")
        for img in images_list:
            log.info(f"  {img.get('id', '?')} — {img.get('prompt', '')[:60]}")

    @images.command("styles", help="List available styles")
    @click.pass_context
    def images_styles(ctx):
        r = api_get(ctx, "/images/styles")
        if r.status_code != 200:
            log.error(f"Styles failed: {r.text}")
            return
        data = r.json()
        styles = data.get("data", data).get("styles", [])
        if output_json(ctx, {"styles": styles}):
            return
        log.header("Available Styles")
        for s in styles:
            log.info(f"  {s}")

    return images
