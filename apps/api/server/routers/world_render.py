"""
World Render Router — rendering endpoints for the programmable world.
"""
import logging
import time as _time
from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel, Field

from schemas.common import success_response, raise_error, classify_and_raise, safe_audit_log

logger = logging.getLogger("slo.routers.world_render")


class RenderConfigRequest(BaseModel):
    """Configuration for rendering."""
    width: int = Field(default=160, ge=32, le=1024)
    height: int = Field(default=120, ge=32, le=1024)
    samples: int = Field(default=16, ge=1, le=256)
    camera_height: float = Field(default=40.0, ge=1.0)
    camera_distance: float = Field(default=30.0, ge=1.0)


class SimTickRequest(BaseModel):
    """Configuration for a simulation tick with rendering."""
    max_ticks: int = Field(default=1, ge=1, le=100)
    render: bool = Field(default=True)
    neural: bool = Field(default=False)


class WorldRenderRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/world", tags=["world"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/render", self.render_world, methods=["POST"])
        self.router.add_api_route("/render/image", self.render_world_image, methods=["POST"])
        self.router.add_api_route("/neural", self.neural_process, methods=["POST"])
        self.router.add_api_route("/tick", self.run_tick, methods=["POST"])
        self.router.add_api_route("/stats", self.get_stats, methods=["GET"])

    async def render_world(self, config: RenderConfigRequest | None = None) -> dict:
        """Render the current world state and return state tensors."""
        _t0 = _time.monotonic()
        try:
            from domains.shell.world_render import RenderBridge, RenderConfig
            from domains.shell.simulation import WorldGrid

            cfg = RenderConfig(
                width=config.width if config else 160,
                height=config.height if config else 120,
                samples=config.samples if config else 16,
                camera_height=config.camera_height if config else 40.0,
                camera_distance=config.camera_distance if config else 30.0,
            )
            bridge = RenderBridge(cfg)

            world = WorldGrid()
            world.material[world.idx(32, 0, 32)] = 1
            world.material[world.idx(32, 1, 32)] = 1
            world.material[world.idx(33, 0, 32)] = 2
            world.energy[world.idx(33, 0, 32)] = 5.0

            bridge.build_scene(world)
            tensors = bridge.render_state_tensors()
            _elapsed_ms = (_time.monotonic() - _t0) * 1000

            shapes = {k: list(v.shape) for k, v in tensors.items()}
            logger.info("World render in %.1fms (shapes=%s)", _elapsed_ms, list(shapes.keys()))
            safe_audit_log("world.render", resource="state_tensors", detail=f"elapsed={_elapsed_ms:.0f}ms shapes={list(shapes.keys())}")

            return success_response(data={
                "shapes": shapes,
                "stats": bridge.stats,
                "tensor_keys": list(tensors.keys()),
            })
        except Exception as e:
            classify_and_raise(e, source="render_world")

    async def render_world_image(self, config: RenderConfigRequest | None = None) -> Response:
        """Render the world and return a PNG image."""
        _t0 = _time.monotonic()
        try:
            from domains.shell.world_render import RenderBridge, RenderConfig
            from domains.shell.simulation import WorldGrid
            import numpy as np

            cfg = RenderConfig(
                width=config.width if config else 160,
                height=config.height if config else 120,
                samples=config.samples if config else 16,
            )
            bridge = RenderBridge(cfg)

            world = WorldGrid()
            world.material[world.idx(32, 0, 32)] = 1
            world.material[world.idx(32, 1, 32)] = 1
            world.material[world.idx(33, 0, 32)] = 2
            world.energy[world.idx(33, 0, 32)] = 5.0

            bridge.build_scene(world)
            image = bridge.render()

            img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            header = f"P6\n{img_uint8.shape[1]} {img_uint8.shape[0]}\n255\n"
            ppm_data = header.encode() + img_uint8.tobytes()

            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            safe_audit_log("world.render_image", resource="ppm", detail=f"elapsed={_elapsed_ms:.0f}ms size={img_uint8.shape}")

            return Response(content=ppm_data, media_type="image/x-portable-pixmap")
        except Exception as e:
            classify_and_raise(e, source="render_world_image")

    async def neural_process(self, config: RenderConfigRequest | None = None) -> dict:
        """Render the world and process through the neural pipeline."""
        _t0 = _time.monotonic()
        try:
            from domains.shell.world_render import NeuralRenderBridge, RenderConfig
            from domains.shell.simulation import WorldGrid

            cfg = RenderConfig(
                width=config.width if config else 160,
                height=config.height if config else 120,
                samples=config.samples if config else 16,
            )
            bridge = NeuralRenderBridge(cfg)

            world = WorldGrid()
            world.material[world.idx(32, 0, 32)] = 1
            world.material[world.idx(32, 1, 32)] = 1

            bridge.render_tick(world)
            neural_result = bridge.process_neural()

            embedding = neural_result.get("embedding")
            descriptor = bridge.get_descriptor()

            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            safe_audit_log("world.neural_process", resource="neural", detail=f"elapsed={_elapsed_ms:.0f}ms embedding_shape={list(embedding.shape) if embedding is not None else None}")

            return success_response(data={
                "embedding_shape": list(embedding.shape) if embedding is not None else None,
                "descriptor": descriptor,
                "stats": bridge.stats,
            })
        except Exception as e:
            classify_and_raise(e, source="neural_process")

    async def run_tick(self, config: SimTickRequest | None = None) -> dict:
        """Run a simulation tick with optional rendering."""
        _t0 = _time.monotonic()
        try:
            from domains.shell.simulation import SimScene, Simulation, WorldParams

            params = WorldParams()
            scene = SimScene(params)

            render_bridge = None
            if config and config.render:
                from domains.shell.world_render import RenderBridge
                render_bridge = RenderBridge()

            sim = Simulation(scene, max_ticks=config.max_ticks if config else 1,
                           render_bridge=render_bridge)
            results = sim.step()

            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            safe_audit_log("world.run_tick", resource="simulation", detail=f"elapsed={_elapsed_ms:.0f}ms tick={scene.tick} babies={len(results)}")

            return success_response(data={
                "tick": scene.tick,
                "babies": len(results),
                "render_stats": render_bridge.stats if render_bridge else None,
            })
        except Exception as e:
            classify_and_raise(e, source="run_tick")

    async def get_stats(self) -> dict:
        """Get world rendering statistics."""
        try:
            return success_response(data={
                "status": "available",
                "components": ["RenderBridge", "NeuralRenderBridge", "WorldToSceneMapper"],
                "materials": {
                    "air": 0, "ground": 1, "food": 2, "toxic": 3,
                    "signal": 4, "nest": 5, "water": 6,
                },
            })
        except Exception as e:
            classify_and_raise(e, source="world.stats")


router = WorldRenderRouter().router
