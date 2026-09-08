"""
World command group — render and simulate the programmable world.
"""

from core.framework import click
from core.helpers import ns as _ns


def register(cli):
    """Register world commands with the CLI group."""

    @cli.group(help="Render and simulate the programmable world")
    def world():
        pass

    @world.command("render", help="Render the current world state")
    @click.option("--width", default=160, type=int, help="Render width")
    @click.option("--height", default=120, type=int, help="Render height")
    @click.option("--samples", default=16, type=int, help="Render samples")
    @click.option("--output", "-o", default=None, help="Output file (PPM)")
    @click.option("--neural", is_flag=True, help="Run neural processing on render")
    def world_render(width, height, samples, output, neural):
        from domains.shell.world_render import RenderBridge, NeuralRenderBridge, RenderConfig
        from domains.shell.simulation import WorldGrid
        import numpy as np

        cfg = RenderConfig(width=width, height=height, samples=samples)

        if neural:
            bridge = NeuralRenderBridge(cfg)
        else:
            bridge = RenderBridge(cfg)

        world_grid = WorldGrid()
        for x in range(10, 54):
            world_grid.material[world_grid.idx(x, 0, 32)] = 1
        for x in range(20, 44):
            world_grid.material[world_grid.idx(x, 0, 32)] = 2
            world_grid.energy[world_grid.idx(x, 0, 32)] = 3.0
        world_grid.material[world_grid.idx(32, 1, 32)] = 4
        world_grid.energy[world_grid.idx(32, 1, 32)] = 5.0

        log.header("Rendering world...")
        bridge.build_scene(world_grid)
        image = bridge.render()
        log.success(f"Rendered {image.shape[1]}x{image.shape[0]} image ({bridge.stats['total_time_ms']:.0f}ms)")

        if output:
            img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            header = f"P6\n{img_uint8.shape[1]} {img_uint8.shape[0]}\n255\n"
            with open(output, "wb") as f:
                f.write(header.encode() + img_uint8.tobytes())
            log.info(f"Saved: {output}")

        if neural:
            result = bridge.process_neural()
            emb = result.get("embedding")
            if emb is not None:
                log.key_value("Embedding dim", str(len(emb)))
                log.key_value("Embedding norm", f"{np.linalg.norm(emb):.4f}")
            desc = bridge.get_descriptor()
            log.key_value("Scene features", str(len(desc.get("tensor_stats", {}))))

    @world.command("tick", help="Run simulation ticks with optional rendering")
    @click.option("--ticks", default=5, type=int, help="Number of ticks")
    @click.option("--babies", default=4, type=int, help="Number of baby agents")
    @click.option("--render", is_flag=True, help="Enable rendering")
    @click.option("--neural", is_flag=True, help="Enable neural processing")
    @click.option("--verbose", is_flag=True, help="Verbose output")
    def world_tick(ticks, babies, render, neural, verbose):
        from domains.shell.simulation import SimScene, Simulation, WorldParams
        from domains.shell.world_render import RenderBridge, NeuralRenderBridge, RenderConfig

        params = WorldParams()
        scene = SimScene(params)

        for _ in range(babies):
            import numpy as np
            from domains.shell.simulation import SimBaby, Entity, EntityType
            baby = SimBaby()
            baby.entity.position[0] = 32 + np.random.randint(-10, 10)
            baby.entity.position[2] = 32 + np.random.randint(-10, 10)
            scene.add_baby(baby)

        render_bridge = None
        if render or neural:
            if neural:
                render_bridge = NeuralRenderBridge()
            else:
                render_bridge = RenderBridge()

        sim = Simulation(scene, max_ticks=ticks, verbose=verbose, render_bridge=render_bridge)
        log.header(f"Running {ticks} ticks with {len(scene.babies)} babies...")
        sim.run()
        summary = sim.summary()

        log.key_value("Ticks", str(summary.get("total_ticks", 0)))
        log.key_value("Babies at end", str(summary.get("alive_at_end", False)))
        log.key_value("Avg energy", f"{summary.get('avg_energy', 0):.1f}")
        log.key_value("Cells written", str(summary.get("total_cells_written", 0)))

        if render_bridge:
            log.key_value("Renders", str(render_bridge.stats.get("renders", 0)))
            log.key_value("Render time", f"{render_bridge.stats.get('total_time_ms', 0):.0f}ms")

    @world.command("analyze", help="Analyze render history over simulation ticks")
    @click.option("--ticks", default=20, type=int, help="Number of ticks to simulate")
    @click.option("--babies", default=4, type=int, help="Number of baby agents")
    @click.option("--threshold", default=0.1, type=float, help="Change detection threshold")
    def world_analyze(ticks, babies, threshold):
        from domains.shell.simulation import SimScene, Simulation, WorldParams
        from domains.shell.world_render import RenderBridge, RenderAnalyzer, RenderConfig

        config = RenderConfig(width=64, height=48, samples=1)
        bridge = RenderBridge(config)

        params = WorldParams()
        scene = SimScene(params)

        for _ in range(babies):
            import numpy as np
            from domains.shell.simulation import SimBaby
            baby = SimBaby()
            baby.entity.position[0] = 32 + np.random.randint(-10, 10)
            baby.entity.position[2] = 32 + np.random.randint(-10, 10)
            scene.add_baby(baby)

        sim = Simulation(scene, max_ticks=ticks, render_bridge=bridge)
        analyzer = RenderAnalyzer(bridge._history if hasattr(bridge, '_history') else None)

        sim.run()

        for i, entry in enumerate(bridge._history._entries if hasattr(bridge, '_history') else []):
            analyzer.history.add(entry["image"], tick=entry["tick"])

        summary = analyzer.summary()
        log.header("Render Analysis")
        log.key_value("Total renders", str(summary.get("count", 0)))
        log.key_value("Significant changes", str(summary.get("significant_changes", 0)))
        if summary.get("mean_range"):
            log.key_value("Mean range", f"{summary['mean_range'][0]:.4f} - {summary['mean_range'][1]:.4f}")
        if summary.get("mean_trend") is not None:
            log.key_value("Mean trend", f"{summary['mean_trend']:+.4f}")

        changes = analyzer.detect_significant_changes(threshold)
        if changes:
            log.header("Significant Changes")
            for c in changes:
                log.info(f"  Tick {c['tick_from']} -> {c['tick_to']}: "
                         f"{c['change_ratio']:.1%} changed, MSE={c['mse']:.6f}")

    @world.command("diff", help="Compare two render images")
    @click.argument("image_a", type=click.Path(exists=True))
    @click.argument("image_b", type=click.Path(exists=True))
    def world_diff(image_a, image_b):
        import numpy as np
        from domains.shell.world_render import RenderDiff
        from PIL import Image as PILImage

        a = np.array(PILImage.open(image_a)).astype(np.float32) / 255.0
        b = np.array(PILImage.open(image_b)).astype(np.float32) / 255.0

        diff = RenderDiff(a, b)
        s = diff.summary()

        log.header("Render Diff")
        log.key_value("MSE", f"{s['mse']:.6f}")
        log.key_value("MAE", f"{s['mae']:.6f}")
        log.key_value("Max diff", f"{s['max_diff']:.4f}")
        log.key_value("Changed pixels", f"{s['changed_pixels']}/{s['total_pixels']} ({s['change_ratio']:.1%})")
        log.key_value("Mean A", f"{s['mean_a']:.4f}")
        log.key_value("Mean B", f"{s['mean_b']:.4f}")

    @world.command("ingest", help="Feed data into the world grid")
    @click.argument("source_type", type=click.Choice(["file", "url", "rss", "records"]))
    @click.argument("source_value")
    @click.option("--radius", default=15, type=int, help="Placement radius around center")
    @click.option("--decay", default=0.95, type=float, help="Energy decay rate per tick")
    @click.option("--verbose", is_flag=True, help="Verbose output")
    def world_ingest(source_type, source_value, radius, decay, verbose):
        import numpy as np
        from domains.collections.perception import WorldPerception, PerceptionConfig
        from domains.collections.sources import FileSource, UrlSource, RssSource, GeneratorSource, Record
        from domains.shell.simulation import WorldGrid

        config = PerceptionConfig(radius=radius, decay_rate=decay)
        perception = WorldPerception(config)
        world_grid = WorldGrid()

        if source_type == "file":
            records = []
            with open(source_value, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(Record(content=line))
            events = perception.ingest_records(records)
        elif source_type == "url":
            from domains.collections.sources import UrlSource
            source = UrlSource(source_value)
            events = perception.ingest_source(source)
        elif source_type == "rss":
            from domains.collections.sources import RssSource
            source = RssSource(source_value)
            events = perception.ingest_source(source)
        else:
            records = [Record(content=source_value)]
            events = perception.ingest_records(records)

        perception.apply_to_grid(world_grid, events)

        log.header("World Ingestion")
        log.key_value("Records ingested", str(len(events)))
        log.key_value("Grid cells filled", str(np.sum(world_grid.material != 0)))
        log.key_value("Avg energy", f"{np.mean(world_grid.energy[world_grid.material != 0]):.2f}" if np.any(world_grid.material != 0) else "0.00")

        if verbose:
            summary = perception.summary()
            for cls, count in summary.get("material_counts", {}).items():
                log.key_value(f"  {cls}", str(count))

    return world
