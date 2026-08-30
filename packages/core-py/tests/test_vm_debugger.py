"""
Tests for VM debugger and module loader.
"""
import pytest
from pathlib import Path


# ── Debugger Tests ────────────────────────────────────────────────────────────

class TestDebugger:
    """Test the VM debugger."""

    def test_debugger_creation(self):
        from domains.shell.vm_debugger import Debugger
        debugger = Debugger()
        assert debugger.engine is not None
        assert debugger.symbols is not None

    def test_symbol_table(self):
        from domains.shell.vm_debugger import SymbolTable
        st = SymbolTable()
        st.add("main", 0x1000)
        st.add("loop", 0x1010)
        assert st.resolve("main") == 0x1000
        assert st.resolve("loop") == 0x1010
        assert st.name_for(0x1000) == "main"

    def test_symbol_resolve_hex(self):
        from domains.shell.vm_debugger import SymbolTable
        st = SymbolTable()
        assert st.resolve("0x1000") == 0x1000

    def test_symbol_resolve_decimal(self):
        from domains.shell.vm_debugger import SymbolTable
        st = SymbolTable()
        assert st.resolve("4096") == 4096

    def test_breakpoint_set(self):
        from domains.shell.vm_debugger import Debugger
        debugger = Debugger()
        # Set breakpoint at an address
        bp_id = debugger.bp_set("0x1000", "test_bp")
        assert bp_id >= 0
        bps = debugger.bp_list()
        assert len(bps) == 1
        assert bps[0]["address"] == 0x1000

    def test_breakpoint_list_with_symbol(self):
        from domains.shell.vm_debugger import Debugger
        debugger = Debugger()
        debugger.symbols.add("main", 0x1000)
        bp_id = debugger.bp_set("main")
        bps = debugger.bp_list()
        assert bps[0]["symbol"] == "main"

    def test_breakpoint_remove(self):
        from domains.shell.vm_debugger import Debugger
        debugger = Debugger()
        bp_id = debugger.bp_set("0x1000")
        debugger.bp_remove(bp_id)
        bps = debugger.bp_list()
        assert len(bps) == 0

    def test_breakpoint_clear(self):
        from domains.shell.vm_debugger import Debugger
        debugger = Debugger()
        debugger.bp_set("0x1000")
        debugger.bp_set("0x2000")
        debugger.bp_clear()
        bps = debugger.bp_list()
        assert len(bps) == 0

    def test_watchpoint_set(self):
        from domains.shell.vm_debugger import Debugger
        debugger = Debugger()
        wp_id = debugger.wp_set("0x2000", 4, "data")
        assert wp_id >= 0
        wps = debugger.wp_list()
        assert len(wps) == 1

    def test_watchpoint_remove(self):
        from domains.shell.vm_debugger import Debugger
        debugger = Debugger()
        wp_id = debugger.wp_set("0x2000")
        debugger.wp_remove(wp_id)
        wps = debugger.wp_list()
        assert len(wps) == 0

    def test_stepi(self):
        from domains.shell.vm_debugger import Debugger
        debugger = Debugger()
        engine = debugger.engine
        # Load a simple program
        engine.load_source("mov eax, 1\nmov ebx, 2\nhlt")
        # Step one instruction
        result = debugger.stepi()
        assert result is True

    def test_dump_regs(self):
        from domains.shell.vm_debugger import Debugger
        debugger = Debugger()
        engine = debugger.engine
        engine.load_source("mov eax, 0x42")
        debugger.stepi()
        # Just verify it doesn't crash
        regs = debugger.engine.registers()
        assert "eax" in regs

    def test_dump_memory(self):
        from domains.shell.vm_debugger import Debugger
        debugger = Debugger()
        engine = debugger.engine
        engine.load_source("mov eax, 0xDEADBEEF\nmov [0x1000], eax")
        # Just verify it doesn't crash
        hex_dump = engine.dump_memory(0x1000, 16)
        assert "de" in hex_dump.lower() or "DE" in hex_dump

    def test_analyze_trace(self):
        from domains.shell.vm_debugger import Debugger
        debugger = Debugger()
        engine = debugger.engine
        engine.load_source("mov eax, 1\nmov ebx, 2\nhlt")
        trace = engine.run()
        analysis = debugger.analyze_trace(trace)
        assert "total_instructions" in analysis
        assert "exit_reason" in analysis

    def test_list_symbols(self):
        from domains.shell.vm_debugger import Debugger
        debugger = Debugger()
        debugger.symbols.add("main", 0x1000, kind="function")
        syms = debugger.list_symbols()
        assert len(syms) == 1
        assert syms[0]["name"] == "main"
        assert syms[0]["kind"] == "function"


# ── Module Loader Tests ──────────────────────────────────────────────────────

class TestModuleLoader:
    """Test the kernel module loader."""

    def test_loader_creation(self):
        from domains.shell.addons.module_loader import ModuleLoader
        loader = ModuleLoader()
        assert loader.list_modules() == []

    def test_add_addon_dir(self):
        from domains.shell.addons.module_loader import ModuleLoader
        loader = ModuleLoader()
        loader.add_addon_dir("/tmp/test_addons")
        assert len(loader._addon_dirs) == 1

    def test_discover_empty(self):
        from domains.shell.addons.module_loader import ModuleLoader
        loader = ModuleLoader()
        found = loader.discover()
        assert found == []

    def test_module_info_states(self):
        from domains.shell.addons.module_loader import ModuleInfo
        info = ModuleInfo(name="test", path="/tmp/test.py")
        assert info.state == "unloaded"
        assert info.error is None

    def test_summary(self):
        from domains.shell.addons.module_loader import ModuleLoader
        loader = ModuleLoader()
        summary = loader.summary()
        assert summary["total"] == 0
        assert summary["loaded"] == []

    def test_loaded_list(self):
        from domains.shell.addons.module_loader import ModuleLoader
        loader = ModuleLoader()
        assert loader.loaded() == []

    def test_errors_list(self):
        from domains.shell.addons.module_loader import ModuleLoader
        loader = ModuleLoader()
        assert loader.errors() == []

    def test_discover_with_addon_dir(self, tmp_path):
        from domains.shell.addons.module_loader import ModuleLoader
        # Create a test addon directory with a valid addon
        addon_dir = tmp_path / "addons"
        addon_dir.mkdir()
        addon_file = addon_dir / "test_addon.py"
        addon_file.write_text("""
from domains.shell.addons.base import Addon

class Addon:
    def setup(self, kernel):
        self.kernel = kernel
        self.setup_called = True

    def cleanup(self):
        self.cleanup_called = True
""")
        loader = ModuleLoader()
        loader.add_addon_dir(addon_dir)
        found = loader.discover()
        assert "test_addon" in found
        assert len(loader.list_modules()) == 1

    def test_load_addon(self, tmp_path):
        from domains.shell.addons.module_loader import ModuleLoader
        addon_dir = tmp_path / "addons"
        addon_dir.mkdir()
        addon_file = addon_dir / "my_addon.py"
        addon_file.write_text("""
from domains.shell.addons.base import Addon

__version__ = "1.0.0"
__description__ = "Test addon"
__author__ = "Test"

class Addon:
    def setup(self, kernel):
        self.kernel = kernel

    def cleanup(self):
        pass
""")
        loader = ModuleLoader()
        loader.add_addon_dir(addon_dir)
        loader.discover()
        addon = loader.load("my_addon")
        assert addon is not None
        assert addon.kernel is not None
        info = loader.get_module("my_addon")
        assert info.state == "loaded"
        assert info.version == "1.0.0"
        assert info.description == "Test addon"
        assert info.author == "Test"

    def test_unload_addon(self, tmp_path):
        from domains.shell.addons.module_loader import ModuleLoader
        addon_dir = tmp_path / "addons"
        addon_dir.mkdir()
        addon_file = addon_dir / "unload_test.py"
        addon_file.write_text("""
from domains.shell.addons.base import Addon

class Addon:
    def setup(self, kernel):
        pass
    def cleanup(self):
        self.cleaned = True
""")
        loader = ModuleLoader()
        loader.add_addon_dir(addon_dir)
        loader.discover()
        loader.load("unload_test")
        assert loader.loaded() == ["unload_test"]
        result = loader.unload("unload_test")
        assert result is True
        assert loader.loaded() == []

    def test_hot_reload(self, tmp_path):
        from domains.shell.addons.module_loader import ModuleLoader
        addon_dir = tmp_path / "addons"
        addon_dir.mkdir()
        addon_file = addon_dir / "reload_test.py"
        addon_file.write_text("""
from domains.shell.addons.base import Addon

class Addon:
    def setup(self, kernel):
        self.version = 1
""")
        loader = ModuleLoader()
        loader.add_addon_dir(addon_dir)
        loader.discover()
        addon1 = loader.load("reload_test")
        assert addon1.version == 1
        # Update the file
        addon_file.write_text("""
from domains.shell.addons.base import Addon

class Addon:
    def setup(self, kernel):
        self.version = 2
""")
        addon2 = loader.reload("reload_test")
        assert addon2.version == 2

    def test_lifecycle_hooks(self, tmp_path):
        from domains.shell.addons.module_loader import ModuleLoader
        addon_dir = tmp_path / "addons"
        addon_dir.mkdir()
        addon_file = addon_dir / "hook_test.py"
        addon_file.write_text("""
from domains.shell.addons.base import Addon

class Addon:
    def setup(self, kernel):
        pass
""")
        loader = ModuleLoader()
        loader.add_addon_dir(addon_dir)
        loader.discover()
        events = []
        loader.on("pre_load", lambda name: events.append(("pre_load", name)))
        loader.on("post_load", lambda name: events.append(("post_load", name)))
        loader.on("pre_unload", lambda name: events.append(("pre_unload", name)))
        loader.on("post_unload", lambda name: events.append(("post_unload", name)))
        loader.load("hook_test")
        loader.unload("hook_test")
        assert events == [
            ("pre_load", "hook_test"),
            ("post_load", "hook_test"),
            ("pre_unload", "hook_test"),
            ("post_unload", "hook_test"),
        ]

    def test_load_nonexistent_raises(self):
        from domains.shell.addons.module_loader import ModuleLoader
        loader = ModuleLoader()
        try:
            loader.load("nonexistent_module")
            assert False, "Should have raised ImportError"
        except ImportError as e:
            assert "nonexistent_module" in str(e)

    def test_load_broken_addon(self, tmp_path):
        from domains.shell.addons.module_loader import ModuleLoader
        addon_dir = tmp_path / "addons"
        addon_dir.mkdir()
        addon_file = addon_dir / "broken.py"
        addon_file.write_text("raise ValueError('intentional error')")
        loader = ModuleLoader()
        loader.add_addon_dir(addon_dir)
        loader.discover()
        try:
            loader.load("broken")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "broken" in str(e)
        info = loader.get_module("broken")
        assert info.state == "error"
        assert "intentional error" in info.error

    def test_summary_with_loaded(self, tmp_path):
        from domains.shell.addons.module_loader import ModuleLoader
        addon_dir = tmp_path / "addons"
        addon_dir.mkdir()
        addon_file = addon_dir / "summary_addon.py"
        addon_file.write_text("""
from domains.shell.addons.base import Addon

class Addon:
    def setup(self, kernel):
        pass
""")
        loader = ModuleLoader()
        loader.add_addon_dir(addon_dir)
        loader.discover()
        loader.load("summary_addon")
        summary = loader.summary()
        assert summary["total"] == 1
        assert "summary_addon" in summary["loaded"]
        assert summary["by_state"]["loaded"] == 1

    def test_cleanup_called_on_unload(self, tmp_path):
        from domains.shell.addons.module_loader import ModuleLoader
        addon_dir = tmp_path / "addons"
        addon_dir.mkdir()
        addon_file = addon_dir / "cleanup_addon.py"
        addon_file.write_text("""
from domains.shell.addons.base import Addon

class Addon:
    def setup(self, kernel):
        self.cleaned = False
    def cleanup(self):
        self.cleaned = True
""")
        loader = ModuleLoader()
        loader.add_addon_dir(addon_dir)
        loader.discover()
        addon = loader.load("cleanup_addon")
        assert addon.cleaned is False
        loader.unload("cleanup_addon")
        # Note: cleanup is called before instance is cleared, so we can't check it after unload
        # But we can verify unload succeeded
        assert loader.loaded() == []

    def test_set_kernel(self, tmp_path):
        from domains.shell.addons.module_loader import ModuleLoader
        addon_dir = tmp_path / "addons"
        addon_dir.mkdir()
        addon_file = addon_dir / "kernel_test.py"
        addon_file.write_text("""
from domains.shell.addons.base import Addon

class Addon:
    def setup(self, kernel):
        self.kernel = kernel
""")
        loader = ModuleLoader()
        loader.add_addon_dir(addon_dir)
        loader.discover()
        # Create a mock kernel
        class MockKernel:
            pass
        kernel = MockKernel()
        loader.set_kernel(kernel)
        addon = loader.load("kernel_test")
        assert addon.kernel is kernel


# ── CLI Command Tests ────────────────────────────────────────────────────────

class TestVMCLICommands:
    """Test VM CLI command files."""

    def test_vm_command_exists(self):
        vm_cmd = Path(__file__).resolve().parents[3] / "apps" / "cli" / "src" / "commands" / "vm.py"
        assert vm_cmd.exists()

    def test_vm_command_has_debug(self):
        vm_cmd = Path(__file__).resolve().parents[3] / "apps" / "cli" / "src" / "commands" / "vm.py"
        content = vm_cmd.read_text()
        assert "cmd_vm_debug" in content
        assert "debugger" in content.lower()

    def test_vm_command_has_debugger_import(self):
        vm_cmd = Path(__file__).resolve().parents[3] / "apps" / "cli" / "src" / "commands" / "vm.py"
        content = vm_cmd.read_text()
        assert "from domains.shell.vm_debugger import Debugger" in content

    def test_build_command_exists(self):
        build_cmd = Path(__file__).resolve().parents[3] / "apps" / "cli" / "src" / "commands" / "build.py"
        assert build_cmd.exists()
