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
