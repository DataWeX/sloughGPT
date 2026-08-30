"""
VM commands - x86 Virtual Machine console and management.
"""
import sys
import os
import json
import time
from pathlib import Path
from typing import Optional

from domains.logging import get_global

log = get_global()


def cmd_vm(args):
    """Show VM status and information."""
    log.header("x86 Virtual Machine")

    try:
        from domains.shell.vm import X86CPU, X86Assembler
        from domains.shell.vm_engine import VMEngine
        from domains.shell.vm_permissions import X86RBAC, Role

        log.section("Status")
        log.key_value("Engine", "Available")
        log.key_value("ISA", "x86-32 (custom)")
        log.key_value("Memory", "64KB")
        log.key_value("Registers", "16 (EAX-EDI)")
        log.key_value("Syscalls", "INT 0x80")

        log.section("RBAC Roles")
        log.key_value("USER", "Basic I/O, print, scan")
        log.key_value("ADMIN", "Device I/O, training")
        log.key_value("KERNEL", "Unrestricted access")

        log.section("Syscall Table")
        syscalls = [
            ("1", "SYS_EXIT", "Exit program"),
            ("2", "SYS_PRINT", "Print string"),
            ("3", "SYS_SCAN", "Read input"),
            ("4", "SYS_OPEN", "Open file"),
            ("5", "SYS_READ", "Read from file"),
            ("6", "SYS_WRITE", "Write to file"),
            ("7", "SYS_CLOSE", "Close file"),
            ("8", "SYS_DEV_OPEN", "Open device"),
            ("9", "SYS_DEV_CALL", "Call device"),
            ("10", "SYS_DEV_CLOSE", "Close device"),
            ("28", "SYS_TRAIN_START", "Start training"),
            ("29", "SYS_TRAIN_STATUS", "Get training status"),
            ("30", "SYS_TRAIN_GET_RESULT", "Get training result"),
        ]
        for num, name, desc in syscalls:
            log.key_value(f"  {num}", f"{name} - {desc}")

    except ImportError as e:
        log.error(f"VM modules not available: {e}")


def cmd_vm_run(args):
    """Run assembly code in the VM."""
    log.header("VM Execution")

    source = getattr(args, "source", "") or ""
    file_path = getattr(args, "file", None)
    if not source:
        # Try to read from file
        if file_path:
            try:
                source = Path(file_path).read_text()
            except FileNotFoundError:
                log.error(f"File not found: {file_path}")
                return
        else:
            log.info("Usage: slooughgpt vm run <assembly_code>")
            log.info("   or: slooughgpt vm run --file <assembly_file>")
            log.info("   or: slooughgpt vm run <program_name>")
            log.info("")
            log.info("Built-in programs: test_hello, hello, test_syscalls, test_files,")
            log.info("  test_exec, test_memory, test_arith, test_stack, test_privilege,")
            log.info("  test_multiprocess, test_fork, test_pipe, test_mmap, test_signal,")
            log.info("  test_usermode, test_ebx_ecx, test_ergonomics, test_singlestep,")
            log.info("  test_v86_dos, empty, hello_linux")
            return

    # Check if source is a built-in program name
    if not file_path and not source.strip().startswith(('[', 'mov', 'push', 'pop', 'jmp', 'call', 'ret', 'int', 'nop')):
        try:
            from domains.shell.vm_programs import PROGRAMS
            if source.strip().lower() in PROGRAMS:
                source = PROGRAMS[source.strip().lower()]
                log.info(f"Running built-in program: {source.strip().lower()}")
        except ImportError:
            pass

    try:
        from domains.shell.vm import X86CPU, X86Assembler
        from domains.shell.vm_engine import VMEngine

        engine = VMEngine()
        engine.load_source(source)

        log.section("Executing")
        start_time = time.time()
        result = engine.run()
        elapsed = time.time() - start_time

        log.section("Result")
        log.key_value("Status", "Completed" if result.exit_reason == "halt" else "Failed")
        log.key_value("Cycles", str(result.total_instructions))
        log.key_value("Time", f"{elapsed:.3f}s")

        output = bytes(engine._console.output)
        if output:
            log.section("Output")
            print(output.decode("ascii", errors="replace"))

        log.section("Registers")
        regs = engine.registers()
        for name, value in regs.items():
            log.key_value(f"  {name}", f"0x{value:08X}")

    except Exception as e:
        log.error(f"VM execution failed: {e}")


def cmd_vm_list(args):
    """List available VM programs."""
    log.header("VM Programs")

    try:
        from domains.shell.vm_programs import PROGRAMS

        for name, program in PROGRAMS.items():
            log.key_value(name, program.description)
    except ImportError:
        log.error("VM programs module not available")


def cmd_vm_info(args):
    """Show detailed VM information."""
    log.header("VM Details")

    try:
        from domains.shell.vm import X86CPU, MEM_SIZE, NUM_REGS
        from domains.shell.vm_permissions import X86RBAC, Role

        log.section("Memory")
        log.key_value("Size", f"{MEM_SIZE} bytes")
        log.key_value("Stack Base", "0xFFFF")

        log.section("CPU")
        log.key_value("Registers", str(NUM_REGS))
        log.key_value("Max Instructions", "100,000")
        log.key_value("Max Call Depth", "256")

        log.section("RBAC")
        rbac = X86RBAC()
        for role in Role:
            perms = rbac.get_permissions(role)
            log.key_value(role.name, ", ".join([p.name for p in perms]))

    except ImportError as e:
        log.error(f"VM modules not available: {e}")


def _run_debug_script(debugger, script_path: str, log):
    """Run debug commands from a script file (non-interactive mode)."""
    try:
        script = Path(script_path).read_text()
    except FileNotFoundError:
        log.error(f"Script not found: {script_path}")
        return

    log.section("Script Mode")
    log.key_value("Script", script_path)

    for line_num, line in enumerate(script.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        action = parts[0].lower()

        log.info(f"[{line_num}] {line}")

        try:
            if action in ("quit", "q", "exit"):
                break
            elif action == "bp":
                target = parts[1] if len(parts) > 1 else "eip"
                label = parts[2] if len(parts) > 2 else ""
                debugger.bp_set(target, label)
            elif action == "stepi" or action == "si":
                count = int(parts[1]) if len(parts) > 1 else 1
                if not debugger.stepi(count):
                    log.warning("Fault or halt")
                    break
            elif action == "cont" or action == "c":
                trace = debugger.continue_exec()
                log.key_value("Exit", trace.exit_reason)
            elif action == "regs" or action == "r":
                debugger.dump_regs()
            elif action == "mem" or action == "m":
                addr = parts[1] if len(parts) > 1 else "eip"
                length = int(parts[2]) if len(parts) > 2 else 64
                debugger.dump_memory(addr, length)
            elif action == "stack" or action == "st":
                depth = int(parts[1]) if len(parts) > 1 else 8
                debugger.dump_stack(depth)
            elif action == "assert_eax":
                expected = int(parts[1], 0)
                actual = debugger.engine.get_reg("eax")
                if actual != expected:
                    log.error(f"ASSERT FAIL: eax=0x{actual:08x}, expected 0x{expected:08x}")
                    return 1
                log.info(f"PASS: eax=0x{actual:08x}")
            elif action == "assert_reg":
                reg = parts[1].lower()
                expected = int(parts[2], 0)
                actual = debugger.engine.get_reg(reg)
                if actual != expected:
                    log.error(f"ASSERT FAIL: {reg}=0x{actual:08x}, expected 0x{expected:08x}")
                    return 1
                log.info(f"PASS: {reg}=0x{actual:08x}")
            elif action == "assert_exit":
                expected = parts[1]
                # Run to completion and check exit reason
                trace = debugger.continue_exec()
                if trace.exit_reason != expected:
                    log.error(f"ASSERT FAIL: exit={trace.exit_reason}, expected {expected}")
                    return 1
                log.info(f"PASS: exit={trace.exit_reason}")
            else:
                log.warning(f"Unknown command: {action}")

        except Exception as e:
            log.error(f"Error at line {line_num}: {e}")
            return 1

    log.section("Script Complete")
    return 0


def cmd_vm_debug(args):
    """Debug assembly code interactively or from a script file."""
    log.header("VM Debugger")

    try:
        from domains.shell.vm_debugger import Debugger

        debugger = Debugger()

        # Set up output callback
        debugger.set_output(lambda text: print(text))

        log.section("Debugger Ready")
        log.info("Commands: bp <addr>, stepi, step_over, step_out, continue")
        log.info("          regs, flags, mem <addr> [len], stack, symbols, quit")

        source = getattr(args, "source", "") or ""
        if not source:
            file_path = getattr(args, "file", None)
            if file_path:
                try:
                    source = Path(file_path).read_text()
                except FileNotFoundError:
                    log.error(f"File not found: {file_path}")
                    return
            else:
                log.info("Provide assembly code or --file option")
                return

        # Load and parse
        engine = debugger.engine
        engine.load_source(source)
        debugger.load_symbols(source)

        log.section("Loaded")
        log.key_value("Entry", f"0x{engine.get_reg('eip'):08x}")

        # Check for non-interactive script mode
        script_path = getattr(args, "script", None)
        if script_path:
            _run_debug_script(debugger, script_path, log)
            return

        # Interactive loop
        while True:
            try:
                cmd = input(f"dbg> ").strip()
                if not cmd:
                    continue

                parts = cmd.split()
                action = parts[0].lower()

                if action in ("quit", "q", "exit"):
                    break
                elif action == "bp":
                    target = parts[1] if len(parts) > 1 else "eip"
                    label = parts[2] if len(parts) > 2 else ""
                    debugger.bp_set(target, label)
                elif action == "bpl":
                    bps = debugger.bp_list()
                    for bp in bps:
                        sym = bp.get("symbol", "")
                        addr = bp["address"]
                        name = f" {sym}" if sym else ""
                        log.key_value(f"  #{bp['id']}", f"0x{addr:08x}{name}")
                elif action == "stepi" or action == "si":
                    count = int(parts[1]) if len(parts) > 1 else 1
                    if not debugger.stepi(count):
                        log.warning("Fault or halt")
                elif action == "step" or action == "s":
                    if not debugger.step_over():
                        log.warning("Step over failed")
                elif action == "finish" or action == "fin":
                    if not debugger.step_out():
                        log.warning("Step out failed")
                elif action == "cont" or action == "c":
                    trace = debugger.continue_exec()
                    log.key_value("Exit", trace.exit_reason)
                elif action == "regs" or action == "r":
                    debugger.dump_regs()
                elif action == "flags" or action == "f":
                    debugger.dump_flags()
                elif action == "mem" or action == "m":
                    addr = parts[1] if len(parts) > 1 else "eip"
                    length = int(parts[2]) if len(parts) > 2 else 64
                    debugger.dump_memory(addr, length)
                elif action == "stack" or action == "st":
                    depth = int(parts[1]) if len(parts) > 1 else 8
                    debugger.dump_stack(depth)
                elif action == "symbols" or action == "sym":
                    syms = debugger.list_symbols()
                    for s in syms:
                        log.key_value(f"  {s['name']}", f"0x{s['address']:08x}")
                elif action == "help" or action == "h":
                    log.info("Commands:")
                    log.info("  bp <addr> [label]  - Set breakpoint")
                    log.info("  bpl                - List breakpoints")
                    log.info("  stepi [n]          - Step N instructions")
                    log.info("  step               - Step over CALL")
                    log.info("  finish             - Step out of function")
                    log.info("  cont               - Continue execution")
                    log.info("  regs               - Dump registers")
                    log.info("  flags              - Dump flags")
                    log.info("  mem <addr> [len]   - Hex dump memory")
                    log.info("  stack [depth]      - Dump stack")
                    log.info("  symbols            - List symbols")
                    log.info("  quit               - Exit debugger")
                else:
                    log.warning(f"Unknown command: {action}")

            except EOFError:
                break
            except KeyboardInterrupt:
                print()
                continue
            except Exception as e:
                log.error(f"Error: {e}")

    except ImportError as e:
        log.error(f"VM debugger not available: {e}")


# CLI registration
COMMANDS = {
    "vm": {
        "func": cmd_vm,
        "help": "Show VM status and information",
        "subcommands": {
            "run": cmd_vm_run,
            "list": cmd_vm_list,
            "info": cmd_vm_info,
            "debug": cmd_vm_debug,
        }
    }
}
