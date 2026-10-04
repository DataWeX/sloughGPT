"""Marker guard: proves this file is never collected (module name does not start with test_)
and therefore cannot affect the honest 35-test committed gate."""

# Deliberately not `assert False`: python -O strips asserts, which would let this
# guard pass silently — exactly the failure it exists to catch. raise is
# unconditional, and satisfies ruff B011.
raise AssertionError("intentional collection guard — remove file; you should never run this")
