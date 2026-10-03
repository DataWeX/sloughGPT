"""Marker guard: proves this file is never collected (module name does not start with test_)
and therefore cannot affect the honest 35-test committed gate."""
assert False, "intentional collection guard — remove file; you should never run this"
