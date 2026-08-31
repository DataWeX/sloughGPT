---
id: 20260803_083150_wave-r-cpu-topology-to-100
title: Wave R: cpu_topology to 100%
status: done
tags: infra,coverage,cpu_topology
created: 2026-08-03T08:31:50.722869+00:00
---

Wave R: cpu_topology to 100%

cpu_topology.py 230 stmts 0 miss (100%). Added 33 tests to tests/test_cpu_topology.py: dataclass threads_per_core zero-physical branch; direct helper coverage for _parse_sysctl/_parse_sysctl_string (success+error), _sysctl_cache_kb (known/unknown/None), _linux_cpuinfo (missing-file, full parse, bad-value excepts, no-core-ids phys*cores, cores-only n_physical recheck), _linux_lscpu (ok/error), _detect_numa (lscpu/sysfs/unavailable), _detect_freq_macos (value/brand/error/no-match), _detect_freq_linux (cpuinfo/sysfs/unavailable), _has_cgroup_cpuset (v1/v2/false/read-error); detect_topology under mocked platform (Linux lscpu cache fallback + physical estimate + HT, macOS full sysctl path, macOS fallback model re-read). Autouse fixture clears lru_cache between tests. Infra sweep: 481 passed, 1 pre-existing failure (test_lifecycle_endpoint, fastapi missing). pycache cleared, py_compile OK. Board was 138 cards.