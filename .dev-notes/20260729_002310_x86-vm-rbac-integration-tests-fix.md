---
id: 20260729_002310_x86-vm-rbac-integration-tests-fix
title: x86 VM RBAC integration tests fix
status: done
tags: shell,rbac,virtual-machine
created: 2026-07-29T00:23:10.469626+00:00
---

x86 VM RBAC integration tests fix


Completed x86 VM RBAC system: all 26 syscalls mapped with permissions, fork inherits role, SYS_GETROLE (27) allows querying current role, diagnostic logging on denial. Fixed spawn() org parameter propagation to assembler so data labels resolve correctly. 173 VM tests pass (16 RBAC + 157 OS layer).