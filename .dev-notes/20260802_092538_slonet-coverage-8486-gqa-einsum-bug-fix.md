---
id: 20260802_092538_slonet-coverage-8486-gqa-einsum-bug-fix
title: SloNet coverage 84%→86%, GQA einsum bug fix
status: done
tags: core,tests,slonet
created: 2026-08-02T09:25:38.737886+00:00
---

SloNet coverage 84%→86%, GQA einsum bug fix

Covered export_to_sou failure branch, int4 lazy unpack, .points.json import fallback. Fixed GQA B>1 einsum bug (slonet.py:2154). pragma:no cover on numba-only bodies. test_slonet_legacy.py: 75 tests. slonet.py coverage 70%→86%.