---
id: 20260813_163523_strui-component-bug-fixes-regression-tests
title: strui component bug fixes + regression tests
status: done
tags: strui,ui,frontend
created: 2026-08-13T16:35:23.076419+00:00
---

strui component bug fixes + regression tests

Fixed 6 component bugs in packages/strui: (1) tooltip asChild leaked to DOM + position not recomputed on defaultOpen, (2) DropdownMenuRadioItem swallowed children + added onSelect to Item, (3) Dialog/AlertDialog trigger asChild + overlay click-to-close, (4) tabs aria-controls/aria-labelledby + arrow-key nav, (5) ModelStatusPill className drop, (6) RangeSlider showValue prop. 14 new regression tests. Strui suite 720/720, tsc exit 0 (strui + web).