---
id: 20260830_060916_component-naming-cleanup
title: Component naming cleanup
status: done
tags: frontend,refactoring
created: 2026-08-30T06:09:16.884260+00:00
---

Component naming cleanup

Completed component naming cleanup: split specialized.tsx and display.tsx into individual files, moved hooks to hooks/ and utils to lib/, renamed cryptic components (DPOCard→PreferenceOptimizationCard, KvCacheCard→KVCacheCard, TurboCard→QuickTrainCard, ShellPanel→TerminalPanel), standardized Modal→Dialog suffix (DatasetImportModal→DatasetImportDialog, DatasetInlineImportModal→QuickImportDialog, KeyboardShortcutsModal→KeyboardShortcutsDialog), and renamed StatusCard→SystemStatusCard. All changes pass typecheck and lint.