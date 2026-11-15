---
id: 20261031_165214_font-verdict-doc-table-magazine-specimen-truth-up
status: todo
board: 20261031_165214_font-verdict-doc-table-magazine-specimen-truth-up
title: "Font verdict: doc table + magazine specimen truth-up"
tags: ["area:docs"]
sprint: ""
gh: null
---

# Font verdict: doc table + magazine specimen truth-up

Verdict (user, 2026-10-31, per-glyph split, moodboard tab D): letters & words = Outfit (body, what the app ships under the --font-rubik misnomer at apps/web/index.html:72); digits = Rubik (numeric role, replaces the never-loaded ghost Lato — --font-lato is defined nowhere, its only ref is the magazine specimen which silently falls back to system-ui); code = JetBrains Mono. Even '5m ago': 5 = Rubik, 'm ago' = Outfit. SCOPE (doc+specimen only — user chose card-only, do not build in the noir session): (1) docs/design/DESIGN_SYSTEM.md:113-117 font table still says Body=Rubik/Extra=Lato — rewrite to Body=Outfit/Numeric=Rubik/Code=JetBrains Mono; Log Entry spec 243-270 stays mono (spec unchanged, only the table lies). (2) apps/web/app/(app)/magazine/page.tsx Font Families specimen (~388-430): row labeled 'Rubik' renders var(--font-rubik) which IS Outfit (lying label); Lato row is dead (var(--font-lato) undefined) — relabel to truth: Outfit/body, Rubik/numeric, JetBrains Mono/code; drop or fix Lato row. (3) apps/web/app/(app)/magazine/page.test.tsx:180 pins 'Lato' — update to match. NOT in scope: loading a real Rubik font-file app-wide (font-numeric token switch) — separate decision, needs index.html + tailwind config + font file. Evidence: /tmp/opencode/moodboard-modern-noir.html tab D. Gates when built: tsc, eslint, vitest magazine closure.
