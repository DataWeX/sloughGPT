# v86 BIOS binaries

Vendored from the copy.sh v86 demo (https://copy.sh/v86/bios/seabios.bin,
https://copy.sh/v86/bios/vgabios.bin) at commit time. The legacy copy.sh
server sends no `Access-Control-Allow-Origin` header, so the browser cannot
fetch these at runtime — they are committed here and served same-origin.

- `seabios.bin` — SeaBIOS (GPL-2.0, seabios.org)
- `vgabios.bin` — Bochs/VGA BIOS (GPL-2.0, bochs.sourceforge.net)

`hooks/useV86.ts` resolves the BIOS pair in order: local (this directory)
first, then the copy.sh-hosted pair as a fallback probe.
