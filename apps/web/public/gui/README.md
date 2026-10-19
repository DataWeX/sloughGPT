# GUI desktop saved-states (fetched — not committed)

v86 saved-state snapshots of GUI desktop OSes from the copy.sh demo,
fetched by `scripts/fetch_gui_images.sh` (`*.zst` is gitignored). The
browser cannot fetch these directly: `c.copy.sh` answers cross-origin
requests with 403 (hotlink protection), so they are vendored same-origin.

A state is a full machine snapshot (memory + devices + sparse disk) — it
restores **without** the original disk image, which is how these desktops
stay bootable even though their raw images 404 on both copy.sh hosts
(verified 2026-10-19). Restore with v86's `initial_state: { url }` option
(this build rejects the older `state:` name) and **the exact memory size
the snapshot was saved with**:

| id | name | memory | state file | size |
|---|---|---|---|---|
| windows98 | Windows 98 | 128 MB | windows98_state-v2.bin.zst | 12.8 MB |
| windows-me | Windows ME | 256 MB | windows-me_state-v3.bin.zst | 27.7 MB |
| windows2000 | Windows 2000 | 512 MB | windows2k_state-v4.bin.zst | 28.2 MB |
| reactos | ReactOS | 512 MB | reactos_state-v3.bin.zst | 17.5 MB |
| serenity | SerenityOS | 512 MB | serenity_state-v4.bin.zst | 16.2 MB |
| haiku | Haiku | 512 MB | haiku_state-v5.bin.zst | 36.5 MB |
| redox | Redox | 1024 MB | redox_state-v2.bin.zst | 15.8 MB |
| 9front | 9front (Plan 9) | 128 MB | 9front_state-v3.bin.zst | 5.2 MB |

Windows 98 was verified end-to-end on `public/v86-test.html?state=…&mem=128`
(restores to the desktop — teal background, icons, taskbar — in ~12 s).
Raw GUI images that are still alive on the CDN (Win 3.1, KolibriOS,
Oberon, NanoShell, …) are catalogued in card `6f2c91aa` if a from-scratch
boot gallery is wanted later.
