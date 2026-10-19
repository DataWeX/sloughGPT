# GUI desktop saved-states (fetched — not committed)

v86 saved-state snapshots of GUI desktop OSes from the copy.sh demo,
fetched by `scripts/fetch_gui_images.sh` (`*.zst` is gitignored). The
browser cannot fetch these directly: `c.copy.sh` answers cross-origin
requests with 403 (hotlink protection), so they are vendored same-origin.

A state is a full machine snapshot (memory + devices + sparse disk) — it
restores **without** the original disk image, which is how these desktops
stay bootable even though their raw image URLs 404 on both copy.sh hosts
(verified 2026-10-19). Restore with v86's `initial_state: { url }` option
(this build rejects the older `state:` name) and the exact save-time
machine profile: **memory size, an async-file hda, and the matching NIC
class** (`bc:"virtio"` in copy.sh profiles = virtio-net, not storage).
The per-preset table lives in `lib/v86-gui-states.ts` and drives the
panel's "Desktop states" selector; this README mirrors it:

| id | name | memory | NIC | chunk | hda size | state file |
|---|---|---|---|---|---|---|
| windows98 | Windows 98 | 128 MB | ne2k | 256 KiB | 300 MiB | windows98_state-v2.bin.zst |
| windows-me | Windows ME | 256 MB | ne2k | 256 KiB | 1024 MiB | windows-me_state-v3.bin.zst |
| windows2000 | Windows 2000 | 512 MB | ne2k | 256 KiB | 2048 MiB | windows2k_state-v4.bin.zst |
| reactos | ReactOS | 512 MB | virtio | 1 MiB | 700 MiB | reactos_state-v3.bin.zst |
| serenity | SerenityOS | 512 MB | ne2k | 1 MiB | 700 MiB | serenity_state-v4.bin.zst |
| haiku | Haiku | 512 MB | ne2k | 1 MiB | 1280 MiB | haiku_state-v5.bin.zst |
| redox | Redox | 1024 MB | ne2k | 1 MiB | 640 MiB | redox_state-v2.bin.zst |
| 9front | 9front (Plan 9) | 128 MB | ne2k | 1 MiB | 467 MiB (iso) | 9front_state-v3.bin.zst |

Windows 98 and ReactOS were verified end-to-end on
`public/v86-test.html?state=…` (restore → running CPU, painted screen,
working keyboard input; ReactOS additionally needs `&net=virtio`).

**The hda images are NOT vendored yet** (user decision 2026-10-19): a
snapshot only serializes guest-*written* sectors, so a desktop boots and
idles live, but file reads beyond that Range-fetch `/gui/disks/<id>.img`
and stall on 404. When vendoring is approved: the images exist on
`c.copy.sh` as byte-range part files (`<dir>/<start>-<end>.img`, e.g.
`windows98/0-262144.img` — the dotfile URL `windows98/.img` 404s; that
was the whole hunt), assembled in order into one file per preset at the
`hdaUrl` slot above. Raw GUI images that are still alive on the CDN
(Win 3.1, KolibriOS, Oberon, NanoShell, …) are catalogued in card
`6f2c91aa` if a from-scratch boot gallery is wanted later.
