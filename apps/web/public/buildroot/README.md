# Buildroot guest images (not committed — `*.bin` is gitignored)

Download from the copy.sh v86 demo hosts (same files the demo presets use):

- `bzimage68.bin` — Linux 6.8.12 bzImage (10,068,480 bytes)
  https://copy.sh/v86/buildroot-bzimage68.bin — demo preset `buildroot6`,
  boots to a busybox shell with
  `earlyprintk=serial,ttyS0,115200 tsc=reliable mitigations=off random.trust_cpu=on`
  (`hooks/useV86.ts` KERNEL_CMDLINE). Verified end-to-end 2026-10-18.
- `buildroot-bzimage.bin` — Linux 5.6.15 bzImage, v86 author's build
  (alternate/test kernel).

The Browser VM boots `bzimage68.bin` in kernel mode; `buildroot.img`/iso are
optional disk images (probed, absent here → selector shows hda/iso n/a).
