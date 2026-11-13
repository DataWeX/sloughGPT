# SloughGPT Buildroot

Custom Buildroot configuration for building a minimal Linux image
for the v86 browser VM and x86 VM.

## Overview

This directory contains the Buildroot configuration to build a minimal
Linux image with:

- x86_64 architecture
- Linux kernel 6.1.x
- BusyBox shell environment
- Python 3 (for Dait shell integration)
- Custom SloughGPT packages
- Init system with proper service management

## Directory Structure

```
buildroot/
├── configs/
│   ├── sloughgpt_defconfig          # Buildroot defconfig
│   ├── linux-soughgpt.defconfig     # Linux kernel config
│   └── busybox-soughgpt.defconfig   # BusyBox config
├── overlays/
│   ├── etc/
│   │   ├── init.d/                  # Init scripts
│   │   ├── fstab                    # Mount table
│   │   └── profile                  # Shell profile
│   ├── usr/bin/                     # Custom binaries
│   └── root/                        # Root home
├── packages/
│   └── sloughgpt/
│       ├── sloughgpt.mk             # Package makefile
│       └── Config.in                # Kconfig entry
├── post-build.sh                    # Rootfs customization
├── post-image.sh                    # Image creation
└── README.md                        # This file
```

## Prerequisites

- Buildroot 2024.02.x or later
- Docker (optional, for reproducible builds)
- ~10GB free disk space
- Internet connection (for downloading Buildroot)

## Quick Start

### 1. Clone Buildroot

```bash
cd /home/mana/Documents/Default\ Project/sloughGPT
git clone https://github.com/buildroot/buildroot.git --depth 1 --branch 2024.02.x
cd buildroot
```

### 2. Copy Configuration

```bash
cp ../buildroot/configs/sloughgpt_defconfig configs/
cp ../buildroot/configs/linux-soughgpt.defconfig configs/
cp ../buildroot/configs/busybox-soughgpt.defconfig configs/
```

### 3. Build

```bash
make sloughgpt_defconfig
make
```

### 4. Output

The built image will be at:
```
buildroot/output/images/rootfs.ext4
```

## v86 Integration

After building, convert the image for v86:

```bash
# Copy to web public directory
cp buildroot/output/images/rootfs.ext4 apps/web/public/buildroot/buildroot.img

# Update the v86 hook to use local image
# In apps/web/hooks/useV86.ts, change LINUX_IMAGE_URL to:
# const LINUX_IMAGE_URL = '/buildroot/buildroot.img'
```

## Customization

### Adding Packages

1. Create package directory in `packages/sloughgpt/`
2. Add `.mk` file with build instructions
3. Add `Config.in` for Kconfig integration
4. Update `sloughgpt_defconfig` to include package

### Modifying Init

Edit files in `overlays/etc/init.d/` to change boot behavior.

### Adding Services

Create systemd unit files in `overlays/etc/systemd/system/` or
sysv init scripts in `overlays/etc/init.d/`.

## Testing

### Test in v86

1. Build the image
2. Start the web dev server: `make web`
3. Navigate to `/vm` page
4. Select "Linux" tab
5. The custom image should boot

### Test in x86 VM

1. Build the image
2. Use the x86 VM to load the image
3. Boot and test Dait shell integration

## Troubleshooting

### Build Fails

- Ensure you have enough disk space
- Check internet connection for downloading packages
- Verify Buildroot version compatibility

### Image Won't Boot

- Check kernel config matches your hardware
- Verify rootfs is properly formatted
- Check init scripts have correct permissions

### v86 Issues

- Ensure image is raw ext4 format
- Check BIOS and VGA BIOS paths are correct
- Verify WASM file is accessible

## Architecture Notes

### Why Buildroot?

- Minimal footprint (~128MB image)
- Fast boot time (~2-3 seconds in v86)
- Full control over packages
- Reproducible builds

### v86 Compatibility

The image is designed for the v86 x86 emulator:
- x86_64 architecture
- VGA text mode support
- IDE disk interface
- 256MB RAM requirement

### Dait Integration

The image includes:
- Python 3 for Dait shell
- Custom init scripts for Dait services
- Persistent storage for VM state
