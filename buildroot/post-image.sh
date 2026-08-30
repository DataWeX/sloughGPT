#!/bin/bash
# post-image.sh - SloughGPT Buildroot post-image hook
# This script runs after the image is created to prepare for v86

set -e

OUTPUT_DIR="$1"
IMAGE_FILE="$OUTPUT_DIR/images/rootfs.ext4"

echo "SloughGPT: Running post-image hooks..."

# Check if image exists
if [ ! -f "$IMAGE_FILE" ]; then
    echo "SloughGPT: Error - Image file not found: $IMAGE_FILE"
    exit 1
fi

# Create v86 compatible image
V86_IMAGE="$OUTPUT_DIR/images/buildroot.img"
echo "SloughGPT: Creating v86 compatible image..."

# Copy the ext4 image
cp "$IMAGE_FILE" "$V86_IMAGE"

# Get image size in bytes
IMAGE_SIZE=$(stat -c%s "$V86_IMAGE")
echo "SloughGPT: Image size: $IMAGE_SIZE bytes"

# Create a simple MBR partition table for v86
# Note: v86 can boot from raw disk images without partition tables
# but having one improves compatibility

# For v86, we can use the raw image directly
# The v86 emulator will handle the filesystem

echo "SloughGPT: v86 image created at: $V86_IMAGE"

# Create checksum
md5sum "$V86_IMAGE" > "$V86_IMAGE.md5"
echo "SloughGPT: Checksum created: $V86_IMAGE.md5"

# Create a manifest
cat > "$OUTPUT_DIR/images/MANIFEST.txt" << EOF
SloughGPT Buildroot Image
========================
Version: 0.1.0
Date: $(date)
Image: buildroot.img
Size: $IMAGE_SIZE bytes
MD5: $(cat "$V86_IMAGE.md5" | awk '{print $1}')

Files in image:
$(ls -la "$TARGET_DIR" 2>/dev/null || echo "Unable to list files")

Boot instructions:
1. Load BIOS: /bios/seabios.bin
2. Load VGA BIOS: /bios/vgabios.bin
3. Load WASM: /v86/v86.wasm
4. Load image: buildroot.img
5. Set memory: 256MB

EOF

echo "SloughGPT: Post-image hooks completed."
