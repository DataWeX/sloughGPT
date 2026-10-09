#!/bin/bash
# post-build.sh - SloughGPT Buildroot post-build hook
# This script runs after the rootfs is built but before the image is created

set -e

BOARD_DIR="$(dirname "$0")"
TARGET_DIR="$1"

echo "SloughGPT: Running post-build hooks..."

# Create necessary directories
mkdir -p "$TARGET_DIR/etc/sloughgpt"
mkdir -p "$TARGET_DIR/var/lib/sloughgpt"
mkdir -p "$TARGET_DIR/var/log/sloughgpt"
mkdir -p "$TARGET_DIR/tmp/sloughgpt"

# Install default configuration
if [ -f "$BOARD_DIR/config/dait.conf" ]; then
    cp "$BOARD_DIR/config/dait.conf" "$TARGET_DIR/etc/sloughgpt/"
fi

# Setup default services
mkdir -p "$TARGET_DIR/etc/systemd/system"
if [ -f "$BOARD_DIR/systemd/*.service" ]; then
    cp "$BOARD_DIR/systemd/"*.service "$TARGET_DIR/etc/systemd/system/"
fi

# Setup init scripts
mkdir -p "$TARGET_DIR/etc/init.d"
if [ -f "$BOARD_DIR/init.d/S*" ]; then
    cp "$BOARD_DIR/init.d/S"* "$TARGET_DIR/etc/init.d/"
    chmod +x "$TARGET_DIR/etc/init.d/S"*
fi

# Clean up unnecessary files
rm -rf "$TARGET_DIR/usr/share/doc"
rm -rf "$TARGET_DIR/usr/share/man"
rm -rf "$TARGET_DIR/usr/share/info"

# Set permissions
chmod -R 755 "$TARGET_DIR/etc/init.d"
chmod -R 755 "$TARGET_DIR/usr/bin"

echo "SloughGPT: Post-build hooks completed."
