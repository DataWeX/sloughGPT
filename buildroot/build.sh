#!/bin/bash
# build.sh - Docker-based Buildroot build for SloughGPT
# Usage: ./build.sh [command]
# Commands: build, clean, shell, status

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_IMAGE="sloughgpt-buildroot"
DOCKER_CONTAINER="sloughgpt-build"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[buildroot]${NC} $1"
}

success() {
    echo -e "${GREEN}[buildroot]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[buildroot]${NC} $1"
}

error() {
    echo -e "${RED}[buildroot]${NC} $1"
}

# Build the Docker image
build_docker() {
    log "Building Docker image..."
    docker build -t "$DOCKER_IMAGE" "$SCRIPT_DIR"
    success "Docker image built: $DOCKER_IMAGE"
}

# Run Buildroot build in Docker
build_image() {
    if ! docker image inspect "$DOCKER_IMAGE" &>/dev/null; then
        build_docker
    fi

    log "Building Buildroot image..."
    docker run --rm \
        --name "$DOCKER_CONTAINER" \
        -v "$REPO_ROOT:/home/builder/repo" \
        -w /home/builder/repo/buildroot \
        "$DOCKER_IMAGE" \
        bash -c "make sloughgpt_defconfig && make -j\$(nproc)"

    # Check output
    if [ -f "$SCRIPT_DIR/output/images/rootfs.ext4" ]; then
        success "Build complete: $SCRIPT_DIR/output/images/rootfs.ext4"

        # Create v86 image
        create_v86_image
    else
        error "Build failed - image not found"
        exit 1
    fi
}

# Create v86 compatible image
create_v86_image() {
    log "Creating v86 compatible image..."

    local output_dir="$SCRIPT_DIR/output/images"
    local src_image="$output_dir/rootfs.ext4"
    local v86_image="$output_dir/buildroot.img"

    if [ ! -f "$src_image" ]; then
        error "Source image not found: $src_image"
        exit 1
    fi

    # Copy for v86
    cp "$src_image" "$v86_image"

    # Get size
    local size=$(stat -c%s "$v86_image")
    log "Image size: $size bytes"

    # Create checksum
    md5sum "$v86_image" > "$v86_image.md5"

    # Copy to web public directory
    local web_dir="$REPO_ROOT/apps/web/public/buildroot"
    mkdir -p "$web_dir"
    cp "$v86_image" "$web_dir/"

    success "v86 image created: $v86_image"
    success "Copied to: $web_dir/buildroot.img"
}

# Clean build
clean_build() {
    log "Cleaning build..."
    docker run --rm \
        --name "$DOCKER_CONTAINER-clean" \
        -v "$REPO_ROOT:/home/builder/repo" \
        -w /home/builder/repo/buildroot \
        "$DOCKER_IMAGE" \
        make clean 2>/dev/null || true

    rm -rf "$SCRIPT_DIR/output"
    success "Build directory cleaned"
}

# Open shell in build environment
build_shell() {
    log "Opening build shell..."
    docker run --rm -it \
        --name "$DOCKER_CONTAINER-shell" \
        -v "$REPO_ROOT:/home/builder/repo" \
        -w /home/builder/repo/buildroot \
        "$DOCKER_IMAGE" \
        bash
}

# Show build status
build_status() {
    log "Build Status"
    echo "============"

    # Check Docker image
    if docker image inspect "$DOCKER_IMAGE" &>/dev/null; then
        echo -e "${GREEN}Docker image:${NC} Built"
    else
        echo -e "${RED}Docker image:${NC} Not built"
    fi

    # Check output image
    local image="$SCRIPT_DIR/output/images/rootfs.ext4"
    if [ -f "$image" ]; then
        local size=$(stat -c%s "$image")
        echo -e "${GREEN}Buildroot image:${NC} Built ($(_format_size $size))"
    else
        echo -e "${RED}Buildroot image:${NC} Not built"
    fi

    # Check v86 image
    local v86="$SCRIPT_DIR/output/images/buildroot.img"
    if [ -f "$v86" ]; then
        echo -e "${GREEN}v86 image:${NC} Ready"
    else
        echo -e "${RED}v86 image:${NC} Not created"
    fi

    # Check web directory
    local web="$REPO_ROOT/apps/web/public/buildroot/buildroot.img"
    if [ -f "$web" ]; then
        echo -e "${GREEN}Web image:${NC} Installed"
    else
        echo -e "${RED}Web image:${NC} Not installed"
    fi
}

# Format bytes
_format_size() {
    local bytes=$1
    if [ $bytes -ge 1073741824 ]; then
        echo "$(echo "scale=2; $bytes/1073741824" | bc)GB"
    elif [ $bytes -ge 1048576 ]; then
        echo "$(echo "scale=2; $bytes/1048576" | bc)MB"
    elif [ $bytes -ge 1024 ]; then
        echo "$(echo "scale=2; $bytes/1024" | bc)KB"
    else
        echo "${bytes}B"
    fi
}

# Main
case "${1:-build}" in
    build)
        build_image
        ;;
    clean)
        clean_build
        ;;
    shell)
        build_shell
        ;;
    status)
        build_status
        ;;
    docker)
        build_docker
        ;;
    *)
        echo "Usage: $0 {build|clean|shell|status|docker}"
        exit 1
        ;;
esac
