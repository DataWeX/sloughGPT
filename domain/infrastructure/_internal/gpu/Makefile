# Makefile for sloughGPT GPU engine
#
# Targets:
#   make all        — Build shared library + compile shaders
#   make lib        — Build libgpu_engine.so (Linux) / .dylib (macOS) / .dll (Windows)
#   make shaders    — Compile WGSL → SPIR-V + HLSL + MSL
#   make clean      — Remove build artifacts
#   make test       — Build and run basic smoke test

# ── Config ───────────────────────────────────────────────────────────────

CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -fPIC -std=c11
LDFLAGS ?=

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Sources
GPU_DIR := $(shell pwd)
SOURCES_C := engine.c
SOURCES_PLATFORM :=

# Libraries
LIBS :=

ifeq ($(UNAME_S),Linux)
    LIB_EXT := .so
    SOURCES_PLATFORM += vulkan.c
    LIBS += -lvulkan -ldl
    LDFLAGS += -shared
endif

ifeq ($(UNAME_S),Darwin)
    LIB_EXT := .dylib
    SOURCES_PLATFORM += metal.c
    LIBS += -framework Metal -framework Foundation -lobjc -ldl
    LDFLAGS += -shared -undefined dynamic_lookup
endif

# Windows cross-compile or native
ifdef WINDIR
    LIB_EXT := .dll
    SOURCES_PLATFORM += dx12.c
    LIBS += -ld3d12 -ldxgi
    LDFLAGS += -shared
endif

# Output
LIB_NAME := libgpu_engine
LIB_TARGET := $(LIB_NAME)$(LIB_EXT)

# Shader compiler
NAGA ?= naga
SHADERS := matmul softmax rmsnorm rope silu gelu
SHADER_WGSL := $(addprefix shaders/,$(addsuffix .wgsl,$(SHADERS)))
SHADER_SPV  := $(addprefix shaders/,$(addsuffix .spv,$(SHADERS)))
SHADER_HLSL := $(addprefix shaders/,$(addsuffix .hlsl,$(SHADERS)))
SHADER_MSL  := $(addprefix shaders/,$(addsuffix .metal,$(SHADERS)))

# ── Targets ──────────────────────────────────────────────────────────────

.PHONY: all lib shaders clean test

all: lib shaders

lib: $(LIB_TARGET)

$(LIB_TARGET): $(SOURCES_C) $(SOURCES_PLATFORM) engine.h
	@echo "Building $(LIB_TARGET)..."
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SOURCES_C) $(SOURCES_PLATFORM) $(LIBS)
	@echo "✓ Built $(LIB_TARGET)"

shaders: $(SHADER_SPV) $(SHADER_HLSL) $(SHADER_MSL)

shaders/%.spv: shaders/%.wgsl
	@echo "  SPIR-V: $<"
	@$(NAGA) $< $@ 2>/dev/null || echo "  (SPIR-V skipped — install naga: cargo install naga-cli)"

shaders/%.hlsl: shaders/%.wgsl
	@echo "  HLSL: $<"
	@$(NAGA) $< $@ 2>/dev/null || echo "  (HLSL skipped — install naga: cargo install naga-cli)"

shaders/%.metal: shaders/%.wgsl
	@echo "  MSL: $<"
	@$(NAGA) $< $@ 2>/dev/null || echo "  (MSL skipped — install naga: cargo install naga-cli)"

test: lib
	@echo "Running smoke test..."
	@python3 -c "from domains.infrastructure.gpu.gpu_engine import is_gpu_available; print('GPU available:', is_gpu_available())"

clean:
	rm -f $(LIB_TARGET)
	rm -f shaders/*.spv shaders/*.hlsl shaders/*.metal
	@echo "✓ Cleaned"
