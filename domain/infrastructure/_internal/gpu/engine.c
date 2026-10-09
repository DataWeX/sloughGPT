/*
 * engine.c — GPU compute engine core.
 *
 * Platform detection, CPU fallback, buffer pool.
 * Platform-specific backends (vulkan.c, metal.c, dx12.c) implement
 * the actual GPU device operations.
 */

#include "engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* ── Platform detection ────────────────────────────────────────────────── */

#if defined(__linux__)
  #define GPU_PLATFORM_LINUX 1
  #define GPU_HAS_VULKAN 1
#elif defined(__APPLE__)
  #define GPU_PLATFORM_MACOS 1
  #define GPU_HAS_METAL 1
  #define GPU_HAS_VULKAN 1  /* via MoltenVK */
#elif defined(_WIN32)
  #define GPU_PLATFORM_WINDOWS 1
  #define GPU_HAS_DX12 1
  #define GPU_HAS_VULKAN 1
#endif

/* ── Forward declarations for platform backends ────────────────────────── */

/* Each platform implements these: */
typedef struct {
    const char* name;
    int  (*init)(GpuDevice* device);
    void (*destroy)(GpuDevice* device);
    GpuBuffer* (*buffer_create)(GpuDevice* device, size_t size, uint32_t usage);
    GpuError  (*buffer_write)(GpuBuffer* buf, const void* data, size_t size, size_t offset);
    GpuError  (*buffer_read)(GpuBuffer* buf, void* data, size_t size, size_t offset);
    void      (*buffer_destroy)(GpuBuffer* buf);
    GpuShader* (*shader_create_wgsl)(GpuDevice* dev, const char* src, size_t len, const char* entry);
    GpuShader* (*shader_create_spirv)(GpuDevice* dev, const uint32_t* code, size_t len, const char* entry);
    void       (*shader_destroy)(GpuShader* shader);
    GpuPipeline* (*pipeline_create)(GpuDevice* dev, GpuShader* sh, const char* entry,
                                    GpuBindEntry* entries, uint32_t num);
    void (*pipeline_destroy)(GpuPipeline* pipeline);
    GpuContext* (*compute_begin)(GpuDevice* dev);
    void (*compute_bind_pipeline)(GpuContext* ctx, GpuPipeline* pipeline);
    void (*compute_bind_buffer)(GpuContext* ctx, uint32_t binding, GpuBuffer* buffer);
    void (*compute_set_push)(GpuContext* ctx, const void* data, size_t size);
    void (*compute_dispatch)(GpuContext* ctx, uint32_t x, uint32_t y, uint32_t z);
    GpuError (*compute_end)(GpuContext* ctx);
} GpuBackend;

/* Platform backends (defined in separate files) */
extern const GpuBackend gpu_backend_vulkan;
extern const GpuBackend gpu_backend_metal;
extern const GpuBackend gpu_backend_dx12;
extern const GpuBackend gpu_backend_cpu;

/* ── Device struct ─────────────────────────────────────────────────────── */

struct GpuDevice {
    const GpuBackend* backend;
    void*             backend_data;  /* Platform-specific state */
    char              name[256];
    uint64_t          vram;
    int               compute_units;
};

struct GpuBuffer {
    GpuDevice* device;
    void*      data;      /* Host shadow copy (for CPU backend) */
    void*      gpu_data;  /* GPU-side data (platform-specific handle) */
    size_t     size;
    uint32_t   usage;
    int        mapped;
};

struct GpuShader {
    GpuDevice* device;
    void*      handle;   /* Platform-specific shader handle */
};

struct GpuPipeline {
    GpuDevice* device;
    void*      handle;
};

struct GpuContext {
    GpuDevice*   device;
    GpuPipeline* current_pipeline;
    void*        cmd_buffer;
    void*        bind_group;
    void*        push_data;
    size_t       push_size;
    uint32_t     bound_bindings[16];
    GpuBuffer*   bound_buffers[16];
    uint32_t     num_bindings;
};

/* ── CPU fallback backend ──────────────────────────────────────────────── */

static int cpu_init(GpuDevice* device) {
    strcpy(device->name, "CPU Fallback");
    device->vram = 0;
    device->compute_units = 1;
    return GPU_OK;
}

static void cpu_destroy(GpuDevice* device) {
    (void)device;
}

static GpuBuffer* cpu_buffer_create(GpuDevice* device, size_t size, uint32_t usage) {
    GpuBuffer* buf = calloc(1, sizeof(GpuBuffer));
    if (!buf) return NULL;
    buf->device = device;
    buf->size = size;
    buf->usage = usage;
    buf->data = calloc(1, size);
    if (!buf->data) { free(buf); return NULL; }
    buf->gpu_data = buf->data;  /* Same pointer for CPU */
    return buf;
}

static GpuError cpu_buffer_write(GpuBuffer* buf, const void* data, size_t size, size_t offset) {
    if (offset + size > buf->size) return GPU_ERROR_BUFFER;
    memcpy((char*)buf->data + offset, data, size);
    return GPU_OK;
}

static GpuError cpu_buffer_read(GpuBuffer* buf, void* data, size_t size, size_t offset) {
    if (offset + size > buf->size) return GPU_ERROR_BUFFER;
    memcpy(data, (char*)buf->data + offset, size);
    return GPU_OK;
}

static void cpu_buffer_destroy(GpuBuffer* buf) {
    if (buf) { free(buf->data); free(buf); }
}

static GpuShader* cpu_shader_create_wgsl(GpuDevice* dev, const char* src, size_t len, const char* entry) {
    (void)src; (void)len; (void)entry;
    GpuShader* sh = calloc(1, sizeof(GpuShader));
    if (sh) sh->device = dev;
    return sh;
}

static GpuShader* cpu_shader_create_spirv(GpuDevice* dev, const uint32_t* code, size_t len, const char* entry) {
    (void)code; (void)len; (void)entry;
    return cpu_shader_create_wgsl(dev, "", 0, "");
}

static void cpu_shader_destroy(GpuShader* shader) { free(shader); }

static GpuPipeline* cpu_pipeline_create(GpuDevice* dev, GpuShader* sh, const char* entry,
                                         GpuBindEntry* entries, uint32_t num) {
    (void)sh; (void)entry; (void)entries; (void)num;
    GpuPipeline* p = calloc(1, sizeof(GpuPipeline));
    if (p) p->device = dev;
    return p;
}

static void cpu_pipeline_destroy(GpuPipeline* p) { free(p); }

static GpuContext* cpu_compute_begin(GpuDevice* dev) {
    GpuContext* ctx = calloc(1, sizeof(GpuContext));
    if (ctx) ctx->device = dev;
    return ctx;
}

static void cpu_compute_bind_pipeline(GpuContext* ctx, GpuPipeline* p) {
    ctx->current_pipeline = p;
}

static void cpu_compute_bind_buffer(GpuContext* ctx, uint32_t binding, GpuBuffer* buf) {
    if (ctx->num_bindings < 16) {
        ctx->bound_bindings[ctx->num_bindings] = binding;
        ctx->bound_buffers[ctx->num_bindings] = buf;
        ctx->num_bindings++;
    }
}

static void cpu_compute_set_push(GpuContext* ctx, const void* data, size_t size) {
    free(ctx->push_data);
    ctx->push_data = malloc(size);
    if (ctx->push_data) {
        memcpy(ctx->push_data, data, size);
        ctx->push_size = size;
    }
}

static void cpu_compute_dispatch(GpuContext* ctx, uint32_t x, uint32_t y, uint32_t z) {
    (void)ctx; (void)x; (void)y; (void)z;
    /* CPU dispatch is handled in Python via numpy */
}

static GpuError cpu_compute_end(GpuContext* ctx) {
    free(ctx->push_data);
    free(ctx);
    return GPU_OK;
}

const GpuBackend gpu_backend_cpu = {
    .name = "cpu",
    .init = cpu_init,
    .destroy = cpu_destroy,
    .buffer_create = cpu_buffer_create,
    .buffer_write = cpu_buffer_write,
    .buffer_read = cpu_buffer_read,
    .buffer_destroy = cpu_buffer_destroy,
    .shader_create_wgsl = cpu_shader_create_wgsl,
    .shader_create_spirv = cpu_shader_create_spirv,
    .shader_destroy = cpu_shader_destroy,
    .pipeline_create = cpu_pipeline_create,
    .pipeline_destroy = cpu_pipeline_destroy,
    .compute_begin = cpu_compute_begin,
    .compute_bind_pipeline = cpu_compute_bind_pipeline,
    .compute_bind_buffer = cpu_compute_bind_buffer,
    .compute_set_push = cpu_compute_set_push,
    .compute_dispatch = cpu_compute_dispatch,
    .compute_end = cpu_compute_end,
};

/* ── Platform selection ────────────────────────────────────────────────── */

static const GpuBackend* select_backend(const char* preferred) {
    if (preferred) {
        if (strcmp(preferred, "cpu") == 0) return &gpu_backend_cpu;
#if GPU_HAS_VULKAN
        if (strcmp(preferred, "vulkan") == 0) return &gpu_backend_vulkan;
#endif
#if GPU_HAS_METAL
        if (strcmp(preferred, "metal") == 0) return &gpu_backend_metal;
#endif
#if GPU_HAS_DX12
        if (strcmp(preferred, "dx12") == 0) return &gpu_backend_dx12;
#endif
        fprintf(stderr, "gpu_engine: backend '%s' not available on this platform\n", preferred);
        return &gpu_backend_cpu;
    }

    /* Auto-select best available */
#if GPU_HAS_VULKAN
    return &gpu_backend_vulkan;
#elif GPU_HAS_METAL
    return &gpu_backend_metal;
#elif GPU_HAS_DX12
    return &gpu_backend_dx12;
#else
    return &gpu_backend_cpu;
#endif
}

/* ── Public API ────────────────────────────────────────────────────────── */

GpuDevice* gpu_device_create(void) {
    return gpu_device_create_backend(NULL);
}

GpuDevice* gpu_device_create_backend(const char* backend) {
    GpuDevice* device = calloc(1, sizeof(GpuDevice));
    if (!device) return NULL;

    device->backend = select_backend(backend);
    int err = device->backend->init(device);
    if (err != GPU_OK) {
        fprintf(stderr, "gpu_engine: failed to init backend '%s': %d\n",
                device->backend->name, err);
        free(device);
        return NULL;
    }

    gpu_device_print_info(device);
    return device;
}

const char* gpu_device_name(GpuDevice* device) {
    return device ? device->name : "none";
}

uint64_t gpu_device_vram(GpuDevice* device) {
    return device ? device->vram : 0;
}

int gpu_device_compute_units(GpuDevice* device) {
    return device ? device->compute_units : 0;
}

void gpu_device_destroy(GpuDevice* device) {
    if (!device) return;
    device->backend->destroy(device);
    free(device);
}

/* ── Buffer API ────────────────────────────────────────────────────────── */

GpuBuffer* gpu_buffer_create(GpuDevice* device, size_t size, uint32_t usage) {
    return device->backend->buffer_create(device, size, usage);
}

GpuError gpu_buffer_write(GpuBuffer* buffer, const void* data, size_t size, size_t offset) {
    return buffer->device->backend->buffer_write(buffer, data, size, offset);
}

GpuError gpu_buffer_read(GpuBuffer* buffer, void* data, size_t size, size_t offset) {
    return buffer->device->backend->buffer_read(buffer, data, size, offset);
}

void* gpu_buffer_map(GpuBuffer* buffer) {
    buffer->mapped = 1;
    return buffer->data;
}

void gpu_buffer_unmap(GpuBuffer* buffer) {
    buffer->mapped = 0;
}

void gpu_buffer_destroy(GpuBuffer* buffer) {
    if (!buffer) return;
    buffer->device->backend->buffer_destroy(buffer);
}

/* ── Shader API ────────────────────────────────────────────────────────── */

GpuShader* gpu_shader_create_wgsl(GpuDevice* device, const char* source,
                                   size_t source_len, const char* entry_point) {
    return device->backend->shader_create_wgsl(device, source, source_len, entry_point);
}

GpuShader* gpu_shader_create_spirv(GpuDevice* device, const uint32_t* code,
                                    size_t code_len, const char* entry_point) {
    return device->backend->shader_create_spirv(device, code, code_len, entry_point);
}

void gpu_shader_destroy(GpuShader* shader) {
    if (!shader) return;
    shader->device->backend->shader_destroy(shader);
}

/* ── Pipeline API ──────────────────────────────────────────────────────── */

GpuPipeline* gpu_pipeline_create(GpuDevice* device, GpuShader* shader,
                                  const char* entry_point,
                                  GpuBindEntry* entries, uint32_t num_entries) {
    return device->backend->pipeline_create(device, shader, entry_point, entries, num_entries);
}

void gpu_pipeline_destroy(GpuPipeline* pipeline) {
    if (!pipeline) return;
    pipeline->device->backend->pipeline_destroy(pipeline);
}

/* ── Compute dispatch API ──────────────────────────────────────────────── */

GpuContext* gpu_compute_begin(GpuDevice* device) {
    return device->backend->compute_begin(device);
}

void gpu_compute_bind_pipeline(GpuContext* ctx, GpuPipeline* pipeline) {
    ctx->device->backend->compute_bind_pipeline(ctx, pipeline);
}

void gpu_compute_bind_buffer(GpuContext* ctx, uint32_t binding, GpuBuffer* buffer) {
    ctx->device->backend->compute_bind_buffer(ctx, binding, buffer);
}

void gpu_compute_set_push(GpuContext* ctx, const void* data, size_t size) {
    ctx->device->backend->compute_set_push(ctx, data, size);
}

void gpu_compute_dispatch(GpuContext* ctx, uint32_t x, uint32_t y, uint32_t z) {
    ctx->device->backend->compute_dispatch(ctx, x, y, z);
}

GpuError gpu_compute_end(GpuContext* ctx) {
    return ctx->device->backend->compute_end(ctx);
}

/* ── Buffer Pool ───────────────────────────────────────────────────────── */

GpuBufferPool* gpu_pool_create(GpuDevice* device, uint32_t capacity, size_t min_size) {
    GpuBufferPool* pool = calloc(1, sizeof(GpuBufferPool));
    if (!pool) return NULL;
    pool->device = device;
    pool->capacity = capacity;
    pool->min_size = min_size;
    pool->buffers = calloc(capacity, sizeof(GpuBuffer*));
    pool->count = 0;
    return pool;
}

GpuBuffer* gpu_pool_acquire(GpuBufferPool* pool, size_t min_size) {
    /* Find an unused buffer big enough */
    for (uint32_t i = 0; i < pool->count; i++) {
        if (pool->buffers[i] && pool->buffers[i]->size >= min_size) {
            GpuBuffer* buf = pool->buffers[i];
            pool->buffers[i] = NULL;
            return buf;
        }
    }
    /* Allocate new */
    return gpu_buffer_create(pool->device, min_size,
                             GPU_BUF_STORAGE | GPU_BUF_COPY_SRC | GPU_BUF_COPY_DST);
}

void gpu_pool_release(GpuBufferPool* pool, GpuBuffer* buffer) {
    for (uint32_t i = 0; i < pool->capacity; i++) {
        if (pool->buffers[i] == NULL) {
            pool->buffers[i] = buffer;
            return;
        }
    }
    /* Pool full, destroy */
    gpu_buffer_destroy(buffer);
}

void gpu_pool_destroy(GpuBufferPool* pool) {
    if (!pool) return;
    for (uint32_t i = 0; i < pool->capacity; i++) {
        if (pool->buffers[i]) gpu_buffer_destroy(pool->buffers[i]);
    }
    free(pool->buffers);
    free(pool);
}

/* ── Utilities ─────────────────────────────────────────────────────────── */

size_t gpu_uniform_alignment(GpuDevice* device) {
    (void)device;
    return 256;  /* Safe default for all platforms */
}

void gpu_device_print_info(GpuDevice* device) {
    if (!device) return;
    fprintf(stderr, "gpu_engine: backend=%s device=%s",
            device->backend->name, device->name);
    if (device->vram > 0) {
        fprintf(stderr, " vram=%.1fMB", device->vram / (1024.0 * 1024.0));
    }
    fprintf(stderr, "\n");
}
