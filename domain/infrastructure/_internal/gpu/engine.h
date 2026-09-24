/*
 * gpu_engine.h — Platform-agnostic GPU compute engine.
 *
 * Minimal C API for managing GPU devices, buffers, and compute pipelines.
 * Each platform (Vulkan, Metal, DX12) implements this interface.
 *
 * Design principles:
 *   - Zero ML framework dependencies
 *   - Buffer pool for reuse (no alloc per dispatch)
 *   - Shader modules loaded from WGSL source
 *   - Synchronous dispatch for simplicity (async later)
 */

#ifndef GPU_ENGINE_H
#define GPU_ENGINE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque handles ────────────────────────────────────────────────────── */

typedef struct GpuDevice    GpuDevice;
typedef struct GpuBuffer    GpuBuffer;
typedef struct GpuShader    GpuShader;
typedef struct GpuPipeline  GpuPipeline;
typedef struct GpuContext   GpuContext;

/* ── Error codes ───────────────────────────────────────────────────────── */

typedef enum {
    GPU_OK                   =  0,
    GPU_ERROR_NO_DEVICE      = -1,
    GPU_ERROR_NO_MEMORY      = -2,
    GPU_ERROR_SHADER_COMPILE = -3,
    GPU_ERROR_PIPELINE       = -4,
    GPU_ERROR_BUFFER         = -5,
    GPU_ERROR_DISPATCH       = -6,
    GPU_ERROR_UNSUPPORTED    = -7,
} GpuError;

/* ── Buffer usage flags ────────────────────────────────────────────────── */

typedef enum {
    GPU_BUF_STORAGE  = (1 << 0),  /* Shader read/write storage buffer */
    GPU_BUF_UNIFORM  = (1 << 1),  /* Uniform buffer (read-only) */
    GPU_BUF_VERTEX   = (1 << 2),  /* Vertex buffer */
    GPU_BUF_COPY_SRC = (1 << 3),  /* Copy source */
    GPU_BUF_COPY_DST = (1 << 4),  /* Copy destination */
} GpuBufferUsage;

/* ── Device ────────────────────────────────────────────────────────────── */

/*
 * Create a GPU device. Selects the best available backend.
 * Returns NULL if no GPU is available.
 *
 * Priority: Vulkan > Metal > DX12 > OpenGL > CPU fallback
 */
GpuDevice* gpu_device_create(void);

/*
 * Create a device with a specific backend.
 * backend: "vulkan", "metal", "dx12", "cpu"
 */
GpuDevice* gpu_device_create_backend(const char* backend);

/* Get device name (e.g. "NVIDIA GeForce RTX 4090") */
const char* gpu_device_name(GpuDevice* device);

/* Get device info */
uint64_t gpu_device_vram(GpuDevice* device);  /* Total VRAM in bytes */
int      gpu_device_compute_units(GpuDevice* device);

/* Destroy device and release all resources */
void gpu_device_destroy(GpuDevice* device);

/* ── Buffers ───────────────────────────────────────────────────────────── */

/* Create a GPU buffer */
GpuBuffer* gpu_buffer_create(
    GpuDevice* device,
    size_t     size,
    uint32_t   usage  /* GpuBufferUsage flags */
);

/* Write data from host to GPU buffer */
GpuError gpu_buffer_write(
    GpuBuffer* buffer,
    const void* data,
    size_t      size,
    size_t      offset
);

/* Read data from GPU buffer to host */
GpuError gpu_buffer_read(
    GpuBuffer* buffer,
    void*      data,
    size_t     size,
    size_t     offset
);

/* Map buffer for direct CPU access (if supported) */
void* gpu_buffer_map(GpuBuffer* buffer);
void  gpu_buffer_unmap(GpuBuffer* buffer);

/* Destroy buffer */
void gpu_buffer_destroy(GpuBuffer* buffer);

/* ── Shaders ───────────────────────────────────────────────────────────── */

/* Create a shader module from WGSL source */
GpuShader* gpu_shader_create_wgsl(
    GpuDevice*    device,
    const char*   source,
    size_t        source_len,
    const char*   entry_point  /* e.g. "main" */
);

/* Create a shader module from SPIR-V binary */
GpuShader* gpu_shader_create_spirv(
    GpuDevice*    device,
    const uint32_t* code,
    size_t         code_len,
    const char*    entry_point
);

/* Destroy shader */
void gpu_shader_destroy(GpuShader* shader);

/* ── Compute Pipelines ─────────────────────────────────────────────────── */

/* Bind group layout entry */
typedef struct {
    uint32_t binding;
    uint32_t type;       /* 0=storage, 1=uniform, 2=texture, 3=sampler */
    uint32_t stages;     /* 1=vert, 2=frag, 4=comp */
} GpuBindEntry;

/* Create a compute pipeline */
GpuPipeline* gpu_pipeline_create(
    GpuDevice*      device,
    GpuShader*      shader,
    const char*     entry_point,
    GpuBindEntry*   entries,
    uint32_t        num_entries
);

/* Destroy pipeline */
void gpu_pipeline_destroy(GpuPipeline* pipeline);

/* ── Compute Dispatch ──────────────────────────────────────────────────── */

/* Begin a compute pass */
GpuContext* gpu_compute_begin(GpuDevice* device);

/* Bind a pipeline */
void gpu_compute_bind_pipeline(GpuContext* ctx, GpuPipeline* pipeline);

/* Bind a buffer to a binding point */
void gpu_compute_bind_buffer(
    GpuContext* ctx,
    uint32_t    binding,
    GpuBuffer*  buffer
);

/* Set push constants / uniforms */
void gpu_compute_set_push(
    GpuContext* ctx,
    const void* data,
    size_t      size
);

/* Dispatch workgroups */
void gpu_compute_dispatch(
    GpuContext* ctx,
    uint32_t    x,
    uint32_t    y,
    uint32_t    z
);

/* End compute pass and submit */
GpuError gpu_compute_end(GpuContext* ctx);

/* ── Buffer Pool ───────────────────────────────────────────────────────── */

/* Pre-allocated buffer pool for reuse */
typedef struct {
    GpuDevice* device;
    GpuBuffer** buffers;
    uint32_t    count;
    uint32_t    capacity;
    size_t      min_size;
} GpuBufferPool;

GpuBufferPool* gpu_pool_create(GpuDevice* device, uint32_t capacity, size_t min_size);
GpuBuffer*     gpu_pool_acquire(GpuBufferPool* pool, size_t min_size);
void           gpu_pool_release(GpuBufferPool* pool, GpuBuffer* buffer);
void           gpu_pool_destroy(GpuBufferPool* pool);

/* ── Utilities ─────────────────────────────────────────────────────────── */

/* Align size to alignment boundary */
static inline size_t gpu_align(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/* Get required alignment for uniform buffers */
size_t gpu_uniform_alignment(GpuDevice* device);

/* Print device info to stderr */
void gpu_device_print_info(GpuDevice* device);

#ifdef __cplusplus
}
#endif

#endif /* GPU_ENGINE_H */
