/*
 * metal.c — Metal compute backend for gpu_engine (macOS/iOS).
 *
 * Uses Objective-C runtime to call Metal APIs from pure C.
 * No Xcode project needed — compiles with any C compiler on macOS.
 *
 * Metal shading language: .metal files compiled to metallib at build time.
 * This backend accepts pre-compiled metallib binaries.
 */

#include "engine.h"

#ifdef __APPLE__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

/* Objective-C runtime (always available on macOS) */
typedef struct objc_object* id;
typedef id (*objc_msg_send_fn)(id, const char*, ...);
typedef id (*objc_alloc_fn)(id);
typedef Class (*objc_get_class_fn)(const char*);
typedef id (*objc_autorelease_pool_push_fn)(void);
typedef void (*objc_autorelease_pool_pop_fn)(id);

static objc_msg_send_fn     objc_msgSend;
static objc_get_class_fn    objc_getClass;
static objc_alloc_fn        objc_alloc;
static objc_autorelease_pool_push_fn autorelease_push;
static objc_autorelease_pool_pop_fn  autorelease_pop;

static int metal_runtime_init(void) {
    void* handle = dlopen("/usr/lib/libobjc.A.dylib", RTLD_NOW);
    if (!handle) return -1;

    objc_msgSend = (objc_msg_send_fn)dlsym(handle, "objc_msgSend");
    objc_getClass = (objc_get_class_fn)dlsym(handle, "objc_getClass");
    objc_alloc = (objc_alloc_fn)dlsym(handle, "objc_alloc");
    autorelease_push = (objc_autorelease_pool_push_fn)dlsym(handle, "objc_autoreleasePush");
    autorelease_pop = (objc_autorelease_pool_pop_fn)dlsym(handle, "objc_autoreleasePop");

    return (objc_msgSend && objc_getClass) ? 0 : -1;
}

/* ── Metal state ───────────────────────────────────────────────────────── */

typedef struct {
    id device;           /* MTLDevice */
    id command_queue;    /* MTLCommandQueue */
    id library;          /* MTLLibrary (pre-compiled metallib) */
    id autorelease_pool;

    char name[256];
    uint64_t vram;
} MetalState;

typedef struct {
    id buffer;    /* MTLBuffer */
    size_t size;
    void* pointer;  /* MTLBuffer.contents */
} MetalBuffer;

typedef struct {
    id function;   /* MTLFunction */
} MetalShader;

typedef struct {
    id pipeline;       /* MTLComputePipelineState */
    id desc_layout;    /* unused for Metal, but kept for interface compat */
} MetalPipeline;

typedef struct {
    id command_buffer;
    id compute_encoder;
    id autorel;
} MetalContext;

/* ── Device init ───────────────────────────────────────────────────────── */

static int metal_init(GpuDevice* device) {
    if (metal_runtime_init() != 0) {
        fprintf(stderr, "metal: failed to load Objective-C runtime\n");
        return GPU_ERROR_NO_DEVICE;
    }

    MetalState* s = calloc(1, sizeof(MetalState));
    device->backend_data = s;

    s->autorelease_pool = autorelease_push();

    /* Get default Metal device */
    Class MTLCreateSystemDefaultDevice = objc_getClass("MTLCreateSystemDefaultDevice");
    s->device = objc_msgSend((id)MTLCreateSystemDefaultDevice, sel("init"));
    if (!s->device) {
        fprintf(stderr, "metal: no Metal device available\n");
        return GPU_ERROR_NO_DEVICE;
    }

    /* Get device name */
    id name_id = objc_msgSend(s->device, sel("name"));
    const char* name_str = objc_msgSend(name_id, sel("UTF8String"));
    strncpy(s->name, name_str, sizeof(s->name) - 1);

    /* Get VRAM */
    s->vram = (uint64_t)objc_msgSend(s->device, sel("recommendedMaxWorkingSetSize"));

    /* Create command queue */
    s->command_queue = objc_msgSend(s->device, sel("newCommandQueue"));

    device->vram = s->vram;
    strncpy(device->name, s->name, sizeof(device->name) - 1);
    device->compute_units = 1;  /* Metal doesn't expose this directly */

    fprintf(stderr, "metal: %s (%.1f MB)\n", s->name, s->vram / (1024.0 * 1024.0));

    return GPU_OK;
}

static void metal_destroy(GpuDevice* device) {
    MetalState* s = device->backend_data;
    if (!s) return;

    /* Release Metal objects */
    if (s->command_queue) objc_msgSend(s->command_queue, sel("release"));
    if (s->device) objc_msgSend(s->device, sel("release"));
    if (s->autorelease_pool) autorelease_pop(s->autorelease_pool);

    free(s);
}

/* ── Buffers ───────────────────────────────────────────────────────────── */

static GpuBuffer* metal_buffer_create(GpuDevice* device, size_t size, uint32_t usage) {
    MetalState* s = device->backend_data;

    MetalBuffer* mbuf = calloc(1, sizeof(MetalBuffer));
    mbuf->size = size;

    /* Storage mode shared (CPU + GPU accessible) */
    unsigned long storage_mode = 0;  /* MTLResourceStorageModeShared = 0 */

    mbuf->buffer = objc_msgSend(
        objc_alloc(objc_getClass("MTLBuffer")),
        sel("initWithBytes:length:options:"),
        NULL, size, storage_mode
    );

    if (!mbuf->buffer) {
        free(mbuf);
        return NULL;
    }

    mbuf->pointer = objc_msgSend(mbuf->buffer, sel("contents"));

    GpuBuffer* result = calloc(1, sizeof(GpuBuffer));
    result->device = device;
    result->gpu_data = mbuf;
    result->size = size;
    result->usage = usage;
    result->data = mbuf->pointer;
    return result;
}

static GpuResult metal_buffer_write(GpuBuffer* buffer, const void* data, size_t size, size_t offset) {
    MetalBuffer* mbuf = buffer->gpu_data;
    memcpy((char*)mbuf->pointer + offset, data, size);
    return GPU_OK;
}

static GpuResult metal_buffer_read(GpuBuffer* buffer, void* data, size_t size, size_t offset) {
    MetalBuffer* mbuf = buffer->gpu_data;
    memcpy(data, (char*)mbuf->pointer + offset, size);
    return GPU_OK;
}

static void metal_buffer_destroy(GpuBuffer* buffer) {
    if (!buffer) return;
    MetalBuffer* mbuf = buffer->gpu_data;
    if (mbuf) {
        if (mbuf->buffer) objc_msgSend(mbuf->buffer, sel("release"));
        free(mbuf);
    }
    free(buffer);
}

/* ── Shaders ───────────────────────────────────────────────────────────── */

static GpuShader* metal_shader_create_wgsl(GpuDevice* device, const char* source,
                                             size_t source_len, const char* entry_point) {
    (void)source; (void)source_len; (void)entry_point;
    /* Metal uses .metal shading language, not WGSL */
    fprintf(stderr, "metal: WGSL not supported, use .metal shaders\n");
    return NULL;
}

static GpuShader* metal_shader_create_spirv(GpuDevice* device, const uint32_t* code,
                                              size_t code_len, const char* entry_point) {
    /* Metal doesn't use SPIR-V — it uses metallib */
    (void)code; (void)code_len; (void)entry_point;
    fprintf(stderr, "metal: SPIR-V not supported, use metallib\n");
    return NULL;
}

static GpuShader* metal_shader_create_metallib(GpuDevice* device, const void* metallib_data,
                                                 size_t metallib_size, const char* function_name) {
    MetalState* s = device->backend_data;

    /* Create library from metallib data */
    id lib = objc_msgSend(
        objc_alloc(objc_getClass("MTLLibrary")),
        sel("initWithData:error:"),
        metallib_data, metallib_size, NULL
    );
    if (!lib) {
        fprintf(stderr, "metal: failed to create library from metallib\n");
        return NULL;
    }

    /* Get function */
    id fname = objc_msgSend(objc_alloc(objc_getClass("NSString")),
                            sel("initWithUTF8String:"), function_name);
    id func = objc_msgSend(lib, sel("newFunctionWithName:"), fname);

    MetalShader* shader = calloc(1, sizeof(MetalShader));
    shader->function = func;

    GpuShader* result = calloc(1, sizeof(GpuShader));
    result->device = device;
    result->handle = shader;
    return result;
}

static void metal_shader_destroy(GpuShader* shader) {
    if (!shader) return;
    MetalShader* s = shader->handle;
    if (s) {
        if (s->function) objc_msgSend(s->function, sel("release"));
        free(s);
    }
    free(shader);
}

/* ── Pipelines ─────────────────────────────────────────────────────────── */

static GpuPipeline* metal_pipeline_create(GpuDevice* device, GpuShader* shader,
                                           const char* entry_point,
                                           GpuBindEntry* entries, uint32_t num_entries) {
    MetalState* s = device->backend_data;
    MetalShader* sh = shader->handle;

    /* Create compute pipeline */
    id pipeline = objc_msgSend(s->device, sel("newComputePipelineStateWithFunction:error:"),
                               sh->function, NULL);

    MetalPipeline* pipe = calloc(1, sizeof(MetalPipeline));
    pipe->pipeline = pipeline;

    GpuPipeline* result = calloc(1, sizeof(GpuPipeline));
    result->device = device;
    result->handle = pipe;
    return result;
}

static void metal_pipeline_destroy(GpuPipeline* pipeline) {
    if (!pipeline) return;
    MetalPipeline* p = pipeline->handle;
    if (p) {
        if (p->pipeline) objc_msgSend(p->pipeline, sel("release"));
        free(p);
    }
    free(pipeline);
}

/* ── Compute dispatch ──────────────────────────────────────────────────── */

static GpuContext* metal_compute_begin(GpuDevice* device) {
    MetalState* s = device->backend_data;

    MetalContext* ctx = calloc(1, sizeof(MetalContext));
    ctx->autorel = autorelease_push();

    id cmd_buf = objc_msgSend(s->command_queue, sel("commandBuffer"));
    ctx->command_buffer = cmd_buf;
    ctx->compute_encoder = objc_msgSend(cmd_buf, sel("computeCommandEncoder"));

    GpuContext* result = calloc(1, sizeof(GpuContext));
    result->device = device;
    result->handle = ctx;
    return result;
}

static void metal_compute_bind_pipeline(GpuContext* ctx, GpuPipeline* pipeline) {
    MetalContext* mc = ctx->handle;
    MetalPipeline* mp = pipeline->handle;
    objc_msgSend(mc->compute_encoder, sel("setComputePipelineState:"), mp->pipeline);
}

static void metal_compute_bind_buffer(GpuContext* ctx, uint32_t binding, GpuBuffer* buffer) {
    MetalContext* mc = ctx->handle;
    MetalBuffer* mb = buffer->gpu_data;
    objc_msgSend(mc->compute_encoder, sel("setBuffer:offset:atIndex:"),
                 mb->buffer, 0, binding);
}

static void metal_compute_set_push(GpuContext* ctx, const void* data, size_t size) {
    MetalContext* mc = ctx->handle;
    /* Metal doesn't have push constants — use a buffer instead */
    /* For simplicity, set bytes directly (up to 4KB) */
    objc_msgSend(mc->compute_encoder, sel("setBytes:length:atIndex:"),
                 data, size, 31);  /* Use a high binding index */
}

static void metal_compute_dispatch(GpuContext* ctx, uint32_t x, uint32_t y, uint32_t z) {
    MetalContext* mc = ctx->handle;
    /* Metal uses threadsPerThreadgroup, not workgroup counts */
    /* Assume 256 threads per threadgroup for now */
    unsigned long threads[3] = {256, 1, 1};
    unsigned long groups[3] = {(x + 255) / 256, y, z};

    /* For 2D dispatches (matmul), adjust threadgroup size */
    if (y > 1 && z <= 1) {
        threads[0] = 16;
        threads[1] = 16;
        threads[2] = 1;
        groups[0] = (x + 15) / 16;
        groups[1] = (y + 15) / 16;
    }

    id tg = objc_msgSend(objc_alloc(objc_getClass("MTLSize")),
                         sel("initWithWidth:height:depth:"),
                         threads[0], threads[1], threads[2]);
    id grid = objc_msgSend(objc_alloc(objc_getClass("MTLSize")),
                           sel("initWithWidth:height:depth:"),
                           groups[0], groups[1], groups[2]);

    objc_msgSend(mc->compute_encoder, sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                 grid, tg);
}

static GpuResult metal_compute_end(GpuContext* ctx) {
    MetalContext* mc = ctx->handle;

    objc_msgSend(mc->compute_encoder, sel("endEncoding"));
    objc_msgSend(mc->command_buffer, sel("commit"));
    objc_msgSend(mc->command_buffer, sel("waitUntilCompleted"));

    autorelease_pop(mc->autorel);
    free(mc);
    free(ctx);
    return GPU_OK;
}

/* ── Backend registration ──────────────────────────────────────────────── */

const GpuBackend gpu_backend_metal = {
    .name = "metal",
    .init = metal_init,
    .destroy = metal_destroy,
    .buffer_create = metal_buffer_create,
    .buffer_write = (void*)metal_buffer_write,
    .buffer_read = (void*)metal_buffer_read,
    .buffer_destroy = metal_buffer_destroy,
    .shader_create_wgsl = metal_shader_create_wgsl,
    .shader_create_spirv = metal_shader_create_spirv,
    .shader_destroy = metal_shader_destroy,
    .pipeline_create = metal_pipeline_create,
    .pipeline_destroy = metal_pipeline_destroy,
    .compute_begin = metal_compute_begin,
    .compute_bind_pipeline = metal_compute_bind_pipeline,
    .compute_bind_buffer = metal_compute_bind_buffer,
    .compute_set_push = metal_compute_set_push,
    .compute_dispatch = metal_compute_dispatch,
    .compute_end = (void*)metal_compute_end,
};

/* Helper: create shader from .metal source (compiled at runtime via metallib) */
GpuShader* metal_shader_from_source(GpuDevice* device, const char* source,
                                     size_t source_len, const char* function_name) {
    MetalState* s = device->backend_data;

    /* Compile .metal source to library */
    id source_str = objc_msgSend(objc_alloc(objc_getClass("NSString")),
                                 sel("initWithBytes:length:encoding:"),
                                 source, source_len, 4 /* NSUTF8StringEncoding */);

    id lib = objc_msgSend(s->device, sel("newLibraryWithSource:error:"),
                          source_str, NULL);
    if (!lib) {
        fprintf(stderr, "metal: failed to compile shader source\n");
        return NULL;
    }

    id fname = objc_msgSend(objc_alloc(objc_getClass("NSString")),
                            sel("initWithUTF8String:"), function_name);
    id func = objc_msgSend(lib, sel("newFunctionWithName:"), fname);

    MetalShader* shader = calloc(1, sizeof(MetalShader));
    shader->function = func;

    GpuShader* result = calloc(1, sizeof(GpuShader));
    result->device = device;
    result->handle = shader;
    return result;
}

#endif /* __APPLE__ */
