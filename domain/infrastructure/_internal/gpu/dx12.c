/*
 * dx12.c — DX12 compute backend for gpu_engine (Windows).
 *
 * Minimal Direct3D 12 compute pipeline.
 * Uses D3D12 API directly, no D3DX, no helper libraries.
 *
 * HLSL shaders compiled to DXBC at build time via fxc.
 */

#include "engine.h"

#ifdef _WIN32

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COBJMACROS
#include <d3d12.h>
#include <dxgi1_6.h>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

/* ── DX12 state ────────────────────────────────────────────────────────── */

typedef struct {
    IDXGIFactory4*          factory;
    IDXGIAdapter1*          adapter;
    ID3D12Device*           device;
    ID3D12CommandQueue*      compute_queue;
    ID3D12CommandAllocator*  cmd_alloc;
    ID3D12RootSignature*     root_sig;
    ID3D12PipelineState*     pso;

    char name[256];
    uint64_t vram;
    uint32_t node_mask;
} Dx12State;

typedef struct {
    ID3D12Resource* resource;
    void*           mapped;
    size_t          size;
} Dx12Buffer;

typedef struct {
    ID3DBlob* blob;
} Dx12Shader;

typedef struct {
    ID3D12PipelineState* pso;
    ID3D12RootSignature* root_sig;
} Dx12Pipeline;

typedef struct {
    ID3D12GraphicsCommandList* cmd_list;
    ID3D12Fence*               fence;
    HANDLE                     fence_event;
    UINT64                     fence_value;
} Dx12Context;

/* ── Device init ───────────────────────────────────────────────────────── */

static int dx12_init(GpuDevice* device) {
    Dx12State* s = calloc(1, sizeof(Dx12State));
    device->backend_data = s;

    /* Create DXGI factory */
    if (FAILED(CreateDXGIFactory1(&IID_IDXGIFactory4, (void**)&s->factory))) {
        fprintf(stderr, "dx12: CreateDXGIFactory1 failed\n");
        return GPU_ERROR_NO_DEVICE;
    }

    /* Enumerate adapters */
    IDXGIAdapter1* adapter = NULL;
    for (UINT i = 0; s->factory->lpVtbl->EnumAdapters1(s->factory, i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->lpVtbl->GetDesc1(adapter, &desc);

        /* Skip software adapters */
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

        /* Try to create device */
        if (SUCCEEDED(D3D12CreateDevice((IDXGIAdapter*)adapter, D3D_FEATURE_LEVEL_12_0,
                                        &IID_ID3D12Device, (void**)&s->device))) {
            s->adapter = adapter;
            strncpy(s->name, "DX12", sizeof(s->name) - 1);

            /* Get VRAM */
            s->vram = desc.DedicatedVideoMemory;
            s->node_mask = 1;
            break;
        }
        adapter->lpVtbl->Release(adapter);
    }

    if (!s->device) {
        fprintf(stderr, "dx12: no D3D12-capable GPU found\n");
        return GPU_ERROR_NO_DEVICE;
    }

    /* Create compute command queue */
    D3D12_COMMAND_QUEUE_DESC queue_desc = {
        .Type = D3D12_COMMAND_LIST_TYPE_COMPUTE,
        .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
    };
    s->device->lpVtbl->CreateCommandQueue(s->device, &queue_desc,
        &IID_ID3D12CommandQueue, (void**)&s->compute_queue);

    /* Create command allocator */
    s->device->lpVtbl->CreateCommandAllocator(s->device,
        D3D12_COMMAND_LIST_TYPE_COMPUTE, &IID_ID3D12CommandAllocator,
        (void**)&s->cmd_alloc);

    device->vram = s->vram;
    strncpy(device->name, s->name, sizeof(device->name) - 1);
    device->compute_units = 1;

    fprintf(stderr, "dx12: %s (%.1f MB)\n", s->name, s->vram / (1024.0 * 1024.0));

    return GPU_OK;
}

static void dx12_destroy(GpuDevice* device) {
    Dx12State* s = device->backend_data;
    if (!s) return;

    if (s->cmd_alloc) s->cmd_alloc->lpVtbl->Release(s->cmd_alloc);
    if (s->compute_queue) s->compute_queue->lpVtbl->Release(s->compute_queue);
    if (s->device) s->device->lpVtbl->Release(s->device);
    if (s->adapter) s->adapter->lpVtbl->Release(s->adapter);
    if (s->factory) s->factory->lpVtbl->Release(s->factory);
    free(s);
}

/* ── Buffers ───────────────────────────────────────────────────────────── */

static GpuBuffer* dx12_buffer_create(GpuDevice* device, size_t size, uint32_t usage) {
    Dx12State* s = device->backend_data;

    Dx12Buffer* dbuf = calloc(1, sizeof(Dx12Buffer));
    dbuf->size = size;

    D3D12_HEAP_PROPERTIES heap_props = {
        .Type = D3D12_HEAP_TYPE_UPLOAD,  /* CPU-visible */
    };
    D3D12_RESOURCE_DESC res_desc = {
        .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
        .Width = size,
        .Height = 1,
        .DepthOrArraySize = 1,
        .MipLevels = 1,
        .SampleDesc.Count = 1,
        .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
    };

    HRESULT hr = s->device->lpVtbl->CreateCommittedResource(
        s->device, &heap_props, D3D12_HEAP_FLAG_NONE,
        &res_desc, D3D12_RESOURCE_STATE_GENERIC_READ,
        NULL, &IID_ID3D12Resource, (void**)&dbuf->resource);

    if (FAILED(hr)) {
        free(dbuf);
        return NULL;
    }

    /* Map */
    D3D12_RANGE read_range = {0, 0};
    dbuf->resource->lpVtbl->Map(dbuf->resource, 0, &read_range, (void**)&dbuf->mapped);

    GpuBuffer* result = calloc(1, sizeof(GpuBuffer));
    result->device = device;
    result->gpu_data = dbuf;
    result->size = size;
    result->usage = usage;
    result->data = dbuf->mapped;
    return result;
}

static GpuResult dx12_buffer_write(GpuBuffer* buffer, const void* data, size_t size, size_t offset) {
    Dx12Buffer* dbuf = buffer->gpu_data;
    memcpy((char*)dbuf->mapped + offset, data, size);
    return GPU_OK;
}

static GpuResult dx12_buffer_read(GpuBuffer* buffer, void* data, size_t size, size_t offset) {
    Dx12Buffer* dbuf = buffer->gpu_data;
    memcpy(data, (char*)dbuf->mapped + offset, size);
    return GPU_OK;
}

static void dx12_buffer_destroy(GpuBuffer* buffer) {
    if (!buffer) return;
    Dx12Buffer* dbuf = buffer->gpu_data;
    if (dbuf) {
        if (dbuf->resource) {
            if (dbuf->mapped) {
                D3D12_RANGE range = {0, 0};
                dbuf->resource->lpVtbl->Unmap(dbuf->resource, 0, &range);
            }
            dbuf->resource->lpVtbl->Release(dbuf->resource);
        }
        free(dbuf);
    }
    free(buffer);
}

/* ── Shaders ───────────────────────────────────────────────────────────── */

static GpuShader* dx12_shader_create_spirv(GpuDevice* device, const uint32_t* code,
                                             size_t code_len, const char* entry_point) {
    (void)code; (void)code_len; (void)entry_point;
    fprintf(stderr, "dx12: SPIR-V not supported, use DXBC/HLSL\n");
    return NULL;
}

static GpuShader* dx12_shader_create_wgsl(GpuDevice* device, const char* source,
                                            size_t source_len, const char* entry_point) {
    (void)source; (void)source_len; (void)entry_point;
    fprintf(stderr, "dx12: WGSL not supported, use HLSL\n");
    return NULL;
}

static GpuShader* dx12_shader_create_dxbc(GpuDevice* device, const void* dxbc_data,
                                            size_t dxbc_size) {
    Dx12Shader* shader = calloc(1, sizeof(Dx12Shader));

    HRESULT hr = D3DCreateBlob(dxbc_size, &shader->blob);
    if (FAILED(hr)) { free(shader); return NULL; }

    void* ptr = shader->blob->lpVtbl->GetBufferPointer(shader->blob);
    memcpy(ptr, dxbc_data, dxbc_size);

    GpuShader* result = calloc(1, sizeof(GpuShader));
    result->device = device;
    result->handle = shader;
    return result;
}

static void dx12_shader_destroy(GpuShader* shader) {
    if (!shader) return;
    Dx12Shader* s = shader->handle;
    if (s) {
        if (s->blob) s->blob->lpVtbl->Release(s->blob);
        free(s);
    }
    free(shader);
}

/* ── Pipelines ─────────────────────────────────────────────────────────── */

static GpuPipeline* dx12_pipeline_create(GpuDevice* device, GpuShader* shader,
                                           const char* entry_point,
                                           GpuBindEntry* entries, uint32_t num_entries) {
    Dx12State* s = device->backend_data;
    Dx12Shader* sh = shader->handle;

    /* Create root signature from entries */
    /* Simple: one root descriptor per storage buffer */
    D3D12_ROOT_PARAMETER* params = calloc(num_entries, sizeof(D3D12_ROOT_PARAMETER));
    for (uint32_t i = 0; i < num_entries; i++) {
        params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[i].Descriptor.ShaderRegister = entries[i].binding;
        params[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }

    D3D12_ROOT_SIGNATURE_DESC rs_desc = {
        .NumParameters = num_entries,
        .pParameters = params,
        .Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };

    ID3DBlob* sig_blob = NULL;
    ID3DBlob* error_blob = NULL;
    D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                &sig_blob, &error_blob);

    ID3D12RootSignature* root_sig = NULL;
    s->device->lpVtbl->CreateRootSignature(s->device, s->node_mask,
        sig_blob->lpVtbl->GetBufferPointer(sig_blob),
        sig_blob->lpVtbl->GetBufferSize(sig_blob),
        &IID_ID3D12RootSignature, (void**)&root_sig);
    sig_blob->lpVtbl->Release(sig_blob);
    if (error_blob) error_blob->lpVtbl->Release(error_blob);
    free(params);

    /* Create PSO */
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = {
        .pRootSignature = root_sig,
        .CS = {
            .pShaderBytecode = sh->blob->lpVtbl->GetBufferPointer(sh->blob),
            .BytecodeLength = sh->blob->lpVtbl->GetBufferSize(sh->blob),
        },
    };

    ID3D12PipelineState* pso = NULL;
    s->device->lpVtbl->CreateComputePipelineState(s->device, &pso_desc,
        &IID_ID3D12PipelineState, (void**)&pso);

    Dx12Pipeline* pipe = calloc(1, sizeof(Dx12Pipeline));
    pipe->pso = pso;
    pipe->root_sig = root_sig;

    GpuPipeline* result = calloc(1, sizeof(GpuPipeline));
    result->device = device;
    result->handle = pipe;
    return result;
}

static void dx12_pipeline_destroy(GpuPipeline* pipeline) {
    if (!pipeline) return;
    Dx12Pipeline* p = pipeline->handle;
    if (p) {
        if (p->pso) p->pso->lpVtbl->Release(p->pso);
        if (p->root_sig) p->root_sig->lpVtbl->Release(p->root_sig);
        free(p);
    }
    free(pipeline);
}

/* ── Compute dispatch ──────────────────────────────────────────────────── */

static GpuContext* dx12_compute_begin(GpuDevice* device) {
    Dx12State* s = device->backend_data;

    Dx12Context* ctx = calloc(1, sizeof(Dx12Context));

    s->device->lpVtbl->CreateCommandList(s->device, s->node_mask,
        D3D12_COMMAND_LIST_TYPE_COMPUTE, s->cmd_alloc, NULL,
        &IID_ID3D12GraphicsCommandList, (void**)&ctx->cmd_list);

    /* Create fence */
    s->device->lpVtbl->CreateFence(s->device, 0, D3D12_FENCE_FLAG_NONE,
        &IID_ID3D12Fence, (void**)&ctx->fence);
    ctx->fence_event = CreateEvent(NULL, FALSE, FALSE, NULL);
    ctx->fence_value = 1;

    GpuContext* result = calloc(1, sizeof(GpuContext));
    result->device = device;
    result->handle = ctx;
    return result;
}

static void dx12_compute_bind_pipeline(GpuContext* ctx, GpuPipeline* pipeline) {
    Dx12Context* dc = ctx->handle;
    Dx12Pipeline* dp = pipeline->handle;

    dc->cmd_list->lpVtbl->SetComputeRootSignature(dc->cmd_list, dp->root_sig);
    dc->cmd_list->lpVtbl->SetPipelineState(dc->cmd_list, dp->pso);
}

static void dx12_compute_bind_buffer(GpuContext* ctx, uint32_t binding, GpuBuffer* buffer) {
    Dx12Context* dc = ctx->handle;
    Dx12Buffer* dbuf = buffer->gpu_data;

    D3D12_GPU_VIRTUAL_ADDRESS gpu_addr;
    dbuf->resource->lpVtbl->GetGPUVirtualAddress(dbuf->resource, &gpu_addr);

    dc->cmd_list->lpVtbl->SetComputeRootUnorderedAccessView(dc->cmd_list, binding, gpu_addr);
}

static void dx12_compute_set_push(GpuContext* ctx, const void* data, size_t size) {
    Dx12Context* dc = ctx->handle;
    /* DX12: upload push constants via a small upload buffer */
    /* For simplicity, use root constants (up to 128 DWORDs) */
    dc->cmd_list->lpVtbl->SetComputeRoot32BitConstants(dc->cmd_list, 0,
        (UINT)(size / 4), data, 0);
}

static void dx12_compute_dispatch(GpuContext* ctx, uint32_t x, uint32_t y, uint32_t z) {
    Dx12Context* dc = ctx->handle;
    dc->cmd_list->lpVtbl->Dispatch(dc->cmd_list, x, y, z);
}

static GpuResult dx12_compute_end(GpuContext* ctx) {
    Dx12Context* dc = ctx->handle;
    Dx12State* s = ctx->device->backend_data;

    dc->cmd_list->lpVtbl->Close(dc->cmd_list);

    /* Execute */
    ID3D12CommandList* lists[] = { (ID3D12CommandList*)dc->cmd_list };
    s->compute_queue->lpVtbl->ExecuteCommandLists(s->compute_queue, 1, lists);

    /* Signal fence */
    s->compute_queue->lpVtbl->Signal(s->compute_queue, dc->fence, dc->fence_value);

    /* Wait */
    if (dc->fence->lpVtbl->GetCompletedValue(dc->fence) < dc->fence_value) {
        dc->fence->lpVtbl->SetEventOnCompletion(dc->fence, dc->fence_value, dc->fence_event);
        WaitForSingleObject(dc->fence_event, INFINITE);
    }
    dc->fence_value++;

    /* Cleanup */
    dc->cmd_list->lpVtbl->Release(dc->cmd_list);
    dc->fence->lpVtbl->Release(dc->fence);
    CloseHandle(dc->fence_event);
    free(dc);
    free(ctx);

    return GPU_OK;
}

/* ── Backend registration ──────────────────────────────────────────────── */

const GpuBackend gpu_backend_dx12 = {
    .name = "dx12",
    .init = dx12_init,
    .destroy = dx12_destroy,
    .buffer_create = dx12_buffer_create,
    .buffer_write = (void*)dx12_buffer_write,
    .buffer_read = (void*)dx12_buffer_read,
    .buffer_destroy = dx12_buffer_destroy,
    .shader_create_wgsl = dx12_shader_create_wgsl,
    .shader_create_spirv = dx12_shader_create_spirv,
    .shader_destroy = dx12_shader_destroy,
    .pipeline_create = dx12_pipeline_create,
    .pipeline_destroy = dx12_pipeline_destroy,
    .compute_begin = dx12_compute_begin,
    .compute_bind_pipeline = dx12_compute_bind_pipeline,
    .compute_bind_buffer = dx12_compute_bind_buffer,
    .compute_set_push = dx12_compute_set_push,
    .compute_dispatch = dx12_compute_dispatch,
    .compute_end = (void*)dx12_compute_end,
};

#endif /* _WIN32 */
