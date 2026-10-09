/*
 * vulkan.c — Vulkan compute backend for gpu_engine.
 *
 * Pure Vulkan, no third-party libs. Handles:
 *   - Instance/device/queue creation
 *   - Buffer allocation (host-visible + device-local)
 *   - SPIR-V shader loading
 *   - Compute pipeline + descriptor sets
 *   - Command buffer recording and submission
 *   - Fence-based synchronization
 *
 * WGSL → SPIR-V compilation is handled at build time or by a
 * separate tool (not runtime). This backend accepts SPIR-V only.
 */

#include "engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Vulkan headers — system SDK */
#ifdef _WIN32
  #define VK_USE_PLATFORM_WIN32_KHR
  #include <vulkan/vulkan.h>
  #include <vulkan/vulkan_win32.h>
#elif defined(__APPLE__)
  #define VK_USE_PLATFORM_MACOS_KHR
  #include <vulkan/vulkan_metal.h>
  /* MoltenVK provides Vulkan on macOS/iOS */
#else
  #include <vulkan/vulkan.h>
#endif

/* ── Vulkan state ──────────────────────────────────────────────────────── */

typedef struct {
    VkInstance       instance;
    VkPhysicalDevice phys_device;
    VkDevice         device;
    VkQueue          compute_queue;
    uint32_t         compute_queue_family;
    VkCommandPool    cmd_pool;
    VkDescriptorPool desc_pool;

    /* Properties */
    VkPhysicalDeviceMemoryProperties mem_props;
    VkPhysicalDeviceProperties       dev_props;
    uint64_t                         vram;

    /* Limits */
    uint32_t max_compute_workgroup_count[3];
    uint32_t max_compute_workgroup_size[3];
    uint32_t max_compute_workgroup_invocations;
    uint32_t max_storage_buffer_size;
    uint32_t min_uniform_buffer_offset_alignment;
} VulkanState;

typedef struct {
    VkBuffer       buffer;
    VkDeviceMemory memory;
    VkDeviceSize   size;
    void*          mapped;
    uint32_t       memory_type;
} VulkanBuffer;

typedef struct {
    VkShaderModule module;
} VulkanShader;

typedef struct {
    VkPipeline          pipeline;
    VkPipelineLayout    layout;
    VkDescriptorSetLayout desc_layout;
    VkDescriptorSet     desc_set;
} VulkanPipeline;

typedef struct {
    VkCommandBuffer cmd;
    VkFence         fence;
    uint32_t        num_bindings;
    VkDescriptorSet desc_set;
} VulkanContext;

/* ── Helpers ───────────────────────────────────────────────────────────── */

static uint32_t find_memory_type(VulkanState* s, uint32_t type_bits, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < s->mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1 << i)) &&
            (s->mem_props.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return UINT32_MAX;
}

static VkResult check(VkResult r, const char* msg) {
    if (r != VK_SUCCESS) {
        fprintf(stderr, "vulkan: %s failed: %d\n", msg, r);
    }
    return r;
}

/* ── Device init ───────────────────────────────────────────────────────── */

static int vulkan_init(GpuDevice* device) {
    VulkanState* s = calloc(1, sizeof(VulkanState));
    if (!s) return GPU_ERROR_NO_MEMORY;
    device->backend_data = s;

    /* Instance */
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "sloughGPT gpu_engine",
        .apiVersion = VK_API_VERSION_1_2,
    };
    VkInstanceCreateInfo inst_ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
    };
    if (check(vkCreateInstance(&inst_ci, NULL, &s->instance), "vkCreateInstance") != VK_SUCCESS) {
        return GPU_ERROR_NO_DEVICE;
    }

    /* Physical device */
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(s->instance, &count, NULL);
    if (count == 0) { fprintf(stderr, "vulkan: no GPU found\n"); return GPU_ERROR_NO_DEVICE; }

    VkPhysicalDevice* devs = malloc(count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(s->instance, &count, devs);

    /* Pick first device with compute queue */
    s->phys_device = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qcount, NULL);
        VkQueueFamilyProperties* qprops = malloc(qcount * sizeof(VkQueueFamilyProperties));
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qcount, qprops);
        for (uint32_t q = 0; q < qcount; q++) {
            if (qprops[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                s->phys_device = devs[i];
                s->compute_queue_family = q;
                break;
            }
        }
        free(qprops);
        if (s->phys_device != VK_NULL_HANDLE) break;
    }
    free(devs);

    if (s->phys_device == VK_NULL_HANDLE) {
        fprintf(stderr, "vulkan: no compute-capable GPU\n");
        return GPU_ERROR_NO_DEVICE;
    }

    /* Device properties */
    vkGetPhysicalDeviceProperties(s->phys_device, &s->dev_props);
    vkGetPhysicalDeviceMemoryProperties(s->phys_device, &s->mem_props);

    /* Find memory heap with DEVICE_LOCAL */
    for (uint32_t i = 0; i < s->mem_props.memoryHeapCount; i++) {
        if (s->mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            s->vram = s->mem_props.memoryHeaps[i].size;
            break;
        }
    }

    /* Logical device */
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = s->compute_queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };
    VkDeviceCreateInfo dev_ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_ci,
    };
    if (check(vkCreateDevice(s->phys_device, &dev_ci, NULL, &s->device), "vkCreateDevice") != VK_SUCCESS) {
        return GPU_ERROR_NO_DEVICE;
    }
    vkGetDeviceQueue(s->device, s->compute_queue_family, 0, &s->compute_queue);

    /* Command pool */
    VkCommandPoolCreateInfo pool_ci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = s->compute_queue_family,
    };
    check(vkCreateCommandPool(s->device, &pool_ci, NULL, &s->cmd_pool), "vkCreateCommandPool");

    /* Descriptor pool */
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16 },
    };
    VkDescriptorPoolCreateInfo desc_pool_ci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 64,
        .poolSizeCount = 2,
        .pPoolSizes = pool_sizes,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
    };
    check(vkCreateDescriptorPool(s->device, &desc_pool_ci, NULL, &s->desc_pool), "vkCreateDescriptorPool");

    /* Store limits */
    s->max_compute_workgroup_count[0] = s->dev_props.limits.maxComputeWorkGroupCount[0];
    s->max_compute_workgroup_count[1] = s->dev_props.limits.maxComputeWorkGroupCount[1];
    s->max_compute_workgroup_count[2] = s->dev_props.limits.maxComputeWorkGroupCount[2];
    s->max_compute_workgroup_size[0] = s->dev_props.limits.maxComputeWorkGroupSize[0];
    s->max_compute_workgroup_size[1] = s->dev_props.limits.maxComputeWorkGroupSize[1];
    s->max_compute_workgroup_size[2] = s->dev_props.limits.maxComputeWorkGroupSize[2];
    s->max_compute_workgroup_invocations = s->dev_props.limits.maxComputeWorkGroupInvocations;
    s->max_storage_buffer_size = s->dev_props.limits.maxStorageBufferRange;
    s->min_uniform_buffer_offset_alignment = s->dev_props.limits.minUniformBufferOffsetAlignment;

    /* Device name */
    strncpy(device->name, s->dev_props.deviceName, sizeof(device->name) - 1);
    device->vram = s->vram;
    device->compute_units = s->dev_props.limits.maxComputeWorkGroupInvocations;

    return GPU_OK;
}

static void vulkan_destroy(GpuDevice* device) {
    VulkanState* s = device->backend_data;
    if (!s) return;

    vkDeviceWaitIdle(s->device);
    vkDestroyDescriptorPool(s->device, s->desc_pool, NULL);
    vkDestroyCommandPool(s->device, s->cmd_pool, NULL);
    vkDestroyDevice(s->device, NULL);
    vkDestroyInstance(s->instance, NULL);
    free(s);
}

/* ── Buffers ───────────────────────────────────────────────────────────── */

static GpuBuffer* vulkan_buffer_create(GpuDevice* device, size_t size, uint32_t usage) {
    VulkanState* s = device->backend_data;

    VulkanBuffer* buf = calloc(1, sizeof(VulkanBuffer));
    if (!buf) return NULL;
    buf->size = size;

    /* Determine Vulkan usage flags */
    VkBufferUsageFlags vk_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (usage & GPU_BUF_UNIFORM) {
        vk_usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }

    /* Create buffer */
    VkBufferCreateInfo buf_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = vk_usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (check(vkCreateBuffer(s->device, &buf_ci, NULL, &buf->buffer), "vkCreateBuffer") != VK_SUCCESS) {
        free(buf);
        return NULL;
    }

    /* Query memory requirements */
    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(s->device, buf->buffer, &mem_req);

    /* Prefer host-visible for easy read/write */
    uint32_t mem_type = find_memory_type(s, mem_req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    /* If not available, use device-local (requires staging for upload) */
    if (mem_type == UINT32_MAX) {
        mem_type = find_memory_type(s, mem_req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        buf->memory_type = 1;  /* device-local */
    } else {
        buf->memory_type = 0;  /* host-visible */
    }

    if (mem_type == UINT32_MAX) {
        fprintf(stderr, "vulkan: no suitable memory type\n");
        vkDestroyBuffer(s->device, buf->buffer, NULL);
        free(buf);
        return NULL;
    }

    /* Allocate memory */
    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_req.size,
        .memoryTypeIndex = mem_type,
    };
    if (check(vkAllocateMemory(s->device, &alloc_info, NULL, &buf->memory), "vkAllocateMemory") != VK_SUCCESS) {
        vkDestroyBuffer(s->device, buf->buffer, NULL);
        free(buf);
        return NULL;
    }

    vkBindBufferMemory(s->device, buf->buffer, buf->memory, 0);

    /* Map if host-visible */
    if (buf->memory_type == 0) {
        vkMapMemory(s->device, buf->memory, 0, size, 0, &buf->mapped);
    }

    GpuBuffer* result = calloc(1, sizeof(GpuBuffer));
    result->device = device;
    result->gpu_data = buf;
    result->size = size;
    result->usage = usage;
    return result;
}

static VkResult vulkan_buffer_write(GpuBuffer* buffer, const void* data, size_t size, size_t offset) {
    VulkanBuffer* buf = buffer->gpu_data;
    if (buf->mapped) {
        memcpy((char*)buf->mapped + offset, data, size);
        return VK_SUCCESS;
    }
    /* Host-invisible: staging buffer upload for device-local memory */
    VulkanState* s = buffer->device->backend_data;

    /* Create host-visible staging buffer */
    VkBufferCreateInfo staging_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer staging_buf;
    if (check(vkCreateBuffer(s->device, &staging_ci, NULL, &staging_buf), "staging create") != VK_SUCCESS)
        return VK_ERROR_OUT_OF_HOST_MEMORY;

    VkMemoryRequirements staging_req;
    vkGetBufferMemoryRequirements(s->device, staging_buf, &staging_req);
    uint32_t staging_mem_type = find_memory_type(s, staging_req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (staging_mem_type == UINT32_MAX) {
        vkDestroyBuffer(s->device, staging_buf, NULL);
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    VkMemoryAllocateInfo staging_alloc = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = staging_req.size,
        .memoryTypeIndex = staging_mem_type,
    };
    VkDeviceMemory staging_mem;
    if (check(vkAllocateMemory(s->device, &staging_alloc, NULL, &staging_mem), "staging alloc") != VK_SUCCESS) {
        vkDestroyBuffer(s->device, staging_buf, NULL);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    vkBindBufferMemory(s->device, staging_buf, staging_mem, 0);

    /* Map, copy data, unmap */
    void* mapped;
    vkMapMemory(s->device, staging_mem, 0, size, 0, &mapped);
    memcpy(mapped, data, size);
    vkUnmapMemory(s->device, staging_mem);

    /* Record and submit copy command */
    VkCommandBufferAllocateInfo cb_ai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = s->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(s->device, &cb_ai, &cmd);

    VkCommandBufferBeginInfo begin_ci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &begin_ci);

    VkBufferCopy copy_region = { .srcOffset = 0, .dstOffset = offset, .size = size };
    vkCmdCopyBuffer(cmd, staging_buf, buf->buffer, 1, &copy_region);

    vkEndCommandBuffer(cmd);

    VkFenceCreateInfo fence_ci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    vkCreateFence(s->device, &fence_ci, NULL, &fence);

    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    vkQueueSubmit(s->compute_queue, 1, &submit, fence);
    vkWaitForFences(s->device, 1, &fence, VK_TRUE, UINT64_MAX);

    /* Cleanup */
    vkDestroyFence(s->device, fence, NULL);
    vkFreeCommandBuffers(s->device, s->cmd_pool, 1, &cmd);
    vkFreeMemory(s->device, staging_mem, NULL);
    vkDestroyBuffer(s->device, staging_buf, NULL);

    return VK_SUCCESS;
}

static VkResult vulkan_buffer_read(GpuBuffer* buffer, void* data, size_t size, size_t offset) {
    VulkanBuffer* buf = buffer->gpu_data;
    if (buf->mapped) {
        memcpy(data, (char*)buf->mapped + offset, size);
        return VK_SUCCESS;
    }
    /* Host-invisible: staging buffer readback for device-local memory */
    VulkanState* s = buffer->device->backend_data;

    /* Create host-visible staging buffer */
    VkBufferCreateInfo staging_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer staging_buf;
    if (check(vkCreateBuffer(s->device, &staging_ci, NULL, &staging_buf), "staging create read") != VK_SUCCESS)
        return VK_ERROR_OUT_OF_HOST_MEMORY;

    VkMemoryRequirements staging_req;
    vkGetBufferMemoryRequirements(s->device, staging_buf, &staging_req);
    uint32_t staging_mem_type = find_memory_type(s, staging_req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (staging_mem_type == UINT32_MAX) {
        vkDestroyBuffer(s->device, staging_buf, NULL);
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    VkMemoryAllocateInfo staging_alloc = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = staging_req.size,
        .memoryTypeIndex = staging_mem_type,
    };
    VkDeviceMemory staging_mem;
    if (check(vkAllocateMemory(s->device, &staging_alloc, NULL, &staging_mem), "staging alloc read") != VK_SUCCESS) {
        vkDestroyBuffer(s->device, staging_buf, NULL);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    vkBindBufferMemory(s->device, staging_buf, staging_mem, 0);

    /* Record and submit copy command: device-local → staging */
    VkCommandBufferAllocateInfo cb_ai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = s->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(s->device, &cb_ai, &cmd);

    VkCommandBufferBeginInfo begin_ci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &begin_ci);

    VkBufferCopy copy_region = { .srcOffset = offset, .dstOffset = 0, .size = size };
    vkCmdCopyBuffer(cmd, buf->buffer, staging_buf, 1, &copy_region);

    vkEndCommandBuffer(cmd);

    VkFenceCreateInfo fence_ci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    vkCreateFence(s->device, &fence_ci, NULL, &fence);

    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    vkQueueSubmit(s->compute_queue, 1, &submit, fence);
    vkWaitForFences(s->device, 1, &fence, VK_TRUE, UINT64_MAX);

    /* Read from staging */
    void* mapped;
    vkMapMemory(s->device, staging_mem, 0, size, 0, &mapped);
    memcpy(data, mapped, size);
    vkUnmapMemory(s->device, staging_mem);

    /* Cleanup */
    vkDestroyFence(s->device, fence, NULL);
    vkFreeCommandBuffers(s->device, s->cmd_pool, 1, &cmd);
    vkFreeMemory(s->device, staging_mem, NULL);
    vkDestroyBuffer(s->device, staging_buf, NULL);

    return VK_SUCCESS;
}

static void vulkan_buffer_destroy(GpuBuffer* buffer) {
    if (!buffer) return;
    VulkanBuffer* buf = buffer->gpu_data;
    if (buf) {
        VulkanState* s = buffer->device->backend_data;
        if (buf->mapped) vkUnmapMemory(s->device, buf->memory);
        vkFreeMemory(s->device, buf->memory, NULL);
        vkDestroyBuffer(s->device, buf->buffer, NULL);
        free(buf);
    }
    free(buffer);
}

/* ── Shaders ───────────────────────────────────────────────────────────── */

static GpuShader* vulkan_shader_create_spirv(GpuDevice* device, const uint32_t* code,
                                               size_t code_len, const char* entry_point) {
    (void)entry_point;
    VulkanState* s = device->backend_data;

    VkShaderModuleCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code_len,
        .pCode = code,
    };

    VulkanShader* shader = calloc(1, sizeof(VulkanShader));
    if (check(vkCreateShaderModule(s->device, &ci, NULL, &shader->module), "vkCreateShaderModule") != VK_SUCCESS) {
        free(shader);
        return NULL;
    }

    GpuShader* result = calloc(1, sizeof(GpuShader));
    result->device = device;
    result->handle = shader;
    return result;
}

static GpuShader* vulkan_shader_create_wgsl(GpuDevice* device, const char* source,
                                              size_t source_len, const char* entry_point) {
    /* WGSL not directly supported by Vulkan — must be compiled to SPIR-V first */
    (void)source; (void)source_len; (void)entry_point;
    fprintf(stderr, "vulkan: WGSL not supported, compile to SPIR-V first\n");
    return NULL;
}

static void vulkan_shader_destroy(GpuShader* shader) {
    if (!shader) return;
    VulkanShader* s = shader->handle;
    if (s) {
        VulkanState* vs = shader->device->backend_data;
        vkDestroyShaderModule(vs->device, s->module, NULL);
        free(s);
    }
    free(shader);
}

/* ── Pipelines ─────────────────────────────────────────────────────────── */

static GpuPipeline* vulkan_pipeline_create(GpuDevice* device, GpuShader* shader,
                                            const char* entry_point,
                                            GpuBindEntry* entries, uint32_t num_entries) {
    VulkanState* s = device->backend_data;
    VulkanShader* sh = shader->handle;

    VulkanPipeline* pipe = calloc(1, sizeof(VulkanPipeline));

    /* Descriptor set layout */
    VkDescriptorSetLayoutBinding* bindings = calloc(num_entries, sizeof(VkDescriptorSetLayoutBinding));
    for (uint32_t i = 0; i < num_entries; i++) {
        bindings[i].binding = entries[i].binding;
        bindings[i].descriptorType = entries[i].type == 0 ?
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dsl_ci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = num_entries,
        .pBindings = bindings,
    };
    check(vkCreateDescriptorSetLayout(s->device, &dsl_ci, NULL, &pipe->desc_layout), "vkCreateDescriptorSetLayout");
    free(bindings);

    /* Pipeline layout */
    VkPipelineLayoutCreateInfo pl_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &pipe->desc_layout,
    };
    check(vkCreatePipelineLayout(s->device, &pl_ci, NULL, &pipe->layout), "vkCreatePipelineLayout");

    /* Compute pipeline */
    VkComputePipelineCreateInfo cp_ci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = sh->module,
            .pName = entry_point,
        },
        .layout = pipe->layout,
    };
    check(vkCreateComputePipelines(s->device, VK_NULL_HANDLE, 1, &cp_ci, NULL, &pipe->pipeline), "vkCreateComputePipelines");

    /* Allocate descriptor set */
    VkDescriptorSetAllocateInfo ds_ai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = s->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &pipe->desc_layout,
    };
    check(vkAllocateDescriptorSets(s->device, &ds_ai, &pipe->desc_set), "vkAllocateDescriptorSets");

    GpuPipeline* result = calloc(1, sizeof(GpuPipeline));
    result->device = device;
    result->handle = pipe;
    return result;
}

static void vulkan_pipeline_destroy(GpuPipeline* pipeline) {
    if (!pipeline) return;
    VulkanPipeline* p = pipeline->handle;
    if (p) {
        VulkanState* s = pipeline->device->backend_data;
        vkDestroyPipeline(s->device, p->pipeline, NULL);
        vkDestroyPipelineLayout(s->device, p->layout, NULL);
        vkDestroyDescriptorSetLayout(s->device, p->desc_layout, NULL);
        free(p);
    }
    free(pipeline);
}

/* ── Compute dispatch ──────────────────────────────────────────────────── */

static GpuContext* vulkan_compute_begin(GpuDevice* device) {
    VulkanState* s = device->backend_data;

    VulkanContext* ctx = calloc(1, sizeof(VulkanContext));

    /* Allocate command buffer */
    VkCommandBufferAllocateInfo cb_ai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = s->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    vkAllocateCommandBuffers(s->device, &cb_ai, &ctx->cmd);

    /* Create fence for synchronization */
    VkFenceCreateInfo fence_ci = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    };
    vkCreateFence(s->device, &fence_ci, NULL, &ctx->fence);

    /* Begin command buffer */
    VkCommandBufferBeginInfo begin_ci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(ctx->cmd, &begin_ci);

    GpuContext* result = calloc(1, sizeof(GpuContext));
    result->device = device;
    result->handle = ctx;
    return result;
}

static void vulkan_compute_bind_pipeline(GpuContext* ctx, GpuPipeline* pipeline) {
    VulkanContext* vc = ctx->handle;
    VulkanPipeline* vp = pipeline->handle;
    VulkanState* s = ctx->device->backend_data;

    vc->desc_set = vp->desc_set;
    vkCmdBindPipeline(vc->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vp->pipeline);
}

static void vulkan_compute_bind_buffer(GpuContext* ctx, uint32_t binding, GpuBuffer* buffer) {
    VulkanContext* vc = ctx->handle;
    VulkanState* s = ctx->device->backend_data;
    VulkanBuffer* buf = buffer->gpu_data;

    /* Update descriptor set */
    VkDescriptorBufferInfo buf_info = {
        .buffer = buf->buffer,
        .offset = 0,
        .range = buf->size,
    };
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = vc->desc_set,
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_info,
    };
    vkUpdateDescriptorSets(s->device, 1, &write, 0, NULL);
}

static void vulkan_compute_set_push(GpuContext* ctx, const void* data, size_t size) {
    VulkanContext* vc = ctx->handle;
    vkCmdPushConstants(vc->cmd, /* layout */ VK_NULL_HANDLE,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, size, data);
}

static void vulkan_compute_dispatch(GpuContext* ctx, uint32_t x, uint32_t y, uint32_t z) {
    VulkanContext* vc = ctx->handle;
    vkCmdDispatch(vc->cmd, x, y, z);
}

static GpuResult vulkan_compute_end(GpuContext* ctx) {
    VulkanContext* vc = ctx->handle;
    VulkanState* s = ctx->device->backend_data;

    vkEndCommandBuffer(vc->cmd);

    /* Submit */
    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &vc->cmd,
    };
    vkQueueSubmit(s->compute_queue, 1, &submit, vc->fence);

    /* Wait */
    vkWaitForFences(s->device, 1, &vc->fence, VK_TRUE, UINT64_MAX);

    /* Cleanup */
    vkDestroyFence(s->device, vc->fence, NULL);
    vkFreeCommandBuffers(s->device, s->cmd_pool, 1, &vc->cmd);
    free(vc);
    free(ctx);

    return GPU_OK;
}

/* ── Backend registration ──────────────────────────────────────────────── */

const GpuBackend gpu_backend_vulkan = {
    .name = "vulkan",
    .init = vulkan_init,
    .destroy = vulkan_destroy,
    .buffer_create = vulkan_buffer_create,
    .buffer_write = (void*)vulkan_buffer_write,
    .buffer_read = (void*)vulkan_buffer_read,
    .buffer_destroy = vulkan_buffer_destroy,
    .shader_create_wgsl = vulkan_shader_create_wgsl,
    .shader_create_spirv = vulkan_shader_create_spirv,
    .shader_destroy = vulkan_shader_destroy,
    .pipeline_create = vulkan_pipeline_create,
    .pipeline_destroy = vulkan_pipeline_destroy,
    .compute_begin = vulkan_compute_begin,
    .compute_bind_pipeline = vulkan_compute_bind_pipeline,
    .compute_bind_buffer = vulkan_compute_bind_buffer,
    .compute_set_push = vulkan_compute_set_push,
    .compute_dispatch = vulkan_compute_dispatch,
    .compute_end = (void*)vulkan_compute_end,
};
