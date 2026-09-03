# VM Devices Specification

## Architecture Overview

The VM device framework implements a hardware-inspired architecture with two layers:

```
Assembly (DEV_CALL) → DeviceTable (request handler) → DeviceDriver (hardware gates)
```

**DeviceTable** manages file descriptors and routes ioctls to devices.
**DeviceDriver** is the hardware abstraction — only ioctl, no open/close.

## Baseboard

### DeviceTable

Bit-based file descriptor management.

```python
class DeviceTable:
    _fd_bitmap: int          # 64 fds in one integer
    _fd_device: list[...]    # fd → device (direct index)
    _fd_mode: list[int]      # fd → mode bits
    _fd_offset: list[int]    # fd → offset
    _devices: dict[str, ...] # name → device
```

**Operations:**
- `register(device, type)` → add device to table
- `open(name, mode)` → alloc fd, return int
- `close(fd)` → free fd
- `ioctl(fd, command, *args)` → route to device

**Bit operations:**
- `_alloc_fd()` → bit scan for free fd
- `_free_fd(fd)` → bit clear
- `_fd_is_open(fd)` → bit test

### DeviceDriver

Hardware gates — only ioctl.

```python
class DeviceDriver:
    _name: str
    _device_type: int
    _state: DeviceState

    def ioctl(self, command, *args) -> SyscallResult
    def info(self) -> dict
```

No open/close — that's DeviceTable's job.

## Device Types

Bit flags for device categories:

```python
INFERENCE = 1 << 0  # 0b000001
TRAINING  = 1 << 1  # 0b000010
STORAGE   = 1 << 2  # 0b000100
NETWORK   = 1 << 3  # 0b001000
DISPLAY   = 1 << 4  # 0b010000
INPUT     = 1 << 5  # 0b100000
CUSTOM    = 0       # no type
```

## TensorDevice

Standalone compute hardware — wraps numpy.

**59 commands via ioctl:**
- Linear algebra: MATMUL, DOT, INV, SVD, EIG
- Activations: RELU, SIGMOID, TANH, SOFTMAX, GELU, SILU, ELU, SELU
- Arithmetic: ADD, SUB, MUL, DIV, NEG, ABS, POW, SQRT, EXP, LOG
- Reduction: SUM, MEAN, STD, VAR, MAX, MIN, ARGMAX, ARGMIN
- Shape: RESHAPE, TRANSPOSE, FLATTEN, SQUEEZE, UNSQUEEZE, CAT, STACK
- Convolution: CONV1D, CONV2D
- Pooling: MAX_POOL1D, MAX_POOL2D, AVG_POOL1D, AVG_POOL2D
- Normalization: BATCH_NORM, LAYER_NORM, RMS_NORM
- Attention: ATTENTION
- Loss: CROSS_ENTROPY, MSE, MAE
- Optimizers: SGD_STEP, ADAM_STEP
- Utility: CLIP_GRAD_NORM, DROPOUT, EMBEDDING, LINEAR

**Interface:**
```python
tensor = TensorDevice()
result = tensor.ioctl("MATMUL", a, b)  # → SyscallResult
```

## NPUDevice

Standalone neural processing hardware — uses TensorDevice.

**13 commands via ioctl:**
- INFO, LIST_COMMANDS
- LOAD, UNLOAD, CALL
- BATCH, PIPELINE, PROFILE, QUANTIZE
- CHECKPOINT_SAVE, CHECKPOINT_LOAD
- MEMORY, COMPUTE

**Interface:**
```python
npu = NPUDevice()
result = npu.ioctl("LOAD", "/path/to/model.slnc")
result = npu.ioctl("CALL", "model", "Hello")
result = npu.ioctl("COMPUTE", "MATMUL", a, b)  # → TensorDevice
```

## SyscallResult

Consistent response from all ioctls:

```python
@dataclass
class SyscallResult:
    success: bool
    value: Any = None
    error: str | None = None
    errno: int = 0
    elapsed_ms: float = 0.0
```

## Flow

```
1. Assembly: DEV_OPEN "npu"
   → DeviceTable.open("npu") → fd=1

2. Assembly: DEV_CALL fd=1, "LOAD", path
   → DeviceTable.ioctl(1, "LOAD", path)
   → DeviceTable lookup fd=1 → npu_device
   → npu_device.ioctl("LOAD", path) → SyscallResult

3. Assembly: DEV_CALL fd=1, "COMPUTE", "MATMUL", a, b
   → DeviceTable.ioctl(1, "COMPUTE", "MATMUL", a, b)
   → npu_device.ioctl("COMPUTE", "MATMUL", a, b)
   → tensor_device.ioctl("MATMUL", a, b) → SyscallResult
```

## Usage

```python
from domains.shell.kernel_devices import DeviceTable, DeviceType
from domains.shell.tensor_device import TensorDevice
from domains.shell.npu_device import NPUDevice

# Create table
table = DeviceTable()

# Register devices
table.register(TensorDevice(), DeviceType.INFERENCE)
table.register(NPUDevice(), DeviceType.INFERENCE)

# Open and use
fd = table.open("npu")
result = table.ioctl(fd, "LOAD", "/path/to/model.slnc")
result = table.ioctl(fd, "CALL", "model", "Hello")
table.close(fd)
```
