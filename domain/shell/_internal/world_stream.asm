; world_stream.asm — live world frame stream, as a VM assembly app.
;
; The loop lives in machine: step-sim -> render -> encode_png -> stream
; write, __FRAMES__ frames.  Python only builds the drivers (kernel +
; drivers + socket), registers them on the device bus, and attaches the
; frame consumer to StreamDevice.
;
; Slots cross devices through the shared TensorArena: `world render`
; puts the frame ndarray in the arena and returns its slot id, `cycles
; encode_png` resolves that slot — the register file never holds an
; ndarray (bus-value discipline, card 1c64882c).
;
; Registers: R0 = frame counter (LOOP), R6/R7/R8 = world/cycles/stream
; handles, R1-R5 = working values for one frame pass.

        DEV_OPEN  R6, "world"
        DEV_OPEN  R7, "cycles"
        DEV_OPEN  R8, "stream"
        MOV       R0, __FRAMES__

FRAME:  DEV_CALL  R1, R6, "step"              ; R1 = tick
        DEV_CALL  R2, R6, "render"            ; R2 = arena slot id
        DEV_CALL  R3, R7, "encode_png", R2    ; R3 = PNG bytes
        DEV_CALL  R4, R8, "write", R3, R1     ; stream write, tag = tick
        DEV_CALL  R5, R7, "free_slot", R2     ; slot back to the arena
        LOOP      R0, FRAME
        HALT
