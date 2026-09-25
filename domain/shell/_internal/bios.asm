; AI Compteur Custom BIOS
; Minimal 64KB BIOS — replaces SeaBIOS
; Provides: INT 10h (teletype), INT 13h (disk read), boot to MBR

[BITS 16]
[ORG 0x0000]

; ── Reset Vector (offset 0xFFF0) ──────────────────────────────────────────
; This is where the CPU starts after reset
; ROM is at 0xF0000-0xFFFFF, so offset 0xFFF0 = physical 0xFFFFFFF0

times 0xFFF0-($-$$) db 0
    jmp 0xF000:0x0000      ; far jump to BIOS entry point

; ── BIOS Entry Point (offset 0x0000) ──────────────────────────────────────

bios_entry:
    ; Set up segments
    ; Note: can't mov seg regs directly, use push/pop trick
    mov ax, 0xF000
    mov ds, ax

    ; Set up stack at 0x0000:0x0400 (below IVT)
    xor ax, ax
    mov ss, ax
    mov sp, 0x0400

    ; Clear direction flag
    cld

    ; Set video mode 3 (80x25 text)
    mov ax, 0x0003
    int 0x10

    ; Print banner to VGA
    call print_banner

    ; Load MBR from floppy (LBA 0) to 0x7C00
    call load_mbr

    ; Jump to bootloader
    jmp 0x0000:0x7C00

; ── Print Banner ───────────────────────────────────────────────────────────
; Writes "AI Compteur BIOS v0.1" directly to VGA text buffer

print_banner:
    ; VGA text buffer at physical 0xB8000
    ; In real mode, we can address it as 0xB800:0x0000
    ; But we can't set ES directly, so use stack trick
    push 0xB800
    pop es

    ; DI = row 0, col 0
    xor di, di

    ; SI = message
    mov si, msg_banner

.print_loop:
    lodsb
    or al, al
    jz .print_done
    ; Store char + attribute (0x07 = light gray on black)
    mov ah, 0x07
    stosw
    jmp .print_loop

.print_done:
    ret

; ── Load MBR ───────────────────────────────────────────────────────────────
; Reads sector 0 from floppy to 0x7C00

load_mbr:
    ; Reset floppy controller
    xor ax, ax
    mov dl, 0x00           ; drive 0
    int 0x13

    ; Read sector 0 (LBA 0)
    mov ah, 0x02           ; BIOS disk read
    mov al, 1              ; 1 sector
    mov ch, 0              ; cylinder 0
    mov cl, 1              ; sector 1 (1-based LBA 0)
    mov dh, 0              ; head 0
    mov dl, 0x00           ; drive 0
    mov bx, 0x7C00         ; destination
    int 0x13
    jc .disk_error
    ret

.disk_error:
    ; Print error and halt
    mov si, msg_disk_err
    call print_banner      ; reuse VGA print (es already set)
    hlt
    jmp $

; ── INT 10h Handler (Video) ────────────────────────────────────────────────

int10_handler:
    pusha
    cmp ah, 0x0E
    je .teletype
    cmp ah, 0x00
    je .set_mode
    popa
    iret

.teletype:
    ; AH=0Eh: Teletype output
    ; AL = character, BH = page
    push 0xB800
    pop es
    ; Simple: just write to VGA at cursor position
    ; For now, write to a fixed position (row 1)
    ; A real BIOS would track cursor position
    mov di, 0x00A0         ; row 1
    mov ah, 0x07
    stosw
    popa
    iret

.set_mode:
    ; AH=00h: Set video mode
    ; AL = mode
    ; For now, just return
    popa
    iret

; ── INT 13h Handler (Disk) ─────────────────────────────────────────────────

int13_handler:
    pusha
    cmp ah, 0x02
    je .read_sectors
    cmp ah, 0x00
    je .reset_disk
    popa
    clc                     ; clear carry (success)
    iret

.read_sectors:
    ; AH=02h: Read sectors
    ; AL=count, CH=cl cylinder, CL=sector, DH=head, DL=drive, ES:BX=buffer
    ; For simplicity, just read using BIOS real mode disk services
    ; This handler would need to actually program the floppy controller
    ; For now, return success
    popa
    clc
    iret

.reset_disk:
    ; AH=00h: Reset disk
    popa
    clc
    iret

; ── Data ───────────────────────────────────────────────────────────────────

msg_banner: db "AI Compteur BIOS v0.1", 0
msg_disk_err: db "Disk error", 0

; ── Padding + BIOS Info ────────────────────────────────────────────────────

; Pad to 64KB
times 0xFFF0-($-$$) db 0
