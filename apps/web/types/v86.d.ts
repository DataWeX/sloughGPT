/**
 * Minimal type for the v86 x86 PC emulator. The package ships no type
 * definitions; only the surface used by lib/v86-controller.ts is declared.
 */
declare module "v86" {
  export interface V86Options {
    screen_container: HTMLElement
    bios: { url: string }
    vga_bios?: { url: string }
    hda?: { url: string; async?: boolean; size?: number }
    memory_size?: number
    vga_memory_size?: number
    autostart?: boolean
    fastboot?: boolean
    wasm_path?: string
  }

  export interface V86Emulator {
    add_listener(event: string, listener: () => void): void
    save_state(): Promise<ArrayBuffer>
    restore_state(state: ArrayBuffer): Promise<void>
    restart(): void
    is_running(): boolean
    destroy?(): void
  }

  export type V86Constructor = new (options: V86Options) => V86Emulator

  export const V86: V86Constructor

  /**
   * CJS interop default: a namespace object exposing `{ V86 }` (the original
   * controller chain reads `.V86` off it and falls through when absent).
   */
  const _interop: { V86?: V86Constructor }
  export default _interop
}
