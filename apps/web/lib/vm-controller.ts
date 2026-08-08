/**
 * VM Controller — x86 assembly execution API.
 */

import { apiPost, apiGet } from './http-client'

export interface VMRegister {
  name: string
  value: number
  hex: string
}

export interface VMRunResult {
  success: boolean
  exit_code: number
  steps_executed: number
  elapsed_ms: number
  output: string
  registers: VMRegister[]
  eip: number
  eip_hex: string
  status: string
  error?: string
  trace?: { step: number; eip: string; opcode: string; operands: string }[]
  vga_text?: string
  vga_cells?: { ch: string; fg: string; bg: string }[]
  keyboard_buffer?: string
  memory_dump?: string
}

export interface VMBuiltin {
  name: string
  description: string
}

export interface VMInfo {
  isa: string
  max_steps: number
  default_memory: number
  max_memory: number
  registers: string[]
  features: string[]
}

class VMController {
  async run(
    source: string,
    opts?: { maxSteps?: number; role?: string; debug?: boolean; keyboardInput?: string },
  ): Promise<VMRunResult> {
    return apiPost<VMRunResult>('/vm/run', {
      source,
      max_steps: opts?.maxSteps ?? 5000,
      role: opts?.role ?? 'user',
      debug: opts?.debug ?? false,
      keyboard_input: opts?.keyboardInput,
    })
  }

  async builtins(): Promise<{ programs: VMBuiltin[] }> {
    return apiGet<{ programs: VMBuiltin[] }>('/vm/builtins')
  }

  async info(): Promise<VMInfo> {
    return apiGet<VMInfo>('/vm/info')
  }
}

export const vmController = new VMController()
