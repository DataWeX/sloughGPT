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
  training_job_id?: number | null
  training_result?: string | null
}

export interface VMTrainingJob {
  job_id: number
  api_job_id: string
  status: string
  progress: number
  error?: string | null
  result?: string | null
}

export interface VMBuiltin {
  name: string
  description: string
  code: string
}

export interface VMRegisterMeta {
  size_bits: number
  name: string
}

export interface VMInfo {
  isa: string
  max_steps: number
  default_memory: number
  max_memory: number
  registers: Record<string, VMRegisterMeta>
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

  async trainingJob(jobId: number): Promise<VMTrainingJob> {
    return apiGet<VMTrainingJob>(`/vm/training/jobs/${jobId}`)
  }

  async stopTrainingJob(jobId: number): Promise<{ status: string; job_id: number }> {
    return apiPost(`/vm/training/jobs/${jobId}/stop`, {})
  }
}

export const vmController = new VMController()
