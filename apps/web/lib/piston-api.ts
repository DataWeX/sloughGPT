/**
 * External API wrappers - third-party services
 */

import { createApiClient } from './http-client'

const PISTON_API_URL = 'https://emkc.org/api/v2/piston'
const pistonClient = createApiClient(PISTON_API_URL)

export interface PistonExecuteRequest {
  language: string
  version: string
  files: Array<{ content: string }>
}

export interface PistonExecuteResponse {
  run?: {
    stdout: string
    stderr: string
    output: string
    code: number
    signal: string | null
  }
  compile?: {
    stdout: string
    stderr: string
    output: string
    code: number
    signal: string | null
  }
}

export interface PistonRuntime {
  language: string
  version: string
  aliases: string[]
}



export async function executeCode(
  code: string,
  language: string = 'python',
  version: string = '3.10'
): Promise<{ output: string; error?: string }> {
  try {
    const { data } = await pistonClient.post<PistonExecuteResponse>('/execute', {
      language,
      version,
      files: [{ content: code }],
    })

    if (data.run?.output) {
      return { output: data.run.output }
    } else if (data.run?.stderr) {
      return { output: '', error: data.run.stderr }
    }
    return { output: 'Code executed successfully (no output)' }
  } catch (err) {
    if (err instanceof Error) {
      return { output: '', error: `HTTP ${err.message}` }
    }
    return { output: '', error: 'Failed to execute code' }
  }
}

export async function getPistonRuntimes(): Promise<PistonRuntime[]> {
  try {
    const { data } = await pistonClient.get<PistonRuntime[]>('/runtimes')
    return data
  } catch {
    return []
  }
}