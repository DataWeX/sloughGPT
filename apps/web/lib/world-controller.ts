'use client'

import { apiGet, apiPost } from './http-client'

export interface RenderConfig {
  width?: number
  height?: number
  samples?: number
  camera_height?: number
  camera_distance?: number
}

export interface RenderResult {
  shapes: Record<string, number[]>
  stats: Record<string, unknown>
  tensor_keys: string[]
}

export interface TickResult {
  tick: number
  babies: number
  render_stats: Record<string, unknown> | null
}

export interface NeuralResult {
  embedding_shape: number[] | null
  descriptor: Record<string, unknown>
  stats: Record<string, unknown>
}

export interface WorldStats {
  status: string
  components: string[]
  materials: Record<string, number>
}

export const worldController = {
  async render(config?: RenderConfig): Promise<RenderResult> {
    return apiPost<RenderResult>('/world/render', config ?? {})
  },

  async renderImage(config?: RenderConfig): Promise<Blob> {
    const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/world/render/image`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config ?? {}),
    })
    if (!res.ok) throw new Error('Render failed')
    return res.blob()
  },

  async neuralProcess(config?: RenderConfig): Promise<NeuralResult> {
    return apiPost<NeuralResult>('/world/neural', config ?? {})
  },

  async tick(maxTicks = 1, render = true, neural = false): Promise<TickResult> {
    return apiPost<TickResult>('/world/tick', { max_ticks: maxTicks, render, neural })
  },

  async stats(): Promise<WorldStats> {
    return apiGet<WorldStats>('/world/stats')
  },
}
