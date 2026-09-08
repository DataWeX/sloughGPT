/**
 * Audio Filter Controller — API client for audio filter configuration and processing.
 */

import { apiGet, apiPost } from './http-client'

export interface AudioFilterConfig {
  sample_rate: number
  mode: string
  noise_gate_threshold_db: number
  noise_gate_attack_ms: number
  noise_gate_release_ms: number
  agc_target_db: number
  agc_max_gain_db: number
  agc_frame_ms: number
  target_lufs: number
  vad_energy_threshold_db: number
  vad_min_speech_ms: number
  vad_silence_ratio: number
}

export interface AudioFilterUpdate {
  mode?: string
  noise_gate_threshold_db?: number
  noise_gate_attack_ms?: number
  noise_gate_release_ms?: number
  agc_target_db?: number
  agc_max_gain_db?: number
  agc_frame_ms?: number
  target_lufs?: number
  vad_energy_threshold_db?: number
  vad_min_speech_ms?: number
  vad_silence_ratio?: number
}

export interface AudioFilterResult {
  audio: string
  sample_rate: number
  speech_detected: boolean
  gain_applied_db: number
  frames_gate_open: number
  frames_total: number
}

class AudioFilterController {
  async getConfig(): Promise<AudioFilterConfig> {
    return apiGet<AudioFilterConfig>('/multimodal/audio-filter/config')
  }

  async updateConfig(update: AudioFilterUpdate): Promise<{ status: string }> {
    return apiPost('/multimodal/audio-filter/config', update)
  }

  async processAudio(audioBlob: Blob): Promise<AudioFilterResult> {
    const formData = new FormData()
    formData.append('file', audioBlob, 'audio.wav')
    const res = await fetch(`${window.location.origin}/api/multimodal/audio-filter/process`, {
      method: 'POST',
      body: formData,
    })
    if (!res.ok) throw new Error(`Audio filter failed (${res.status})`)
    const json = await res.json()
    return json.data
  }
}

export const audioFilterController = new AudioFilterController()
