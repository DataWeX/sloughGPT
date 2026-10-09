/**
 * Voice Controller — API for TTS/voice operations.
 */

import { apiGet, apiPost } from './http-client'

export interface VoiceStatus {
  server_tts: boolean
  model: string | null
  error: string | null
}

export interface TTSResult {
  audio?: string
  duration_ms: number
  backend: string
  sample_rate: number
  detail?: string
}

class VoiceController {
  async getStatus(): Promise<VoiceStatus> {
    return apiGet<VoiceStatus>('/voice/status')
  }

  async tts(text: string): Promise<TTSResult> {
    return apiPost('/voice/tts', { text })
  }

  async playAudio(base64Audio: string, sampleRate: number): Promise<void> {
    const audio = new Audio(`data:audio/wav;base64,${base64Audio}`)
    return new Promise((resolve, reject) => {
      audio.onended = () => resolve()
      audio.onerror = (e) => reject(e)
      audio.play().catch(reject)
    })
  }
}

export const voiceController = new VoiceController()
