/**
 * Voice Controller — API for TTS/voice operations.
 */

import { apiGet, apiPost, apiPostForm } from './http-client'

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

export interface TranscribeResult {
  text: string
  language: string
  elapsed_ms: number
}

class VoiceController {
  async getStatus(): Promise<VoiceStatus> {
    return apiGet<VoiceStatus>('/voice/status')
  }

  async tts(text: string): Promise<TTSResult> {
    return apiPost('/voice/tts', { text })
  }

  /**
   * Server-side speech-to-text via the domain VoiceEngine
   * (POST /multimodal/transcribe). The server's real contract is raw
   * PCM s16le mono @ 16 kHz (AudioData(data, 16000, 2) /
   * KaldiRecognizer(model, 16000)) — pass `audio/l16` blobs, not
   * container-encoded audio (vosk cannot parse webm/ogg).
   */
  async transcribe(file: Blob, language = 'en'): Promise<TranscribeResult> {
    const form = new FormData()
    form.append('file', file, 'utterance.pcm')
    form.append('language', language)
    return apiPostForm<TranscribeResult>('/multimodal/transcribe', form)
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
