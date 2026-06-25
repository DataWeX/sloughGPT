import { apiPost, apiGet } from '@/lib/http-client'

export interface TTSResponse {
  audio: string
  sample_rate: number
  duration_ms: number
  backend: 'hf-model' | 'browser-fallback'
}

export interface VoiceStatusResponse {
  server_tts: boolean
  model: string | null
  error: string | null
}

export const voiceController = {
  /** Convert text to speech audio. Returns base64 WAV or browser-fallback signal. */
  async tts(text: string): Promise<TTSResponse> {
    return apiPost<TTSResponse>('/voice/tts', { text })
  },

  /** Check if server-side TTS model is loaded. */
  async status(): Promise<VoiceStatusResponse> {
    return apiGet<VoiceStatusResponse>('/voice/status')
  },

  /** Play base64 WAV audio data through an Audio element. Returns a cleanup fn. */
  playAudio(base64Wav: string, sampleRate: number): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const binary = atob(base64Wav)
        const bytes = new Uint8Array(binary.length)
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
        const blob = new Blob([bytes], { type: 'audio/wav' })
        const url = URL.createObjectURL(blob)
        const audio = new Audio(url)
        audio.onended = () => { URL.revokeObjectURL(url); resolve() }
        audio.onerror = (e) => { URL.revokeObjectURL(url); reject(e) }
        audio.play()
      } catch (e) {
        reject(e)
      }
    })
  },
}
