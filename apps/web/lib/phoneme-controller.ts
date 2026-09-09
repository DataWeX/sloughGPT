import { apiGet, apiPost } from './http-client'

export interface PhonemeEncodeResult {
  text: string
  language: string
  phonemes: string[]
  ids: number[]
  decoded: string
}

export interface PhonemeScoreResult {
  target: string
  spoken: string
  language: string
  score: number
  precision: number
  recall: number
  target_phonemes: string[]
  spoken_phonemes: string[]
}

export interface PhonemeBatchEncodeResult {
  results: PhonemeEncodeResult[]
  count: number
}

export interface PhonemeBatchScoreResult {
  results: PhonemeScoreResult[]
  count: number
}

export interface PhonemeDetectResult {
  text: string
  language: string
  supported_languages: string[]
}

export interface PhonemeSynthesizeResult {
  audio: string
  text: string
  duration_sec: number
  elapsed_ms: number
}

export type PhonemeLanguage = 'en' | 'de' | 'fr' | 'es' | 'it' | 'pt'

export const PHONEME_LANGUAGES: { value: PhonemeLanguage; label: string }[] = [
  { value: 'en', label: 'English' },
  { value: 'de', label: 'German' },
  { value: 'fr', label: 'French' },
  { value: 'es', label: 'Spanish' },
  { value: 'it', label: 'Italian' },
  { value: 'pt', label: 'Portuguese' },
]

export const phonemeController = {
  async encode(text: string, language?: PhonemeLanguage): Promise<PhonemeEncodeResult> {
    return apiPost<PhonemeEncodeResult>('/multimodal/encode-phonemes', { text, language })
  },

  async decode(ids: number[], language: PhonemeLanguage): Promise<{ ids: number[]; language: string; phonemes: string[]; decoded: string }> {
    return apiPost<{ ids: number[]; language: string; phonemes: string[]; decoded: string }>('/multimodal/decode-phonemes', { ids, language })
  },

  async score(target: string, spoken: string, language?: PhonemeLanguage): Promise<PhonemeScoreResult> {
    return apiPost<PhonemeScoreResult>('/multimodal/score-pronunciation', { target, spoken, language })
  },

  async batchEncode(texts: string[], language?: PhonemeLanguage): Promise<PhonemeBatchEncodeResult> {
    return apiPost<PhonemeBatchEncodeResult>('/multimodal/batch-encode-phonemes', { texts, language })
  },

  async batchScore(pairs: { target: string; spoken: string }[], language?: PhonemeLanguage): Promise<PhonemeBatchScoreResult> {
    return apiPost<PhonemeBatchScoreResult>('/multimodal/batch-score-pronunciation', { pairs, language })
  },

  async detectLanguage(text: string): Promise<PhonemeDetectResult> {
    return apiPost<PhonemeDetectResult>('/multimodal/detect-language', { text })
  },

  async synthesize(text: string): Promise<PhonemeSynthesizeResult> {
    const formData = new FormData()
    formData.append('text', text)
    return apiPost<PhonemeSynthesizeResult>('/multimodal/synthesize-speech', formData)
  },
}
