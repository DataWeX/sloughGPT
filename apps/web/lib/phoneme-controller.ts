import { apiGet, apiPost } from './http-client'

export const IPA_MAP: Record<string, string> = {
  'AA': 'ɑː', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔː', 'AW': 'aʊ', 'AY': 'aɪ',
  'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'ɝː',
  'EY': 'eɪ', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'IH': 'ɪ', 'IY': 'iː',
  'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ',
  'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
  'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'uː', 'V': 'v', 'W': 'w',
  'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
  'BOS': '', 'EOS': '', 'PAD': '', 'SPACE': ' ', 'SILENCE': '',
}

export function toIPA(phonemes: string[]): string[] {
  return phonemes.map(p => IPA_MAP[p] || p)
}

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

export interface MelSpectrogram {
  data: number[][]
  n_mels: number
  n_frames: number
}

export interface PhonemeSynthesizeResult {
  audio: string
  text: string
  duration_sec: number
  elapsed_ms: number
  spectrogram: MelSpectrogram
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
