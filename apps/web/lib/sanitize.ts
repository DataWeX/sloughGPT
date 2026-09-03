/**
 * Input sanitization utilities.
 *
 * All user-facing inputs pass through these functions before being sent
 * to the backend or rendered in the UI.
 */

/* ── Numeric clamping ───────────────────────────────────────── */

/** Clamp a number to [min, max]. Returns fallback if value is not finite. */
export function clampNumber(value: number, min: number, max: number, fallback: number = min): number {
  if (!Number.isFinite(value)) return fallback
  return Math.min(Math.max(value, min), max)
}

/** Safely parse a string to number, clamping to [min, max]. */
export function parseAndClamp(raw: string, min: number, max: number, fallback: number = min): number {
  const n = Number(raw)
  return clampNumber(n, min, max, fallback)
}

/* ── HTML/script sanitization ───────────────────────────────── */

/** Characters that must be escaped in HTML content. */
const HTML_ESCAPES: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#x27;',
  '/': '&#x2F;',
  '`': '&#96;',
}

/** Escape HTML special characters to prevent XSS. */
export function escapeHtml(str: string): string {
  return str.replace(/[&<>"'`/]/g, ch => HTML_ESCAPES[ch] ?? ch)
}

/** Strip <script>...</script> blocks and their content. */
export function stripScriptTags(str: string): string {
  return str.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, '')
}

/** Strip all HTML tags, keeping only text content. */
export function stripHtmlTags(str: string): string {
  return str.replace(/<[^>]*>/g, '')
}

/**
 * Sanitize a free-text input for safe storage.
 * - Strips script tags
 * - Trims whitespace
 * - Enforces max length
 */
export function sanitizeText(input: string, maxLength: number = 10_000): string {
  return stripScriptTags(input).trim().slice(0, maxLength)
}

/**
 * Sanitize a text field for safe HTML rendering.
 * Escapes all HTML entities so the text is displayed literally.
 */
export function sanitizeForDisplay(input: string, maxLength: number = 10_000): string {
  return escapeHtml(stripScriptTags(input).trim()).slice(0, maxLength)
}

/* ── Field-level validators ─────────────────────────────────── */

export interface FieldLimit {
  min: number
  max: number
  step?: number
}

/** Backend field limits — single source of truth for config validation. */
export const TRAINING_LIMITS = {
  // Training fields
  epochs:                  { min: 1,     max: 1000,  step: 1     },
  learning_rate:           { min: 1e-5,  max: 1.0,   step: 1e-5  },
  batch_size:              { min: 1,     max: 512,   step: 1     },
  temperature:             { min: 0.1,   max: 2.0,   step: 0.1   },
  early_stopping_patience: { min: 0,     max: 100,   step: 1     },
  n_embed:                 { min: 16,    max: 1024,  step: 16    },
  n_layer:                 { min: 1,     max: 24,    step: 1     },
  n_head:                  { min: 1,     max: 64,    step: 1     },
  block_size:              { min: 8,     max: 2048,  step: 8     },
  dropout:                 { min: 0.0,   max: 0.9,   step: 0.05  },
  vocab_size:              { min: 50,    max: 50000, step: 50    },
  rank:                    { min: 1,     max: 64,    step: 1     },
  max_tgt_len:             { min: 16,    max: 2048,  step: 16    },
  soul_name:               { min: 1,     max: 200,   step: 1     },
  teacher_model:           { min: 1,     max: 200,   step: 1     },
  // Inference fields
  max_tokens:              { min: 1,     max: 4096,  step: 1     },
  top_p:                   { min: 0.0,   max: 1.0,   step: 0.05  },
  top_k:                   { min: 1,     max: 200,   step: 1     },
  repetition_penalty:      { min: 1.0,   max: 2.0,   step: 0.05  },
  // World/Simulation fields
  width:                   { min: 1,     max: 1024,  step: 1     },
  height:                  { min: 1,     max: 1024,  step: 1     },
  samples:                 { min: 1,     max: 256,   step: 1     },
  camera_height:           { min: 0,     max: 500,   step: 0.5   },
  camera_distance:         { min: 0,     max: 500,   step: 0.5   },
} as const

/** Get limits for a training field (returns undefined if field unknown). */
export function getFieldLimit(key: string): FieldLimit | undefined {
  return TRAINING_LIMITS[key as keyof typeof TRAINING_LIMITS]
}

/** Clamp a training config field to its backend limit. */
export function clampTrainingField(key: string, value: number): number {
  const limit = getFieldLimit(key)
  if (!limit) return value
  return clampNumber(value, limit.min, limit.max)
}
