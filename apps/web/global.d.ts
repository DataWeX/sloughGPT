/* eslint-disable no-var */

/**
 * Global type declarations for browser APIs not in standard lib.dom.d.ts
 * and Next.js runtime globals.
 */

interface Window {
  /** Next.js inlined runtime config (set by __NEXT_DATA__ or build-time injection) */
  __NEXT_PUBLIC_API_URL?: string

  /** Web Speech API — Chrome, Edge */
  SpeechRecognition?: typeof SpeechRecognition
  /** Web Speech API — Safari/WebKit prefix */
  webkitSpeechRecognition?: typeof SpeechRecognition
}
