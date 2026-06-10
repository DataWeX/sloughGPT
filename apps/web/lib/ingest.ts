/**
 * Auto-ingestion trigger — actual ingestion runs in Python core.
 * Frontend just triggers ChromaDB init.
 */

'use client'

import { apiPost } from './http-client'

export async function initVectorStore(): Promise<void> {
  try {
    await apiPost('/vector/init', { provider: 'chromadb', dimension: 384 })
  } catch { /* silent */ }
}

if (typeof window !== 'undefined') {
  ;(window as any).__man_ingest_ready = true
}
