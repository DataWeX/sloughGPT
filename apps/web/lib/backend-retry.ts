'use client'

import { revalidateAll } from '@/lib/cache'
import { apiGet } from '@/lib/http-client'

/**
 * ONE retry path for every backend-connection surface (card 87345605): the
 * status-bar Retry button, the startup overlay's stuck-screen Retry and the
 * backend-down banner action all probe silently and revalidate in place.
 *
 * Poll first, never reload: a document reload threw away drafts, open panels
 * and in-flight streams, and under a genuinely-down backend it looped
 * (reload → fail → overlay → Retry → reload). The probe is silent (no error
 * toast — the status surfaces already report the outage) and short (5s) so a
 * Retry answer arrives while the user is still looking at the button.
 */
export async function probeBackendAndRevalidate(): Promise<boolean> {
  try {
    await apiGet('/health', undefined, { signal: AbortSignal.timeout(5000), silent: true })
    revalidateAll()
    return true
  } catch {
    // The existing monitors keep reporting; a failed probe changes nothing.
    return false
  }
}
