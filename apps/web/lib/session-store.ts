/**
 * In-memory session-scoped store.
 *
 * Survives navigation (SPA) but resets on full page reload / tab close.
 * Used by the download confirmation dialog to remember "Don't ask again
 * for this session" choices per model.
 */

const _approved = new Set<string>()

export const sessionStore = {
  isApproved(key: string): boolean {
    return _approved.has(key)
  },

  setApproved(key: string): void {
    _approved.add(key)
  },

  reset(): void {
    _approved.clear()
  },
}
