/**
 * Extract a human-readable error message from an unknown caught value.
 */
export function extractErrorMessage(err: unknown, fallback = 'Unknown error'): string {
  if (err instanceof Error) return err.message
  if (typeof err === 'string') return err
  return fallback
}

/**
 * Format an error for toast display.
 */
export function formatToastError(err: unknown, prefix: string): string {
  return `${prefix}: ${extractErrorMessage(err)}`
}
