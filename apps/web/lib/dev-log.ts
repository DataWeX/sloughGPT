/** Logs only in development — keeps production consoles clean for expected API/network failures. */
export function devDebug(...args: unknown[]) {
  if (process.env.NODE_ENV === 'development') {
    console.debug('[sloughgpt]', ...args)
  }
}
