/** next/web-vitals no-op — Phase 0; wire `web-vitals` package in a later phase. */

export type WebVitalsMetric = {
  name: string
  value: number
  rating?: string
  id?: string
  navigationType?: string
  [key: string]: unknown
}

export function useReportWebVitals(callback: (metric: WebVitalsMetric) => void): void {
  void callback
}
