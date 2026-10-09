export function _corrId(): string {
  return Math.random().toString(16).slice(2, 10)
}

const _recentCorrIds: Array<{ id: string; url: string; ts: number }> = []
const MAX_RECENT = 20

export function getRecentCorrelationIds(): Array<{ id: string; url: string; ts: number }> {
  return _recentCorrIds.slice()
}

export function _trackCorrId(corrId: string, url: string) {
  _recentCorrIds.push({ id: corrId, url, ts: Date.now() })
  if (_recentCorrIds.length > MAX_RECENT) _recentCorrIds.shift()
}
