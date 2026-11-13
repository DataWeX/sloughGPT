// ═══════════════════════════════════════════════════════════════════════════
// clock — server-authoritative time for gating decisions (card 20261108_134212)
// ═══════════════════════════════════════════════════════════════════════════
//
// The API already stamps every response with an HTTP `Date` header (uvicorn
// `date_header=True` default) — no endpoint, no backend change. This module
// turns that header into one offset so TTL/staleTime/timeout gates follow the
// server clock instead of the client's (a skewed client clock expires leases
// early/late and misjudges cache freshness).
//
// INVARIANT: a duration comparison must have BOTH operands on the same clock.
// If you convert a read to `serverNow()`, convert every write site of that
// timestamp too — mixing `Date.now()` and `serverNow()` on one value is the
// exact bug class this module removes.
//
// Notes:
// - HTTP `Date` has 1s granularity, so the offset carries ±500ms quantization
//   — fine for TTL/staleTime scales (seconds+), not for sub-second logic.
// - Initial offset is 0 (trust the local clock until the first response).
// - Last sample wins: each response re-anchors the offset.

/** ms to add to `Date.now()` to get server time. 0 until first sample. */
let offsetMs = 0

/**
 * Re-anchor the offset from a response's `Date` header. Missing or
 * unparseable headers are ignored (keeps the previous offset).
 */
export function sampleServerTime(headers: Headers): void {
  const dateHeader = headers.get('date')
  if (!dateHeader) return
  const serverMs = Date.parse(dateHeader)
  if (Number.isNaN(serverMs)) return
  offsetMs = serverMs - Date.now()
}

/** Server-authoritative "now" in epoch ms. Use this for BOTH sides of a gating comparison. */
export function serverNow(): number {
  return Date.now() + offsetMs
}

/** Current offset (server − local) in ms. Diagnostics/tests. */
export function getServerClockOffsetMs(): number {
  return offsetMs
}

/** Reset to the un-sampled state (offset 0). Tests/diagnostics. */
export function resetServerClock(): void {
  offsetMs = 0
}
