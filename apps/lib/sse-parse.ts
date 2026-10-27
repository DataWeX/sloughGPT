// ═══════════════════════════════════════════════════════════════════════════
// SSE wire-frame parser — pure, no I/O, no app dependencies.
// ═══════════════════════════════════════════════════════════════════════════
//
// Single implementation of the chunk → line → data: → JSON.parse state machine,
// shared by BOTH stream control flows:
//   - `streamSSE`      (apps/lib/sse.ts — async generator + retry)
//   - `createSSEStream` (apps/web/lib/sse-client.ts — callbacks + reconnect)
//
// It lives here rather than in apps/web so the transport never imports the app
// (that dependency used to run the wrong way: apps/lib/sse.ts imported this
// parser from ../web/lib/sse-client, while sse-client re-implemented the
// transport itself). apps/web/lib/sse-client.ts re-exports these for callers
// that reach it through the app layer.
// ═══════════════════════════════════════════════════════════════════════════

export interface SSEParseResult {
  /** Parsed payloads extracted from this chunk (already JSON-decoded). */
  events: unknown[]
  /** Bytes that didn't yet form a complete line; carry into next chunk. */
  buffer: string
}

export function consumeSSEChunk(
  buffer: string,
  decoder: TextDecoder,
  chunk: Uint8Array,
  onMalformed?: (payload: string) => void,
): SSEParseResult {
  let nextBuffer = buffer + decoder.decode(chunk, { stream: true })
  const lines = nextBuffer.split('\n')
  nextBuffer = lines.pop() || ''
  const events: unknown[] = []
  for (const line of lines) {
    const trimmed = line.trimEnd()
    if (!trimmed.startsWith('data:')) continue
    const payload = trimmed.slice(5).trim()
    if (!payload || payload === '[DONE]') continue
    try {
      events.push(JSON.parse(payload))
    } catch {
      onMalformed?.(payload.slice(0, 80))
    }
  }
  return { events, buffer: nextBuffer }
}

export function consumeSSEDrain(
  buffer: string,
  onMalformed?: (payload: string) => void,
): unknown[] {
  const trimmed = buffer.trimEnd()
  if (!trimmed.startsWith('data:')) return []
  const payload = trimmed.slice(5).trim()
  if (!payload || payload === '[DONE]') return []
  try {
    return [JSON.parse(payload)]
  } catch {
    onMalformed?.(payload.slice(0, 80))
    return []
  }
}
