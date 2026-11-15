import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { createSpeechEngine, floatToPcm16, type SpeechEngineHandlers } from './speech-engine'

// ── Mocks ──────────────────────────────────────────────────────────────

vi.mock('@/lib/voice-controller', () => ({
  voiceController: {
    transcribe: vi
      .fn()
      .mockResolvedValue({ text: 'hello from server', language: 'en', elapsed_ms: 12 }),
    tts: vi.fn(),
    playAudio: vi.fn(),
    getStatus: vi.fn(),
  },
}))

import { voiceController } from '@/lib/voice-controller'

const transcribeMock = voiceController.transcribe as ReturnType<typeof vi.fn>

// ── Browser SR fake (same shape as the hook test) ──────────────────────

function makeMockRecognition() {
  const handlers: Record<string, ((...args: unknown[]) => void) | null> = {
    start: null,
    end: null,
    error: null,
    result: null,
  }
  const instance = {
    continuous: false,
    interimResults: false,
    lang: '',
    start: vi.fn(),
    stop: vi.fn(),
    abort: vi.fn(),
    get onstart() {
      return handlers.start
    },
    set onstart(fn) {
      handlers.start = fn
    },
    get onend() {
      return handlers.end
    },
    set onend(fn) {
      handlers.end = fn
    },
    get onerror() {
      return handlers.error
    },
    set onerror(fn) {
      handlers.error = fn
    },
    get onresult() {
      return handlers.result
    },
    set onresult(fn) {
      handlers.result = fn
    },
  }
  return { instance, handlers }
}

// ── AudioContext fake with capture wiring ──────────────────────────────

interface FakeCtx {
  ctx: {
    sampleRate: number
    createMediaStreamSource: ReturnType<typeof vi.fn>
    createScriptProcessor: ReturnType<typeof vi.fn>
    createGain: ReturnType<typeof vi.fn>
    close: ReturnType<typeof vi.fn>
  }
  processor: {
    connect: ReturnType<typeof vi.fn>
    disconnect: ReturnType<typeof vi.fn>
    onaudioprocess: ((e: unknown) => void) | null
  }
}

function installFakeAudioContext(sampleRate = 16000): () => FakeCtx[] {
  const all: FakeCtx[] = []
  ;(window as any).AudioContext = vi.fn((opts?: { sampleRate?: number }) => {
    const processor: FakeCtx['processor'] = {
      connect: vi.fn(),
      disconnect: vi.fn(),
      onaudioprocess: null,
    }
    const ctx = {
      sampleRate: opts?.sampleRate ?? sampleRate,
      createMediaStreamSource: vi.fn(() => ({ connect: vi.fn(), disconnect: vi.fn() })),
      createScriptProcessor: vi.fn(() => processor),
      createGain: vi.fn(() => ({ gain: { value: 0 }, connect: vi.fn(), disconnect: vi.fn() })),
      close: vi.fn(),
    }
    const entry = { ctx, processor }
    all.push(entry)
    return ctx
  })
  return () => all
}

function feed(processor: FakeCtx['processor'], samples: Float32Array) {
  processor.onaudioprocess?.({
    inputBuffer: { getChannelData: () => samples },
  })
}

const CHUNK = 4096
const speechChunk = () => new Float32Array(CHUNK).fill(0.2) // rms 0.2 > 0.01
const silenceChunk = () => new Float32Array(CHUNK) // rms 0

async function flushMicrotasks() {
  for (let i = 0; i < 10; i++) await Promise.resolve()
}

const makeStream = () => ({ getTracks: () => [{ stop: vi.fn() }] }) as unknown as MediaStream

function collectHandlers(): SpeechEngineHandlers & {
  starts: number
  ends: number
  finals: string[]
  interleims: string[]
  errors: Array<[string, boolean]>
  modes: string[]
} {
  return {
    starts: 0,
    ends: 0,
    finals: [],
    interleims: [],
    errors: [],
    modes: [],
    onStart() {
      this.starts++
    },
    onEnd() {
      this.ends++
    },
    onFinal(t: string) {
      this.finals.push(t)
    },
    onInterim(t: string) {
      this.interleims.push(t)
    },
    onError(m: string, f: boolean) {
      this.errors.push([m, f])
    },
    onMode(m: string) {
      this.modes.push(m)
    },
  }
}

// ── Tests ──────────────────────────────────────────────────────────────

describe('floatToPcm16', () => {
  it('converts and clips at 16 kHz (identity rate)', () => {
    const input = Float32Array.from([0, 0.5, -0.5, 2, -2])
    const out = floatToPcm16(input, 16000)
    expect(out.length).toBe(5)
    expect(out[0]).toBe(0)
    expect(out[1]).toBe(Math.floor(0.5 * 0x7fff))
    expect(out[3]).toBe(0x7fff) // clipped high
    expect(out[4]).toBe(-0x8000) // clipped low
  })

  it('linear-resamples from 48 kHz to 16 kHz', () => {
    const input = new Float32Array(480) // 10 ms @ 48 kHz
    const out = floatToPcm16(input, 48000)
    expect(out.length).toBe(160) // 10 ms @ 16 kHz
  })
})

describe('createSpeechEngine — browser mode', () => {
  let mockRec: ReturnType<typeof makeMockRecognition>

  beforeEach(() => {
    transcribeMock.mockClear()
    transcribeMock.mockResolvedValue({ text: 'hello from server', language: 'en', elapsed_ms: 12 })
    mockRec = makeMockRecognition()
    ;(window as any).SpeechRecognition = vi.fn(() => mockRec.instance)
    ;(navigator as any).mediaDevices = {
      getUserMedia: vi.fn().mockResolvedValue(makeStream()),
    }
  })

  afterEach(() => {
    delete (window as any).SpeechRecognition
    delete (window as any).webkitSpeechRecognition
    delete (window as any).AudioContext
    delete (navigator as any).mediaDevices
  })

  it('uses the Web Speech API when available and wires results', () => {
    const h = collectHandlers()
    const engine = createSpeechEngine(h)
    engine.start()
    expect(engine.mode).toBe('browser')
    expect(h.modes).toEqual(['browser'])
    expect(mockRec.instance.start).toHaveBeenCalled()
    expect(mockRec.instance.continuous).toBe(true)
    expect(mockRec.instance.interimResults).toBe(true)

    mockRec.handlers.start?.()
    expect(h.starts).toBe(1)

    mockRec.handlers.result?.({
      resultIndex: 0,
      results: [{ isFinal: false, length: 1, 0: { transcript: 'hel', confidence: 0.5 } }],
    })
    expect(h.interleims).toEqual(['hel'])

    mockRec.handlers.result?.({
      resultIndex: 0,
      results: [{ isFinal: true, length: 1, 0: { transcript: 'hello there', confidence: 0.9 } }],
    })
    expect(h.finals).toEqual(['hello there'])
  })

  it("ignores 'aborted' (no failover, no error)", () => {
    const h = collectHandlers()
    const engine = createSpeechEngine(h)
    engine.start()
    mockRec.handlers.error?.({ error: 'aborted' })
    expect(h.errors).toEqual([])
    expect(h.modes).toEqual(['browser'])
    expect(engine.mode).toBe('browser')
  })

  it("fails over to server mode on 'network' error", async () => {
    const ctxs = installFakeAudioContext()
    const h = collectHandlers()
    const engine = createSpeechEngine(h)
    engine.start(makeStream())
    expect(engine.mode).toBe('browser')

    mockRec.handlers.error?.({ error: 'network' })
    await flushMicrotasks()
    expect(engine.mode).toBe('server')
    expect(h.modes).toEqual(['browser', 'server'])
    expect(ctxs().length).toBe(1) // capture context created
    expect(h.starts).toBe(1) // onStart fired again for server mode
    expect(h.errors).toEqual([]) // failover is silent
  })
})

describe('createSpeechEngine — server mode', () => {
  let getCtxs: () => FakeCtx[]

  beforeEach(() => {
    transcribeMock.mockClear()
    transcribeMock.mockResolvedValue({ text: 'hello from server', language: 'en', elapsed_ms: 12 })
    delete (window as any).SpeechRecognition
    delete (window as any).webkitSpeechRecognition
    getCtxs = installFakeAudioContext(16000)
    ;(navigator as any).mediaDevices = {
      getUserMedia: vi.fn().mockResolvedValue(makeStream()),
    }
  })

  afterEach(() => {
    delete (window as any).AudioContext
    delete (navigator as any).mediaDevices
  })

  it('engages directly when SpeechRecognition is absent', async () => {
    const h = collectHandlers()
    const engine = createSpeechEngine(h)
    engine.start(makeStream())
    expect(engine.mode).toBe('server')
    expect(h.modes).toEqual(['server'])
    expect(h.starts).toBe(1)
  })

  it('speech burst + silence → transcribes the utterance', async () => {
    const h = collectHandlers()
    const engine = createSpeechEngine(h)
    engine.start(makeStream())
    const processor = getCtxs()[0].processor

    feed(processor, speechChunk()) // speech onset
    feed(processor, speechChunk()) // 8192 samples ≈ 512 ms @ 16 kHz
    expect(transcribeMock).not.toHaveBeenCalled()

    // Silence ≥ END_SILENCE_MS (19200 samples @ 16 kHz) → finalize
    for (let i = 0; i < 6; i++) feed(processor, silenceChunk())
    await flushMicrotasks()

    expect(transcribeMock).toHaveBeenCalledTimes(1)
    const blob = transcribeMock.mock.calls[0][0] as Blob
    expect(blob.type).toBe('audio/l16')
    expect(h.finals).toEqual(['hello from server'])
  })

  it('stop() flushes pending speech (capture up to stop is transcribed)', async () => {
    const h = collectHandlers()
    const engine = createSpeechEngine(h)
    engine.start(makeStream())
    const processor = getCtxs()[0].processor

    feed(processor, speechChunk())
    feed(processor, speechChunk()) // 512 ms @ 16 kHz — above the 300 ms floor
    engine.stop()
    await flushMicrotasks()

    expect(transcribeMock).toHaveBeenCalledTimes(1)
    expect(h.ends).toBe(1)
    // Flush-on-stop still delivers the final — that's the point of flushing.
    expect(h.finals).toEqual(['hello from server'])
  })

  it('abort() drops captured audio without transcribing', async () => {
    const h = collectHandlers()
    const engine = createSpeechEngine(h)
    engine.start(makeStream())
    const processor = getCtxs()[0].processor

    feed(processor, speechChunk())
    engine.abort()
    await flushMicrotasks()

    expect(transcribeMock).not.toHaveBeenCalled()
    expect(h.ends).toBe(1)
  })

  it('ignores pre-speech silence (nothing captured before onset)', async () => {
    const h = collectHandlers()
    const engine = createSpeechEngine(h)
    engine.start(makeStream())
    const processor = getCtxs()[0].processor

    for (let i = 0; i < 10; i++) feed(processor, silenceChunk())
    engine.stop()
    await flushMicrotasks()

    expect(transcribeMock).not.toHaveBeenCalled()
  })

  it('mic denied (no stream given) → fatal error + end', async () => {
    ;(navigator as any).mediaDevices.getUserMedia = vi.fn().mockRejectedValue(new Error('denied'))
    const h = collectHandlers()
    const engine = createSpeechEngine(h)
    engine.start(null)
    await flushMicrotasks()

    expect(h.errors).toEqual([['Microphone access denied', true]])
    expect(h.ends).toBe(1)
    expect(getCtxs().length).toBe(0) // no capture context created
  })

  it('transcribe failure surfaces a non-fatal error', async () => {
    transcribeMock.mockRejectedValueOnce(new Error('boom'))
    const h = collectHandlers()
    const engine = createSpeechEngine(h)
    engine.start(makeStream())
    const processor = getCtxs()[0].processor

    feed(processor, speechChunk())
    feed(processor, speechChunk())
    for (let i = 0; i < 6; i++) feed(processor, silenceChunk())
    await flushMicrotasks()

    expect(h.errors).toEqual([['Transcription failed', false]])
    expect(h.finals).toEqual([]) // engine not spent — listening continues
  })

  it('stops owned stream tracks on stop()', async () => {
    const stopTrack = vi.fn()
    ;(navigator as any).mediaDevices.getUserMedia = vi
      .fn()
      .mockResolvedValue({ getTracks: () => [{ stop: stopTrack }] })
    const h = collectHandlers()
    const engine = createSpeechEngine(h)
    engine.start(null) // engine owns the stream it opened
    engine.stop() // stop lands while getUserMedia is in flight...
    await flushMicrotasks() // ...stream resolves, in-flight guard releases it
    expect(stopTrack).toHaveBeenCalled()
  })

  it('never stops a stream the caller provided (hook owns it)', () => {
    const stopTrack = vi.fn()
    const stream = { getTracks: () => [{ stop: stopTrack }] } as unknown as MediaStream
    const h = collectHandlers()
    const engine = createSpeechEngine(h)
    engine.start(stream)
    engine.stop()
    expect(stopTrack).not.toHaveBeenCalled()
  })
})
