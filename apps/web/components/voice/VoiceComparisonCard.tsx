'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'
import { chatDB } from '@/lib/db'

interface Recording {
  id: string
  blob: Blob
  url: string
  duration: number
  label: string
  timestamp: number
}

const RECORDINGS_KEY = 'sloughgpt-voice-recordings'

async function loadRecordings(): Promise<Recording[]> {
  try {
    const entry = await chatDB.getKV<Recording[]>(RECORDINGS_KEY)
    if (entry && Array.isArray(entry)) return entry
  } catch { /* corrupted */ }
  return []
}

function formatDuration(ms: number) {
  const s = Math.floor(ms / 1000)
  const m = Math.floor(s / 60)
  return `${m}:${(s % 60).toString().padStart(2, '0')}`
}

function getPeakData(url: string, bins = 64): Promise<Float32Array> {
  return new Promise((resolve) => {
    const audio = new Audio(url)
    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)()
    const analyser = ctx.createAnalyser()
    analyser.fftSize = bins * 2

    audio.addEventListener('canplaythrough', async () => {
      try {
        const source = ctx.createMediaElementSource(audio)
        source.connect(analyser)
        analyser.connect(ctx.destination)
        const data = new Float32Array(analyser.frequencyBinCount)
        analyser.getFloatTimeDomainData(data)
        resolve(data)
      } catch {
        resolve(new Float32Array(bins))
      }
      ctx.close()
    }, { once: true })

    audio.addEventListener('error', () => {
      resolve(new Float32Array(bins))
      ctx.close()
    }, { once: true })

    audio.load()
  })
}

function WaveformBars({ data, color }: { data: Float32Array; color: string }) {
  const peaks = Array.from(data)
  return (
    <div className="flex items-center gap-px h-8" role="img" aria-label="Waveform visualization">
      {peaks.map((v, i) => (
        <div
          key={i}
          className="flex-1 rounded-sm min-w-[1px] transition-all duration-300"
          style={{
            height: `${Math.max(2, Math.abs(v) * 100)}%`,
            backgroundColor: color,
            opacity: 0.4 + Math.abs(v) * 0.6,
          }}
        />
      ))}
    </div>
  )
}

export function VoiceComparisonCard() {
  const [recordings, setRecordings] = useState<Recording[]>([])
  const [loaded, setLoaded] = useState(false)
  const [leftId, setLeftId] = useState<string | null>(null)
  const [rightId, setRightId] = useState<string | null>(null)
  const [leftWave, setLeftWave] = useState<Float32Array | null>(null)
  const [rightWave, setRightWave] = useState<Float32Array | null>(null)
  const [comparing, setComparing] = useState(false)
  const [result, setResult] = useState<{ similarity: number; durationDiff: number; details: string } | null>(null)
  const playingRef = useRef<{ left: HTMLAudioElement | null; right: HTMLAudioElement | null }>({ left: null, right: null })

  useEffect(() => {
    loadRecordings().then(data => {
      setRecordings(data)
      setLoaded(true)
    })
  }, [])

  const selectLeft = async (id: string) => {
    setLeftId(id)
    setResult(null)
    setLeftWave(null)
    const rec = recordings.find(r => r.id === id)
    if (rec) {
      const wave = await getPeakData(rec.url)
      setLeftWave(wave)
    }
  }

  const selectRight = async (id: string) => {
    setRightId(id)
    setResult(null)
    setRightWave(null)
    const rec = recordings.find(r => r.id === id)
    if (rec) {
      const wave = await getPeakData(rec.url)
      setRightWave(wave)
    }
  }

  const runComparison = useCallback(() => {
    if (!leftId || !rightId) return
    setComparing(true)

    const left = recordings.find(r => r.id === leftId)
    const right = recordings.find(r => r.id === rightId)
    if (!left || !right) { setComparing(false); return }

    const durationDiff = Math.abs(left.duration - right.duration)
    const durationRatio = Math.min(left.duration, right.duration) / Math.max(left.duration, right.duration)

    let similarity = 0
    if (leftWave && rightWave && leftWave.length > 0 && rightWave.length > 0) {
      const len = Math.min(leftWave.length, rightWave.length)
      let dotProduct = 0
      let normA = 0
      let normB = 0
      for (let i = 0; i < len; i++) {
        dotProduct += leftWave[i] * rightWave[i]
        normA += leftWave[i] * leftWave[i]
        normB += rightWave[i] * rightWave[i]
      }
      similarity = normA > 0 && normB > 0
        ? (dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)) + 1) / 2
        : 0.5
    }

    const combinedScore = (similarity * 0.6 + durationRatio * 0.4)

    const details = [
      `Duration: ${formatDuration(left.duration)} vs ${formatDuration(right.duration)} (Δ ${formatDuration(durationDiff)})`,
      `Waveform similarity: ${(similarity * 100).toFixed(1)}%`,
      `Duration match: ${(durationRatio * 100).toFixed(1)}%`,
      `Overall match: ${(combinedScore * 100).toFixed(1)}%`,
    ].join(' · ')

    setResult({ similarity: combinedScore, durationDiff, details })
    setComparing(false)
  }, [leftId, rightId, recordings, leftWave, rightWave])

  const playSide = useCallback((side: 'left' | 'right') => {
    const id = side === 'left' ? leftId : rightId
    const rec = recordings.find(r => r.id === id)
    if (!rec) return

    if (playingRef.current[side]) {
      playingRef.current[side]!.pause()
      playingRef.current[side] = null
    }

    const audio = new Audio(rec.url)
    playingRef.current[side] = audio
    audio.play()
    audio.addEventListener('ended', () => { playingRef.current[side] = null }, { once: true })
  }, [leftId, rightId, recordings])

  const playSequence = useCallback(() => {
    if (!leftId || !rightId) return
    const leftRec = recordings.find(r => r.id === leftId)
    const rightRec = recordings.find(r => r.id === rightId)
    if (!leftRec || !rightRec) return

    const leftAudio = new Audio(leftRec.url)
    const rightAudio = new Audio(rightRec.url)
    leftAudio.addEventListener('ended', () => { rightAudio.play() }, { once: true })
    leftAudio.play()
  }, [leftId, rightId, recordings])

  const left = recordings.find(r => r.id === leftId)
  const right = recordings.find(r => r.id === rightId)

  if (recordings.length < 2) {
    return (
      <Card data-testid="voice-comparison">
        <CardHeader><CardTitle className="text-base">Voice Comparison</CardTitle></CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground">
            {recordings.length === 0 ? 'Record at least 2 clips to compare.' : 'Record at least 1 more clip to compare.'}
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card data-testid="voice-comparison">
      <CardHeader><CardTitle className="text-base">Voice Comparison</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">Recording A</p>
            <select
              className="w-full text-xs border border-border/60 rounded-md p-1.5 bg-background"
              value={leftId ?? ''}
              onChange={(e) => selectLeft(e.target.value)}
              aria-label="Select recording A"
            >
              <option value="">Choose...</option>
              {recordings.map(r => (
                <option key={r.id} value={r.id}>{r.label} ({formatDuration(r.duration)})</option>
              ))}
            </select>
            {leftWave && <WaveformBars data={leftWave} color="#8b5cf6" />}
            {left && <Button size="sm" variant="ghost" onClick={() => playSide('left')} className="h-6 text-[10px]">Play A</Button>}
          </div>
          <div className="space-y-1.5">
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">Recording B</p>
            <select
              className="w-full text-xs border border-border/60 rounded-md p-1.5 bg-background"
              value={rightId ?? ''}
              onChange={(e) => selectRight(e.target.value)}
              aria-label="Select recording B"
            >
              <option value="">Choose...</option>
              {recordings.map(r => (
                <option key={r.id} value={r.id}>{r.label} ({formatDuration(r.duration)})</option>
              ))}
            </select>
            {rightWave && <WaveformBars data={rightWave} color="#22c55e" />}
            {right && <Button size="sm" variant="ghost" onClick={() => playSide('right')} className="h-6 text-[10px]">Play B</Button>}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button
            size="sm"
            className="h-7 text-[11px]"
            onClick={runComparison}
            disabled={!leftId || !rightId || comparing}
          >
            {comparing ? 'Comparing...' : 'Compare'}
          </Button>
          {leftId && rightId && (
            <Button size="sm" variant="ghost" className="h-7 text-[10px]" onClick={playSequence}>
              Play A → B
            </Button>
          )}
        </div>

        {result && (
          <div className="rounded-md border border-border/60 p-3 text-xs space-y-2" data-testid="comparison-result">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Match score</span>
              <span className="font-mono font-medium">{(result.similarity * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${result.similarity * 100}%`,
                  backgroundColor: result.similarity > 0.7 ? '#22c55e' : result.similarity > 0.4 ? '#f59e0b' : '#ef4444',
                }}
              />
            </div>
            <p className="text-[10px] text-muted-foreground/60 leading-relaxed">{result.details}</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
