'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
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

async function getWaveformData(url: string, bins = 128): Promise<{ timeDomain: Float32Array; frequency: Float32Array }> {
  return new Promise((resolve) => {
    const audio = new Audio(url)
    audio.crossOrigin = 'anonymous'

    audio.addEventListener('canplaythrough', async () => {
      try {
        const ctx = new (window.AudioContext || (window as any).webkitAudioContext)()
        const source = ctx.createMediaElementSource(audio)
        const analyser = ctx.createAnalyser()
        analyser.fftSize = bins * 2
        source.connect(analyser)
        analyser.connect(ctx.destination)

        const timeDomain = new Float32Array(analyser.frequencyBinCount)
        const frequency = new Float32Array(analyser.frequencyBinCount)
        analyser.getFloatTimeDomainData(timeDomain)
        analyser.getFloatFrequencyData(frequency)

        resolve({ timeDomain, frequency })
        ctx.close()
      } catch {
        resolve({ timeDomain: new Float32Array(bins), frequency: new Float32Array(bins) })
      }
    }, { once: true })

    audio.addEventListener('error', () => {
      resolve({ timeDomain: new Float32Array(bins), frequency: new Float32Array(bins) })
    }, { once: true })

    audio.load()
  })
}

function TimeDomainCanvas({ data, width = 400, height = 80 }: { data: Float32Array; width?: number; height?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || data.length === 0) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, width, height)
    ctx.strokeStyle = '#8b5cf6'
    ctx.lineWidth = 1.5
    ctx.beginPath()

    const sliceWidth = width / data.length
    let x = 0
    for (let i = 0; i < data.length; i++) {
      const y = (1 - data[i]) * height / 2
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
      x += sliceWidth
    }
    ctx.stroke()
  }, [data, width, height])

  return <canvas ref={canvasRef} width={width} height={height} className="w-full rounded-md bg-muted/30" role="img" aria-label="Waveform visualization" />
}

function FrequencyCanvas({ data, width = 400, height = 80 }: { data: Float32Array; width?: number; height?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || data.length === 0) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, width, height)
    const barWidth = width / data.length
    const minDb = Math.min(...Array.from(data))
    const maxDb = Math.max(...Array.from(data))
    const range = maxDb - minDb || 1

    for (let i = 0; i < data.length; i++) {
      const normalized = (data[i] - minDb) / range
      const barHeight = normalized * height
      const hue = (i / data.length) * 270
      ctx.fillStyle = `hsl(${hue}, 70%, 60%)`
      ctx.fillRect(i * barWidth, height - barHeight, barWidth - 0.5, barHeight)
    }
  }, [data, width, height])

  return <canvas ref={canvasRef} width={width} height={height} className="w-full rounded-md bg-muted/30" role="img" aria-label="Frequency spectrum" />
}

export function VoiceWaveformCard() {
  const [recordings, setRecordings] = useState<Recording[]>([])
  const [loaded, setLoaded] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [waveData, setWaveData] = useState<{ timeDomain: Float32Array; frequency: Float32Array } | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [stats, setStats] = useState<{
    rms: number; peak: number; dynamicRange: number; zeroCrossings: number
  } | null>(null)

  useEffect(() => {
    loadRecordings().then(data => {
      setRecordings(data)
      setLoaded(true)
    })
  }, [])

  const analyze = useCallback(async (id: string) => {
    const rec = recordings.find(r => r.id === id)
    if (!rec) return
    setAnalyzing(true)
    setWaveData(null)
    setStats(null)

    const wave = await getWaveformData(rec.url)
    setWaveData(wave)

    let sumSquares = 0
    let peak = 0
    let zeroCrossings = 0
    const td = wave.timeDomain
    for (let i = 0; i < td.length; i++) {
      const v = td[i]
      sumSquares += v * v
      const abs = Math.abs(v)
      if (abs > peak) peak = abs
      if (i > 0 && ((td[i - 1] >= 0 && v < 0) || (td[i - 1] < 0 && v >= 0))) zeroCrossings++
    }
    const rms = Math.sqrt(sumSquares / td.length)

    let min = Infinity; let max = -Infinity
    for (let i = 0; i < td.length; i++) {
      if (td[i] < min) min = td[i]
      if (td[i] > max) max = td[i]
    }

    setStats({
      rms,
      peak,
      dynamicRange: max - min,
      zeroCrossings,
    })
    setAnalyzing(false)
  }, [recordings])

  const handleSelect = (id: string) => {
    setSelectedId(id)
    analyze(id)
  }

  const rec = recordings.find(r => r.id === selectedId)

  return (
    <Card data-testid="voice-waveform">
      <CardHeader><CardTitle className="text-base">Waveform Analysis</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <Select value={selectedId ?? ''} onValueChange={handleSelect}>
          <SelectTrigger className="w-full text-xs" aria-label="Select recording to analyze">
            <SelectValue placeholder="Choose a recording..." />
          </SelectTrigger>
          <SelectContent>
            {recordings.map(r => (
              <SelectItem key={r.id} value={r.id}>
                {r.label} ({(r.duration / 1000).toFixed(1)}s)
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {analyzing && (
          <div className="space-y-2">
            <div className="h-20 animate-pulse bg-muted/50 rounded-md" />
            <div className="h-20 animate-pulse bg-muted/50 rounded-md" />
          </div>
        )}

        {waveData && !analyzing && (
          <div className="space-y-3">
            <div>
              <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide mb-1">Time Domain</p>
              <TimeDomainCanvas data={waveData.timeDomain} />
            </div>
            <div>
              <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide mb-1">Frequency Spectrum</p>
              <FrequencyCanvas data={waveData.frequency} />
            </div>
          </div>
        )}

        {stats && !analyzing && (
          <div className="grid grid-cols-4 gap-2" data-testid="waveform-stats">
            {[
              { label: 'RMS Level', value: (stats.rms * 100).toFixed(1) + '%' },
              { label: 'Peak', value: (stats.peak * 100).toFixed(1) + '%' },
              { label: 'Dynamic Range', value: (stats.dynamicRange * 100).toFixed(1) + '%' },
              { label: 'Zero Crossings', value: stats.zeroCrossings.toString() },
            ].map(item => (
              <div key={item.label} className="text-center p-2 rounded-md bg-muted/30">
                <p className="text-[10px] text-muted-foreground">{item.label}</p>
                <p className="text-xs font-mono font-medium">{item.value}</p>
              </div>
            ))}
          </div>
        )}

        {selectedId && rec && (
          <Button size="sm" variant="ghost" className="h-7 text-[10px]" onClick={() => new Audio(rec.url).play()}>
            Play Recording
          </Button>
        )}

        {!selectedId && recordings.length === 0 && (
          <p className="text-xs text-muted-foreground text-center py-4">No recordings yet. Record something first.</p>
        )}
      </CardContent>
    </Card>
  )
}
