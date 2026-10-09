'use client'

import { useRef, useEffect } from 'react'
import type { MelSpectrogram } from '@/lib/phoneme-controller'

interface SpectrogramCanvasProps {
  spectrogram: MelSpectrogram
  className?: string
}

function normalizeMelData(data: number[][]): number[][] {
  if (!data || data.length === 0) return []

  let globalMin = Infinity
  let globalMax = -Infinity
  for (const row of data) {
    for (const v of row) {
      if (v < globalMin) globalMin = v
      if (v > globalMax) globalMax = v
    }
  }

  const range = globalMax - globalMin || 1
  return data.map(row => row.map(v => (v - globalMin) / range))
}

function interpolateColor(t: number): [number, number, number, number] {
  // Noir Violet palette: deep charcoal → violet → lilac → peach
  if (t < 0.25) {
    const s = t / 0.25
    return [
      Math.round(17 + s * (52 - 17)),
      Math.round(15 + s * (46 - 15)),
      Math.round(24 + s * (72 - 24)),
      255,
    ]
  }
  if (t < 0.5) {
    const s = (t - 0.25) / 0.25
    return [
      Math.round(52 + s * (124 - 52)),
      Math.round(46 + s * (82 - 46)),
      Math.round(72 + s * (196 - 72)),
      255,
    ]
  }
  if (t < 0.75) {
    const s = (t - 0.5) / 0.25
    return [
      Math.round(124 + s * (192 - 124)),
      Math.round(82 + s * (170 - 82)),
      Math.round(196 + s * (244 - 196)),
      255,
    ]
  }
  const s = (t - 0.75) / 0.25
  return [
    Math.round(192 + s * (236 - 192)),
    Math.round(170 + s * (145 - 170)),
    Math.round(244 + s * (95 - 244)),
    255,
  ]
}

export default function SpectrogramCanvas({ spectrogram, className }: SpectrogramCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const { data, n_mels, n_frames } = spectrogram
    if (!data || n_frames === 0 || n_mels === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height
    const cellW = Math.max(1, Math.ceil(width / n_frames))
    const cellH = Math.max(1, Math.ceil(height / n_mels))

    const imageData = ctx.createImageData(width, height)
    const pixels = imageData.data

    // Fill with charcoal background
    for (let i = 0; i < pixels.length; i += 4) {
      pixels[i] = 17
      pixels[i + 1] = 15
      pixels[i + 2] = 24
      pixels[i + 3] = 255
    }

    const normalized = normalizeMelData(data)

    for (let mel = 0; mel < n_mels; mel++) {
      const row = normalized[n_mels - 1 - mel] // flip vertically (low freq at bottom)
      if (!row) continue
      for (let frame = 0; frame < n_frames; frame++) {
        const t = row[frame] ?? 0
        const [r, g, b, a] = interpolateColor(t)
        const x0 = Math.floor((frame / n_frames) * width)
        const y0 = Math.floor((mel / n_mels) * height)
        const x1 = Math.min(width, x0 + cellW)
        const y1 = Math.min(height, y0 + cellH)

        for (let y = y0; y < y1; y++) {
          for (let x = x0; x < x1; x++) {
            const idx = (y * width + x) * 4
            pixels[idx] = r
            pixels[idx + 1] = g
            pixels[idx + 2] = b
            pixels[idx + 3] = a
          }
        }
      }
    }

    ctx.putImageData(imageData, 0, 0)
  }, [spectrogram])

  return (
    <div className={className}>
      <div className="text-xs text-muted-foreground mb-1 flex justify-between">
        <span>Low frequency</span>
        <span>High frequency</span>
      </div>
      <canvas
        ref={canvasRef}
        width={320}
        height={80}
        className="w-full h-20 rounded-md border border-border"
        aria-label="Mel spectrogram visualization"
        role="img"
      />
      <div className="text-xs text-muted-foreground mt-1 flex justify-between">
        <span>Start</span>
        <span>End</span>
      </div>
    </div>
  )
}
