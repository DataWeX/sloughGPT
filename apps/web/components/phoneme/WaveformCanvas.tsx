'use client'

import { useRef, useEffect } from 'react'

interface WaveformCanvasProps {
  audioSrc: string
  className?: string
}

function decodeBase64Wav(base64: string): Float32Array | null {
  try {
    const raw = atob(base64)
    const bytes = new Uint8Array(raw.length)
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i)

    if (raw.length < 44) return null
    const view = new DataView(bytes.buffer)
    const bitsPerSample = view.getUint16(34, true)
    const numChannels = view.getUint16(22, true)
    const sampleRate = view.getUint32(24, true)
    const dataSize = view.getUint32(40, true)
    const dataStart = 44

    if (bitsPerSample !== 16) return null

    const sampleCount = Math.floor(dataSize / (numChannels * 2))
    const samples = new Float32Array(sampleCount)
    for (let i = 0; i < sampleCount; i++) {
      const offset = dataStart + i * numChannels * 2
      if (offset + 1 >= bytes.length) break
      const rawSample = view.getInt16(offset, true)
      samples[i] = rawSample / 32768
    }
    return samples
  } catch {
    return null
  }
}

function downsample(samples: Float32Array, targetPoints: number): Float32Array {
  if (samples.length <= targetPoints) return samples
  const blockSize = Math.floor(samples.length / targetPoints)
  const result = new Float32Array(targetPoints)
  for (let i = 0; i < targetPoints; i++) {
    const start = i * blockSize
    let sum = 0
    for (let j = 0; j < blockSize; j++) {
      sum += Math.abs(samples[start + j])
    }
    result[i] = sum / blockSize
  }
  return result
}

export default function WaveformCanvas({ audioSrc, className }: WaveformCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !audioSrc) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const base64 = audioSrc.replace('data:audio/wav;base64,', '')
    const samples = decodeBase64Wav(base64)
    if (!samples || samples.length === 0) return

    const width = canvas.width
    const height = canvas.height
    const centerY = height / 2
    const targetPoints = width

    const amplitude = downsample(samples, targetPoints)

    // Clear with background
    ctx.fillStyle = 'rgb(var(--background))'
    ctx.fillRect(0, 0, width, height)

    // Draw center line
    ctx.strokeStyle = 'rgb(var(--border))'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, centerY)
    ctx.lineTo(width, centerY)
    ctx.stroke()

    // Draw waveform
    const gradient = ctx.createLinearGradient(0, 0, width, 0)
    gradient.addColorStop(0, 'rgb(var(--primary))')
    gradient.addColorStop(1, 'rgb(var(--accent))')

    ctx.fillStyle = gradient
    const barWidth = Math.max(1, width / targetPoints)

    for (let i = 0; i < amplitude.length; i++) {
      const x = (i / amplitude.length) * width
      const barHeight = amplitude[i] * height * 0.9
      ctx.fillRect(x, centerY - barHeight / 2, barWidth, barHeight)
    }
  }, [audioSrc])

  return (
    <div className={className}>
      <div className="text-xs text-muted-foreground mb-1 flex justify-between">
        <span>Amplitude</span>
        <span>Time</span>
      </div>
      <canvas
        ref={canvasRef}
        width={320}
        height={60}
        className="w-full h-15 rounded-md border border-border"
        aria-label="Audio waveform visualization"
        role="img"
      />
    </div>
  )
}
