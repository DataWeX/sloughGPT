'use client'

import { useRef, useMemo } from 'react'
import { cn } from '@sloughgpt/strui'

interface VoiceWaveformProps {
  /** 0–1 mic or speaker level from useVoiceChat */
  level: number
  /** Number of bars to render */
  bars?: number
  /** Visual variant */
  variant?: 'mic' | 'speaker' | 'idle'
  /** Width in pixels */
  width?: number
  /** Height in pixels */
  height?: number
  /** CSS class */
  className?: string
}

/**
 * Real-time audio waveform visualizer.
 * Renders frequency-style bars that respond to `level` (0–1).
 * Uses CSS transitions for smooth animation — no canvas needed.
 */
export function VoiceWaveform({
  level,
  bars = 24,
  variant = 'idle',
  width = 200,
  height = 60,
  className = '',
}: VoiceWaveformProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  // Map level to individual bar heights with some randomness for organic feel
  const barHeights = useMemo(() => {
    const result: number[] = []
    for (let i = 0; i < bars; i++) {
      const center = Math.abs(i - bars / 2) / (bars / 2) // 0 at center, 1 at edges
      const base = level * (1 - center * 0.6) // center bars taller
      const jitter = variant === 'idle' ? 0.05 : level * 0.15 * Math.sin(Date.now() / 200 + i * 0.7)
      result.push(Math.max(0.05, Math.min(1, base + jitter)))
    }
    return result
  }, [level, bars, variant])

  // Color based on variant
  const color = variant === 'mic'
    ? 'bg-primary'
    : variant === 'speaker'
      ? 'bg-emerald-500'
      : 'bg-muted-foreground/30'

  const glowColor = variant === 'mic'
    ? 'shadow-primary/20'
    : variant === 'speaker'
      ? 'shadow-emerald-500/20'
      : ''

  return (
    <div
      ref={containerRef}
      className={cn("flex items-center justify-center gap-[2px]", className)}
      style={{ width, height }}
      role="img"
      aria-label={`Audio ${variant} waveform`}
    >
      {barHeights.map((h, i) => (
        <div
          key={i}
          className={`${color} rounded-full transition-all duration-100 ease-out ${glowColor ? `shadow-sm ${glowColor}` : ''}`}
          style={{
            width: Math.max(2, (width - bars * 2) / bars),
            height: `${h * height * 0.9}px`,
            minHeight: '2px',
            opacity: variant === 'idle' ? 0.3 : 0.6 + h * 0.4,
          }}
        />
      ))}
    </div>
  )
}

/**
 * Circular pulsing orb for voice states.
 * Wraps children with animated rings based on state.
 */
export function VoiceOrb({
  state,
  micLevel,
  children,
}: {
  state: 'idle' | 'listening' | 'processing' | 'speaking' | 'error'
  micLevel: number
  children: React.ReactNode
}) {
  const isListening = state === 'listening'
  const isSpeaking = state === 'speaking'
  const isProcessing = state === 'processing'

  // Scale the orb based on mic level when listening
  const orbScale = isListening ? 1 + micLevel * 0.4 : 1
  const orbColor = isListening
    ? 'border-primary'
    : isSpeaking
      ? 'border-emerald-500'
      : isProcessing
        ? 'border-amber-500'
        : state === 'error'
          ? 'border-red-500'
          : 'border-muted-foreground/20'

  return (
    <div className="relative">
      {/* Outer rings */}
      {isListening && (
        <>
          <div
            className="absolute inset-0 rounded-full border-2 border-primary/20 transition-transform duration-200"
            style={{ transform: `scale(${orbScale * 1.6})`, opacity: 0.2 + micLevel * 0.3 }}
          />
          <div
            className="absolute inset-0 rounded-full border border-primary/10 transition-transform duration-300"
            style={{ transform: `scale(${orbScale * 1.3})`, opacity: 0.15 }}
          />
        </>
      )}

      {isSpeaking && (
        <>
          <div
            className="absolute inset-0 rounded-full border-2 border-emerald-500/20 animate-ping"
            style={{ animationDuration: '1.5s' }}
          />
          <div
            className="absolute inset-0 rounded-full border border-emerald-500/10 animate-ping"
            style={{ animationDuration: '2s', animationDelay: '0.3s' }}
          />
        </>
      )}

      {isProcessing && (
        <div className="absolute inset-0 rounded-full border-2 border-amber-500/20 animate-spin" style={{ animationDuration: '3s' }} />
      )}

      {/* Main orb */}
      <div
        className={`
          relative z-10 w-24 h-24 rounded-full flex items-center justify-center
          border-2 ${orbColor} transition-all duration-200 shadow-lg
        `}
        style={{ transform: `scale(${orbScale})` }}
      >
        {children}
      </div>
    </div>
  )
}

/**
 * Animated listening indicator with pulsing concentric rings
 * that respond to mic level. Shows when actively listening.
 */
export function ListeningIndicator({
  micLevel,
  active,
}: {
  micLevel: number
  active: boolean
}) {
  // Generate ring data based on mic level
  const rings = useMemo(() => {
    const count = 5
    return Array.from({ length: count }, (_, i) => ({
      delay: i * 200,
      baseScale: 1.2 + i * 0.4,
      color: `hsl(var(--primary) / ${0.15 - i * 0.02})`,
    }))
  }, [])

  if (!active) return null

  return (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
      {rings.map((ring, i) => (
        <div
          key={i}
          className="absolute rounded-full border border-primary/20"
          style={{
            width: '200px',
            height: '200px',
            transform: `scale(${ring.baseScale + micLevel * 0.5})`,
            opacity: active ? 0.15 + micLevel * 0.2 - i * 0.02 : 0,
            transition: 'transform 150ms ease-out, opacity 150ms ease-out',
            animation: `listening-pulse 2s ease-in-out ${ring.delay}ms infinite`,
          }}
        />
      ))}
      {/* Center glow */}
      <div
        className="absolute w-32 h-32 rounded-full bg-primary/5 blur-xl"
        style={{
          transform: `scale(${1 + micLevel * 0.8})`,
          opacity: 0.3 + micLevel * 0.4,
          transition: 'transform 100ms ease-out, opacity 100ms ease-out',
        }}
      />
    </div>
  )
}

/**
 * Animated vertical bars that bounce when listening.
 * Each bar has a random delay for organic feel.
 */
export function ListeningBars({
  micLevel,
  active,
  barCount = 7,
}: {
  micLevel: number
  active: boolean
  barCount?: number
}) {
  const bars = useMemo(() =>
    Array.from({ length: barCount }, (_, i) => ({
      delay: (i * 120) % 800,
      baseHeight: 20 + Math.random() * 30,
    })),
    [barCount]
  )

  if (!active) return null

  return (
    <div className="flex items-center justify-center gap-[3px] h-10" role="img" aria-label="Listening animation">
      {bars.map((bar, i) => (
        <div
          key={i}
          className="w-1 rounded-full bg-primary/60"
          style={{
            height: `${bar.baseHeight + micLevel * 40}px`,
            animation: `listening-bar 0.8s ease-in-out ${bar.delay}ms infinite`,
            opacity: 0.4 + micLevel * 0.5,
            transition: 'height 100ms ease-out',
          }}
        />
      ))}
    </div>
  )
}
