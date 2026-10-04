'use client'

import { useRef, useMemo, useEffect } from 'react'
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
  const color =
    variant === 'mic'
      ? 'bg-primary'
      : variant === 'speaker'
        ? 'bg-success'
        : 'bg-muted-foreground/30'

  const glowColor =
    variant === 'mic' ? 'shadow-primary/20' : variant === 'speaker' ? 'shadow-success/20' : ''

  return (
    <div
      ref={containerRef}
      className={cn('flex items-center justify-center gap-[2px]', className)}
      style={{ width, height }}
      role="img"
      aria-label={`Audio ${variant} waveform`}
    >
      {barHeights.map((h, i) => (
        <div
          key={i}
          className={cn(
            color,
            'rounded-full transition-all duration-100 ease-out',
            glowColor && `shadow-sm ${glowColor}`,
          )}
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

export type VoiceVisualState = 'idle' | 'listening' | 'processing' | 'speaking' | 'error'

/**
 * Single source of truth for state -> color. The header status dot and the
 * orb ring both read this map, so the two surfaces can never drift apart.
 */
export const VOICE_STATE_TONE: Record<
  VoiceVisualState,
  { dot: string; border: string; fill: string; pulse: boolean }
> = {
  idle: {
    dot: 'bg-muted-foreground/30',
    border: 'border-muted-foreground/20',
    fill: 'color-mix(in srgb, rgb(var(--muted)) 75%, transparent)',
    pulse: false,
  },
  listening: {
    dot: 'bg-primary',
    border: 'border-primary',
    fill: 'color-mix(in srgb, rgb(var(--primary)) 16%, transparent)',
    pulse: true,
  },
  processing: {
    dot: 'bg-warning',
    border: 'border-warning',
    fill: 'color-mix(in srgb, rgb(var(--warning)) 16%, transparent)',
    pulse: true,
  },
  speaking: {
    dot: 'bg-success',
    border: 'border-success',
    fill: 'color-mix(in srgb, rgb(var(--success)) 16%, transparent)',
    pulse: true,
  },
  error: {
    dot: 'bg-destructive',
    border: 'border-destructive',
    fill: 'color-mix(in srgb, rgb(var(--destructive)) 16%, transparent)',
    pulse: true,
  },
}

export function voiceStateTone(state: VoiceVisualState) {
  return VOICE_STATE_TONE[state] ?? VOICE_STATE_TONE.idle
}

/**
 * Hero orb — the single presentation surface for the voice state machine.
 * Wraps children (the mic toggle) with state rings driven by CSS variables.
 */
export function VoiceOrb({
  state,
  micLevel,
  levelRef,
  children,
}: {
  state: VoiceVisualState
  /** Throttled level (0–1) from useVoiceChat — fallback when no `levelRef`. */
  micLevel?: number
  /** Per-frame level ref: sampled on the orb's own rAF loop, so the React tree is never re-rendered at audio rate. */
  levelRef?: React.RefObject<number | null>
  children: React.ReactNode
}) {
  const orbRef = useRef<HTMLDivElement>(null)
  const tone = voiceStateTone(state)
  const isListening = state === 'listening'
  const isSpeaking = state === 'speaking'
  const isProcessing = state === 'processing'

  // Per-frame level -> `--mic-level` written imperatively. CSS consumes the
  // variable for every scale/opacity; React only paints on discrete state changes.
  useEffect(() => {
    const el = orbRef.current
    if (!el || !levelRef) return
    let raf = 0
    const tick = () => {
      el.style.setProperty('--mic-level', String(levelRef.current ?? 0))
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [levelRef])

  const level = 'var(--mic-level, 0)'
  const motion = 'var(--voice-motion, 1)'

  return (
    <div className="voice-orb relative" data-state={state}>
      {/* Listening corona — scale/opacity read --mic-level straight from the DOM */}
      {isListening && (
        <>
          <span
            aria-hidden="true"
            className="absolute inset-0 rounded-full border-2 border-primary/25"
            style={{ transform: `scale(calc(1.25 + ${level} * 0.55 * ${motion}))`, opacity: `calc(0.25 + ${level} * 0.4)` }}
          />
          <span
            aria-hidden="true"
            className="absolute inset-0 rounded-full border border-primary/15"
            style={{ transform: `scale(calc(1.1 + ${level} * 0.28 * ${motion}))`, opacity: 0.2 }}
          />
          <span
            aria-hidden="true"
            className="absolute inset-0 rounded-full bg-primary/10 blur-xl"
            style={{ transform: `scale(calc(1 + ${level} * 0.6 * ${motion}))`, opacity: `calc(0.25 + ${level} * 0.45)` }}
          />
        </>
      )}

      {isSpeaking && (
        <>
          <span
            aria-hidden="true"
            className="absolute inset-0 rounded-full border-2 border-success/25 animate-ping"
            style={{ animationDuration: '1.5s' }}
          />
          <span
            aria-hidden="true"
            className="absolute inset-0 rounded-full border border-success/15 animate-ping"
            style={{ animationDuration: '2s', animationDelay: '0.3s' }}
          />
        </>
      )}

      {isProcessing && (
        <span
          aria-hidden="true"
          className="absolute inset-0 rounded-full border-2 border-warning/30 animate-spin"
          style={{ animationDuration: '3s' }}
        />
      )}

      {/* Main orb — one hero element, tone from the shared state map */}
      <div
        ref={orbRef}
        className={cn(
          'relative z-10 w-44 h-44 rounded-full flex items-center justify-center border-2 shadow-xl transition-colors duration-300',
          tone.border,
        )}
        style={
          {
            '--mic-level': micLevel ?? 0,
            transform: `scale(calc(1 + ${level} * 0.22 * ${motion}))`,
          } as React.CSSProperties
        }
      >
        {/* Base wash + directional glows; intensity follows the mic level */}
        <span
          aria-hidden="true"
          className="absolute inset-0 rounded-full pointer-events-none"
          style={{ background: tone.fill, opacity: `calc(0.7 + ${level} * 0.3)` }}
        />
        <span
          aria-hidden="true"
          className="absolute inset-0 rounded-full pointer-events-none"
          style={{
            background:
              'radial-gradient(circle at 34% 28%, color-mix(in srgb, rgb(var(--primary)) 45%, transparent), transparent 58%), radial-gradient(circle at 72% 78%, color-mix(in srgb, rgb(var(--accent)) 32%, transparent), transparent 55%)',
            opacity: `calc(0.45 + ${level} * 0.55)`,
          }}
        />
        {children}
      </div>
    </div>
  )
}

/**
 * Animated listening indicator with pulsing concentric rings
 * that respond to mic level. Shows when actively listening.
 */
export function ListeningIndicator({ micLevel, active }: { micLevel: number; active: boolean }) {
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
  const bars = useMemo(
    () =>
      Array.from({ length: barCount }, (_, i) => ({
        delay: (i * 120) % 800,
        baseHeight: 20 + Math.random() * 30,
      })),
    [barCount],
  )

  if (!active) return null

  return (
    <div
      className="flex items-center justify-center gap-[3px] h-10"
      role="img"
      aria-label="Listening animation"
    >
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
