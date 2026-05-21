'use client'

import { useEffect, useState } from 'react'
import { cn } from '@/lib/cn'

interface TypingDotsProps {
  className?: string
  size?: 'sm' | 'md' | 'lg'
  color?: 'muted' | 'primary' | 'gradient' | 'rainbow'
}

export function TypingDots({ className, size = 'md', color = 'primary' }: TypingDotsProps) {
  const [hue, setHue] = useState(260)

  useEffect(() => {
    if (color !== 'rainbow') return
    const interval = setInterval(() => setHue(h => (h + 2) % 360), 30)
    return () => clearInterval(interval)
  }, [color])

  const sizes = {
    sm: 'h-1.5 w-1.5',
    md: 'h-2 w-2',
    lg: 'h-2.5 w-2.5',
  }

  const gradients = {
    muted: 'bg-muted-foreground/60',
    primary: 'bg-primary',
    gradient: 'bg-gradient-to-r from-primary to-violet-500',
  }

  const getDotColor = (i: number) => {
    if (color !== 'rainbow') return null
    const dotHue = (hue + i * 45) % 360
    return { backgroundColor: `hsl(${dotHue}, 65%, 55%)` }
  }

  return (
    <div className={cn('flex items-center gap-1', className)}>
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className={cn(
            'rounded-full animate-bounce shadow-sm',
            sizes[size],
            color !== 'rainbow' && gradients[color],
          )}
          style={{
            animationDelay: `${i * 150}ms`,
            animationDuration: '600ms',
            ...getDotColor(i),
          }}
        />
      ))}
    </div>
  )
}

interface TypingIndicatorProps {
  className?: string
  showLabel?: boolean
  label?: string
}

const playfulLabels = [
  'Thinking',
  ' pondering',
  ' doodling',
  ' daydreaming',
  ' tinkering',
  ' cooking',
  ' brewing',
  ' sketching',
]

export function TypingIndicator({ className, showLabel = true, label }: TypingIndicatorProps) {
  const [labelIndex, setLabelIndex] = useState(0)
  const [dots, setDots] = useState('')

  useEffect(() => {
    const labelInterval = setInterval(() => {
      setLabelIndex(i => (i + 1) % playfulLabels.length)
    }, 3000)
    return () => clearInterval(labelInterval)
  }, [])

  useEffect(() => {
    const dotInterval = setInterval(() => {
      setDots(d => d.length >= 3 ? '' : d + '.')
    }, 400)
    return () => clearInterval(dotInterval)
  }, [])

  return (
    <div className={cn('flex items-center gap-2', className)}>
      {showLabel && (
        <span className="text-xs text-muted-foreground/60 font-medium transition-all duration-500">
          {label || playfulLabels[labelIndex]}{dots}
        </span>
      )}
      <TypingDots size="sm" color="rainbow" />
    </div>
  )
}
