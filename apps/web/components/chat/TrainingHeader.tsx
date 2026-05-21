'use client'

import { cn } from '@/lib/cn'
import { Button } from '@/components/ui/button'

export interface TrainingConfig {
  teacherModel: string
  temperature: string
  learningRate: string
  modelPath: string
  isRunning: boolean
  stepCount: number
  currentLoss: number | null
  hasMessages?: boolean
}

interface TrainingHeaderProps {
  config: TrainingConfig
  onStart: () => void
  onStop: () => void
  onReset?: () => void
  className?: string
}

export function TrainingHeader({ 
  config, 
  onStart, 
  onStop,
  onReset,
  className 
}: TrainingHeaderProps) {
  return (
    <div className={cn("flex items-center justify-between gap-2 px-2 py-1.5 bg-muted rounded-lg", className)}>
      <div className="flex items-center gap-2">
        <h1 className="text-sm font-semibold">Auto-Train</h1>
        
        {config.isRunning && (
          <div className="flex items-center gap-1 px-1.5 py-0.5 bg-success/10 rounded-full">
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success opacity-75"></span>
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-success"></span>
            </span>
            <span className="text-xs text-success font-medium">Training</span>
          </div>
        )}
        
        <span className="text-xs text-muted-foreground hidden sm:inline">
          {config.teacherModel}
        </span>
      </div>
      
      <div className="flex items-center gap-1.5">
        {config.isRunning ? (
          <Button variant="destructive" size="sm" className="h-7 text-xs px-2" onClick={onStop}>
            Stop
          </Button>
        ) : (
          <>
            <Button size="sm" className="h-7 text-xs px-2" onClick={onStart}>
              Start
            </Button>
            {config.hasMessages && onReset && (
              <Button variant="ghost" size="sm" className="h-7 text-xs px-2" onClick={onReset}>
                Clear
              </Button>
            )}
          </>
        )}
      </div>
    </div>
  )
}