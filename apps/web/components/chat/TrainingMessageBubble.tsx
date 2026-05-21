'use client'

import { cn } from '@/lib/cn'
import { useState } from 'react'

export type TrainingRole = 'teacher' | 'baby' | 'correction' | 'system'

export interface TrainingMessage {
  id: string
  role: TrainingRole
  content: string
  step?: number
  timestamp?: Date
  personality?: {
    formality: number
    detail_level: number
    certainty: number
  }
}

interface TrainingMessageBubbleProps {
  message: TrainingMessage
  className?: string
}

const roleConfig = {
  teacher: { label: 'Teacher', bg: 'bg-blue-50 dark:bg-blue-950', border: 'border-blue-200 dark:border-blue-800' },
  baby: { label: 'Baby', bg: 'bg-green-50 dark:bg-green-950', border: 'border-green-200 dark:border-green-800' },
  correction: { label: 'Correction', bg: 'bg-purple-50 dark:bg-purple-950', border: 'border-purple-200 dark:border-purple-800' },
  system: { label: 'System', bg: 'bg-muted', border: 'border-border' },
}

export function TrainingMessageBubble({ message, className }: TrainingMessageBubbleProps) {
  const config = roleConfig[message.role] || roleConfig.system
  
  return (
    <div className={cn(
      "p-2 sm:p-2.5 rounded-lg border text-xs transition-all duration-300",
      config.bg,
      config.border,
      className
    )}>
      <div className="text-xs font-medium mb-0.5 text-muted-foreground">
        {config.label}
      </div>
      <div className="text-xs whitespace-pre-wrap break-words leading-tight">
        {message.content}
      </div>
    </div>
  )
}

export function TrainingTurn({ messages }: { messages: TrainingMessage[] }) {
  const [showCorrection, setShowCorrection] = useState(false)
  const teacherMsg = messages.find(m => m.role === 'teacher')
  const babyMsg = messages.find(m => m.role === 'baby')
  const correctionMsg = messages.find(m => m.role === 'correction')
  
  if (!teacherMsg) return null
  
  const stepNum = Math.ceil((teacherMsg.step || 0) / 3)
  
  return (
    <div 
      className="space-y-1.5 sm:space-y-2 pb-2 sm:pb-3 border-b border-gray-200 dark:border-gray-700 mb-2 sm:mb-3"
      onMouseEnter={() => setShowCorrection(true)}
      onMouseLeave={() => setShowCorrection(false)}
    >
      <div className="text-xs font-bold text-muted-foreground flex items-center gap-1.5 sm:gap-2">
        <span>T{stepNum}</span>
        {correctionMsg && (
          <span className={cn(
            "text-xs px-1 py-0.5 rounded transition-opacity",
            showCorrection ? "opacity-100 bg-purple-100 dark:bg-purple-900" : "opacity-50"
          )}>
            {showCorrection ? correctionMsg.content : "Corr"}
          </span>
        )}
      </div>
      {teacherMsg && <TrainingMessageBubble message={teacherMsg} />}
      {babyMsg && <TrainingMessageBubble message={babyMsg} />}
    </div>
  )
}