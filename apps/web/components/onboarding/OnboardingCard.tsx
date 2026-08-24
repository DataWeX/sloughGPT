'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button } from '@sloughgpt/strui'

const ONBOARDING_KEY = 'sloughgpt-onboarded'

const STEPS = [
  {
    title: 'Welcome to SloughGPT',
    description: 'Your AI that grows with you. I learn from our conversations and remember what matters to you.',
    action: 'Get started',
  },
  {
    title: 'Talk to me',
    description: 'Start a conversation in the Chat tab. I will remember the things you tell me and use them in future chats.',
    action: 'Open chat',
  },
  {
    title: 'Tell me about yourself',
    description: 'The more you tell me, the better I can help. Add facts in the Knowledge tab — preferences, projects, anything you want me to remember.',
    action: 'Add knowledge',
  },
  {
    title: 'Shape my personality',
    description: 'In the Companion tab, you can choose how I behave — more curious, more creative, more warm. Or pick a personality preset.',
    action: 'Customize me',
  },
]

interface Props {
  onComplete: () => void
}

export function OnboardingCard({ onComplete }: Props) {
  const [step, setStep] = useState(0)

  useEffect(() => {
    const onboarded = localStorage.getItem(ONBOARDING_KEY)
    if (onboarded) onComplete()
  }, [onComplete])

  const handleComplete = () => {
    localStorage.setItem(ONBOARDING_KEY, 'true')
    onComplete()
  }

  const handleNext = () => {
    if (step < STEPS.length - 1) {
      setStep(step + 1)
    } else {
      handleComplete()
    }
  }

  const current = STEPS[step]

  return (
    <Card className="border-primary/20">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">{current.title}</CardTitle>
          <button
            type="button"
            onClick={handleComplete}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Skip
          </button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">{current.description}</p>

        {/* Progress dots */}
        <div className="flex gap-1.5">
          {STEPS.map((_, i) => (
            <div
              key={i}
              className={`h-1.5 rounded-full transition-all ${
                i === step ? 'bg-primary w-6' : i < step ? 'bg-primary/40 w-1.5' : 'bg-muted w-1.5'
              }`}
            />
          ))}
        </div>

        <Button size="sm" onClick={handleNext} className="w-full">
          {current.action}
        </Button>
      </CardContent>
    </Card>
  )
}
