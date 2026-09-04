'use client'

import { useState, useEffect, useCallback } from 'react'
import { Button, Card, CardContent } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'

const STORAGE_KEY = 'sloughgpt-training-onboarding-done'

interface OnboardingStep {
  id: string
  title: string
  description: string
  visual: 'data' | 'configure' | 'train' | 'results'
}

const STEPS: OnboardingStep[] = [
  {
    id: 'welcome',
    title: 'Teach your AI with data',
    description: 'Training lets you teach your agent new knowledge from text, datasets, or conversations. No ML expertise required.',
    visual: 'data',
  },
  {
    id: 'pick-data',
    title: '1. Pick your data',
    description: 'Choose from imported datasets, paste text directly, or import from Kaggle/HuggingFace in one click. Conversation-format data (JSONL) trains best.',
    visual: 'data',
  },
  {
    id: 'configure',
    title: '2. Configure',
    description: 'Select a training method: train from scratch, continue training an existing model, or use the fast turbo mode for quick experiments.',
    visual: 'configure',
  },
  {
    id: 'train',
    title: '3. Train',
    description: 'Start training with one click. Watch live progress with loss curves, step counts, and ETA. Pause or stop anytime.',
    visual: 'train',
  },
  {
    id: 'results',
    title: '4. Results',
    description: 'Your trained model is saved as a checkpoint. Load it into the agent, test it, or compare against previous versions.',
    visual: 'results',
  },
]

function StepVisual({ type, isActive }: { type: OnboardingStep['visual']; isActive: boolean }) {
  const baseClass = 'w-16 h-16 rounded-xl flex items-center justify-center text-2xl transition-all duration-300'

  const visuals: Record<OnboardingStep['visual'], { bg: string; icon: string; label: string }> = {
    data: { bg: 'bg-primary/10', icon: '📊', label: 'Data' },
    configure: { bg: 'bg-accent/10', icon: '⚙️', label: 'Configure' },
    train: { bg: 'bg-success/10', icon: '🚀', label: 'Train' },
    results: { bg: 'bg-info/10', icon: '📈', label: 'Results' },
  }

  const v = visuals[type]

  return (
    <div
      className={cn(
        baseClass,
        v.bg,
        isActive && 'scale-110 shadow-lg'
      )}
      aria-hidden="true"
    >
      <span role="img" aria-label={v.label}>{v.icon}</span>
    </div>
  )
}

export function GuidedOnboarding({ onComplete }: { onComplete: () => void }) {
  const [currentStep, setCurrentStep] = useState(0)
  const [isExiting, setIsExiting] = useState(false)

  const handleComplete = useCallback(() => {
    setIsExiting(true)
    setTimeout(() => {
      try {
        localStorage.setItem(STORAGE_KEY, 'true')
      } catch {}
      onComplete()
    }, 300)
  }, [onComplete])

  const handleSkip = useCallback(() => {
    handleComplete()
  }, [handleComplete])

  const handleNext = useCallback(() => {
    if (currentStep < STEPS.length - 1) {
      setCurrentStep(prev => prev + 1)
    } else {
      handleComplete()
    }
  }, [currentStep, handleComplete])

  const handlePrev = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1)
    }
  }, [currentStep])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') handleSkip()
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        handleNext()
      }
      if (e.key === 'ArrowRight') handleNext()
      if (e.key === 'ArrowLeft') handlePrev()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleNext, handlePrev, handleSkip])

  const step = STEPS[currentStep]
  const progress = ((currentStep + 1) / STEPS.length) * 100

  return (
    <div
      className={cn(
        'fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm',
        'transition-opacity duration-300',
        isExiting ? 'opacity-0' : 'opacity-100'
      )}
      role="dialog"
      aria-modal="true"
      aria-label="Training onboarding"
    >
      <Card className="w-full max-w-lg mx-4 shadow-xl border-border/50">
        <CardContent className="p-6 space-y-6">
          {/* Progress bar */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-[10px] text-muted-foreground">
              <span>Step {currentStep + 1} of {STEPS.length}</span>
              <button
                type="button"
                onClick={handleSkip}
                className="hover:text-foreground transition-colors"
              >
                Skip tour
              </button>
            </div>
            <div className="h-1 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-all duration-500 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>

          {/* Step content */}
          <div className="flex items-start gap-4">
            <StepVisual type={step.visual} isActive={true} />
            <div className="flex-1 space-y-2">
              <h2 className="text-base font-medium text-foreground">{step.title}</h2>
              <p className="text-sm text-muted-foreground leading-relaxed">{step.description}</p>
            </div>
          </div>

          {/* Step indicators */}
          <div className="flex items-center justify-center gap-2" aria-label="Onboarding progress">
            {STEPS.map((s, i) => (
              <div
                key={s.id}
                className={cn(
                  'w-2 h-2 rounded-full transition-all duration-300',
                  i === currentStep ? 'bg-primary scale-125' :
                  i < currentStep ? 'bg-primary/50' : 'bg-muted'
                )}
                aria-label={`Step ${i + 1}: ${s.title}`}
              />
            ))}
          </div>

          {/* Navigation */}
          <div className="flex items-center justify-between">
            <Button
              size="sm"
              variant="ghost"
              onClick={handlePrev}
              disabled={currentStep === 0}
            >
              Back
            </Button>
            <Button size="sm" onClick={handleNext}>
              {currentStep === STEPS.length - 1 ? 'Start training' : 'Next'}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export function shouldShowOnboarding(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) !== 'true'
  } catch {
    return true
  }
}
