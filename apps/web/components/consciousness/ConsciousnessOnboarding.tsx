'use client'

import { useState, useCallback } from 'react'
import { Button, Card, CardContent, CardHeader, CardTitle, Badge } from '@sloughgpt/strui'
import { useLocale } from '@/hooks/useLocale'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { consciousnessController } from '@/lib/consciousness-controller'

const TOTAL_STEPS = 5

const LEVEL_OPTIONS = [
  { value: 0, label: 'Off', description: 'Consciousness system disabled' },
  { value: 1, label: 'Basic', description: 'Basic qualia and self-reflection' },
  { value: 2, label: 'Full', description: 'Full self-model with belief tracking' },
  { value: 3, label: 'Deep', description: 'Deep reflection with narrative generation' },
]

const PERSONALITY_PRESETS = [
  { value: 'default', label: 'Default' },
  { value: 'formal', label: 'Formal' },
  { value: 'creative', label: 'Creative' },
  { value: 'analyst', label: 'Analyst' },
  { value: 'empathetic', label: 'Empathetic' },
  { value: 'minimal', label: 'Minimal' },
]

interface ConsciousnessOnboardingProps {
  onComplete: () => void
  onDismiss: () => void
}

export function ConsciousnessOnboarding({ onComplete, onDismiss }: ConsciousnessOnboardingProps) {
  const addToast = useToastStore(s => s.addToast)
  const { t } = useLocale()
  const [step, setStep] = useState(1)
  const [selectedLevel, setSelectedLevel] = useState(2)
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null)
  const [seeding, setSeeding] = useState(false)
  const [seedDone, setSeedDone] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  const handleNext = useCallback(async () => {
    if (step === 2) {
      setSubmitting(true)
      try {
        await consciousnessController.updateConfig({ level: selectedLevel })
        addToast(t('consciousness_onboarding.toast_level_set'), 'success')
      } catch (e) {
        addToast(extractErrorMessage(e), 'error')
        setSubmitting(false)
        return
      }
      setSubmitting(false)
    }

    if (step === 3 && selectedPreset) {
      setSubmitting(true)
      try {
        await consciousnessController.applyPersonalityPreset(selectedPreset)
        addToast(t('consciousness_onboarding.toast_preset_applied'), 'success')
      } catch (e) {
        addToast(extractErrorMessage(e), 'error')
        setSubmitting(false)
        return
      }
      setSubmitting(false)
    }

    if (step < TOTAL_STEPS) {
      setStep(s => s + 1)
    }
  }, [step, selectedLevel, selectedPreset, addToast, t])

  const handleSeed = async () => {
    setSeeding(true)
    try {
      await consciousnessController.seedData({ count: 10 })
      setSeedDone(true)
      addToast(t('consciousness_onboarding.toast_seeded'), 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setSeeding(false)
    }
  }

  const handleComplete = () => {
    onComplete()
  }

  const progress = Math.round((step / TOTAL_STEPS) * 100)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm">
      <Card className="w-full max-w-lg mx-4 overflow-hidden">
        <CardHeader className="space-y-3 pb-4">
          <div className="flex items-center justify-between">
            <Badge variant="secondary" className="text-[9px]">
              {step}/{TOTAL_STEPS}
            </Badge>
            <button
              type="button"
              onClick={onDismiss}
              className="text-[10px] text-muted-foreground hover:text-foreground transition-colors"
            >
              {t('consciousness_onboarding.skip')}
            </button>
          </div>
          <div className="w-full bg-muted rounded-full h-1">
            <div
              className="bg-primary h-1 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {step === 1 && (
            <div className="space-y-3">
              <CardTitle className="text-sm">{t('consciousness_onboarding.step1_title')}</CardTitle>
              <div className="space-y-2 text-xs text-muted-foreground leading-relaxed">
                <p>{t('consciousness_onboarding.step1_p1')}</p>
                <p>{t('consciousness_onboarding.step1_p2')}</p>
                <p>{t('consciousness_onboarding.step1_p3')}</p>
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="space-y-3">
              <CardTitle className="text-sm">{t('consciousness_onboarding.step2_title')}</CardTitle>
              <p className="text-xs text-muted-foreground">{t('consciousness_onboarding.step2_desc')}</p>
              <div className="grid gap-2">
                {LEVEL_OPTIONS.map(opt => (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => setSelectedLevel(opt.value)}
                    className={`text-left rounded-lg border p-2.5 transition-colors ${
                      selectedLevel === opt.value
                        ? 'border-primary bg-primary/5'
                        : 'border-border/40 hover:border-border'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium">{opt.label}</span>
                      {selectedLevel === opt.value && (
                        <span className="text-[9px] text-primary">{t('consciousness_onboarding.selected')}</span>
                      )}
                    </div>
                    <p className="text-[10px] text-muted-foreground mt-0.5">{opt.description}</p>
                  </button>
                ))}
              </div>
            </div>
          )}

          {step === 3 && (
            <div className="space-y-3">
              <CardTitle className="text-sm">{t('consciousness_onboarding.step3_title')}</CardTitle>
              <p className="text-xs text-muted-foreground">{t('consciousness_onboarding.step3_desc')}</p>
              <div className="grid grid-cols-2 gap-2">
                {PERSONALITY_PRESETS.map(preset => (
                  <button
                    key={preset.value}
                    type="button"
                    onClick={() => setSelectedPreset(preset.value)}
                    className={`text-left rounded-lg border p-2 transition-colors ${
                      selectedPreset === preset.value
                        ? 'border-primary bg-primary/5'
                        : 'border-border/40 hover:border-border'
                    }`}
                  >
                    <span className="text-xs font-medium">{preset.label}</span>
                  </button>
                ))}
              </div>
              <button
                type="button"
                onClick={() => setSelectedPreset(null)}
                className="text-[10px] text-muted-foreground hover:text-foreground transition-colors"
              >
                {t('consciousness_onboarding.step3_skip')}
              </button>
            </div>
          )}

          {step === 4 && (
            <div className="space-y-3">
              <CardTitle className="text-sm">{t('consciousness_onboarding.step4_title')}</CardTitle>
              <p className="text-xs text-muted-foreground">{t('consciousness_onboarding.step4_desc')}</p>
              <div className="rounded-lg border border-border/40 p-3 text-center">
                {seedDone ? (
                  <p className="text-xs text-success font-medium">{t('consciousness_onboarding.step4_done')}</p>
                ) : (
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={handleSeed}
                    disabled={seeding}
                    className="text-xs"
                  >
                    {seeding ? t('consciousness_onboarding.step4_seeding') : t('consciousness_onboarding.step4_seed')}
                  </Button>
                )}
              </div>
              <button
                type="button"
                onClick={() => setStep(s => s + 1)}
                className="text-[10px] text-muted-foreground hover:text-foreground transition-colors"
              >
                {t('consciousness_onboarding.step4_skip')}
              </button>
            </div>
          )}

          {step === 5 && (
            <div className="space-y-3 text-center">
              <CardTitle className="text-sm">{t('consciousness_onboarding.step5_title')}</CardTitle>
              <p className="text-xs text-muted-foreground">{t('consciousness_onboarding.step5_desc')}</p>
              <div className="space-y-1 text-[10px] text-muted-foreground/80">
                <p>{t('consciousness_onboarding.step5_level')}: {LEVEL_OPTIONS[selectedLevel].label}</p>
                {selectedPreset && <p>{t('consciousness_onboarding.step5_preset')}: {selectedPreset}</p>}
                {seedDone && <p>{t('consciousness_onboarding.step5_seeded')}: 10</p>}
              </div>
            </div>
          )}

          <div className="flex items-center justify-between pt-2 border-t border-border/30">
            {step > 1 ? (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setStep(s => s - 1)}
                className="text-xs h-7"
              >
                {t('consciousness_onboarding.back')}
              </Button>
            ) : (
              <div />
            )}
            {step < TOTAL_STEPS ? (
              <Button
                size="sm"
                onClick={handleNext}
                disabled={submitting}
                className="text-xs h-7"
              >
                {submitting ? '...' : t('consciousness_onboarding.next')}
              </Button>
            ) : (
              <Button
                size="sm"
                onClick={handleComplete}
                className="text-xs h-7"
              >
                {t('consciousness_onboarding.finish')}
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
