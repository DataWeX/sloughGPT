'use client'

/** Personality tab: trait sliders, presets and reset. */
import { PRESETS } from './all-settings-prefs'
import {
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Slider,
  TabsContent,
} from '@sloughgpt/strui'
import type { AllSettingsWorkbench } from './useAllSettingsWorkbench'

export function AllSettingsPersonalityTab({ wb }: { wb: AllSettingsWorkbench }) {
  const { applyPreset, personality, t, updatePersonality } = wb

  return (
    <TabsContent value="personality" className="mt-6 space-y-6">
      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-base">
              {t('consciousness_all_settings.personality_summary_title')}
            </CardTitle>
            <CardDescription>
              {t('consciousness_all_settings.personality_summary_desc')}
            </CardDescription>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">{personality.summary}</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-base">
              {t('consciousness_all_settings.voice_title')}
            </CardTitle>
            <CardDescription>{t('consciousness_all_settings.voice_desc')}</CardDescription>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {(['warmth', 'confidence', 'humor', 'empathy'] as const).map((key) => (
            <div key={key} className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium capitalize">
                  {t(`consciousness_all_settings.voice_${key}`)}
                </label>
                <span className="text-xs text-muted-foreground font-mono">{personality[key]}</span>
              </div>
              <Slider
                value={[personality[key]]}
                onValueChange={([v]) => updatePersonality({ [key]: v })}
                min={0}
                max={100}
                step={1}
              />
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-base">
              {t('consciousness_all_settings.traits_title')}
            </CardTitle>
            <CardDescription>{t('consciousness_all_settings.traits_desc')}</CardDescription>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {(['openness', 'agreeableness', 'conscientiousness'] as const).map((key) => (
            <div key={key} className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium capitalize">
                  {t(`consciousness_all_settings.trait_${key}`)}
                </label>
                <span className="text-xs text-muted-foreground font-mono">{personality[key]}</span>
              </div>
              <Slider
                value={[personality[key]]}
                onValueChange={([v]) => updatePersonality({ [key]: v })}
                min={0}
                max={100}
                step={1}
              />
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-base">
              {t('consciousness_all_settings.presets_title')}
            </CardTitle>
            <CardDescription>{t('consciousness_all_settings.presets_desc')}</CardDescription>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {Object.keys(PRESETS).map((preset) => (
              <Button
                key={preset}
                size="sm"
                variant="secondary"
                onClick={() => applyPreset(preset)}
              >
                <span className="capitalize">
                  {t(`consciousness_all_settings.preset_${preset}`)}
                </span>
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>
    </TabsContent>
  )
}
