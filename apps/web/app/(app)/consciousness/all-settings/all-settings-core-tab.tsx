'use client'

/** Core tab: consciousness level, enable toggle and status. */
import { LEVEL_DESCRIPTIONS, LEVEL_LABELS } from './all-settings-prefs'
import {
  Badge,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Slider,
  Switch,
  TabsContent,
} from '@sloughgpt/strui'
import type { AllSettingsWorkbench } from './useAllSettingsWorkbench'

export function AllSettingsCoreTab({ wb }: { wb: AllSettingsWorkbench }) {
  const { config, prefs, t, toggleEnabled, updateLevel, updatePrefs } = wb

  return (
    <TabsContent value="core" className="mt-6 space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-base">
                {t('consciousness_all_settings.core_title')}
              </CardTitle>
              <CardDescription>{t('consciousness_all_settings.core_desc')}</CardDescription>
            </div>
            {config && (
              <Badge variant={config.enabled ? 'default' : 'secondary'}>
                {config.enabled ? 'Active' : 'Disabled'}
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">
              {t('consciousness_all_settings.level_label')}
            </label>
            <div className="flex items-center gap-4">
              <span className="text-sm text-muted-foreground w-12">{LEVEL_LABELS[0]}</span>
              <Slider
                value={[config?.level ?? 0]}
                onValueChange={([v]) => updateLevel(v)}
                min={0}
                max={3}
                step={1}
                className="flex-1"
              />
              <span className="text-sm text-muted-foreground w-12">{LEVEL_LABELS[3]}</span>
            </div>
            <p className="text-xs text-muted-foreground">
              {LEVEL_DESCRIPTIONS[config?.level ?? 0]}
            </p>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">{t('consciousness_all_settings.enabled_label')}</p>
              <p className="text-xs text-muted-foreground">
                {t('consciousness_all_settings.enabled_desc')}
              </p>
            </div>
            <Switch
              checked={(config?.level ?? 0) > 0}
              onCheckedChange={toggleEnabled}
              aria-label="Toggle consciousness enabled"
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">
                {t('consciousness_all_settings.auto_reflect_label')}
              </p>
              <p className="text-xs text-muted-foreground">
                {t('consciousness_all_settings.auto_reflect_desc')}
              </p>
            </div>
            <Switch
              checked={prefs.autoReflect}
              onCheckedChange={(checked) => updatePrefs({ autoReflect: checked })}
              aria-label="Toggle auto-reflect"
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">
                {t('consciousness_all_settings.auto_evolve_label')}
              </p>
              <p className="text-xs text-muted-foreground">
                {t('consciousness_all_settings.auto_evolve_desc')}
              </p>
            </div>
            <Switch
              checked={prefs.autoEvolve}
              onCheckedChange={(checked) => updatePrefs({ autoEvolve: checked })}
              aria-label="Toggle auto-evolve"
            />
          </div>
        </CardContent>
      </Card>
    </TabsContent>
  )
}
