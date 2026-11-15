'use client'

/** Notifications tab: browser/sound alerts and thresholds. */
import {
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

export function AllSettingsNotificationsTab({ wb }: { wb: AllSettingsWorkbench }) {
  const { prefs, t, updatePrefs } = wb

  return (
    <TabsContent value="notifications" className="mt-6 space-y-6">
      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-base">
              {t('consciousness_all_settings.notifications_title')}
            </CardTitle>
            <CardDescription>{t('consciousness_all_settings.notifications_desc')}</CardDescription>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">
                {t('consciousness_all_settings.browser_notifications_label')}
              </p>
              <p className="text-xs text-muted-foreground">
                {t('consciousness_all_settings.browser_notifications_desc')}
              </p>
            </div>
            <Switch
              checked={prefs.browserNotifications}
              onCheckedChange={(checked) => updatePrefs({ browserNotifications: checked })}
              aria-label="Toggle browser notifications"
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">
                {t('consciousness_all_settings.sound_notifications_label')}
              </p>
              <p className="text-xs text-muted-foreground">
                {t('consciousness_all_settings.sound_notifications_desc')}
              </p>
            </div>
            <Switch
              checked={prefs.soundNotifications}
              onCheckedChange={(checked) => updatePrefs({ soundNotifications: checked })}
              aria-label="Toggle sound notifications"
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-base">
              {t('consciousness_all_settings.alert_thresholds_title')}
            </CardTitle>
            <CardDescription>
              {t('consciousness_all_settings.alert_thresholds_desc')}
            </CardDescription>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">
                {t('consciousness_all_settings.threshold_low')}
              </label>
              <span className="text-xs text-muted-foreground font-mono">
                {prefs.alertThresholdLow}
              </span>
            </div>
            <Slider
              value={[prefs.alertThresholdLow]}
              onValueChange={([v]) => updatePrefs({ alertThresholdLow: v })}
              min={0}
              max={100}
              step={5}
            />
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">
                {t('consciousness_all_settings.threshold_high')}
              </label>
              <span className="text-xs text-muted-foreground font-mono">
                {prefs.alertThresholdHigh}
              </span>
            </div>
            <Slider
              value={[prefs.alertThresholdHigh]}
              onValueChange={([v]) => updatePrefs({ alertThresholdHigh: v })}
              min={0}
              max={100}
              step={5}
            />
          </div>
        </CardContent>
      </Card>
    </TabsContent>
  )
}
