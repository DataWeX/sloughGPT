'use client'

/** Display tab: chat/status-bar visibility and refresh prefs. */
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Switch,
  TabsContent,
} from '@sloughgpt/strui'
import type { AllSettingsWorkbench } from './useAllSettingsWorkbench'

export function AllSettingsDisplayTab({ wb }: { wb: AllSettingsWorkbench }) {
  const { prefs, t, updatePrefs } = wb

  return (
    <TabsContent value="display" className="mt-6 space-y-6">
      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-base">
              {t('consciousness_all_settings.display_title')}
            </CardTitle>
            <CardDescription>{t('consciousness_all_settings.display_desc')}</CardDescription>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">
                {t('consciousness_all_settings.show_in_chat_label')}
              </p>
              <p className="text-xs text-muted-foreground">
                {t('consciousness_all_settings.show_in_chat_desc')}
              </p>
            </div>
            <Switch
              checked={prefs.showInChat}
              onCheckedChange={(checked) => updatePrefs({ showInChat: checked })}
              aria-label="Toggle show in chat"
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">
                {t('consciousness_all_settings.show_in_statusbar_label')}
              </p>
              <p className="text-xs text-muted-foreground">
                {t('consciousness_all_settings.show_in_statusbar_desc')}
              </p>
            </div>
            <Switch
              checked={prefs.showInStatusBar}
              onCheckedChange={(checked) => updatePrefs({ showInStatusBar: checked })}
              aria-label="Toggle show in status bar"
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">
                {t('consciousness_all_settings.realtime_label')}
              </p>
              <p className="text-xs text-muted-foreground">
                {t('consciousness_all_settings.realtime_desc')}
              </p>
            </div>
            <Switch
              checked={prefs.realTimeUpdates}
              onCheckedChange={(checked) => updatePrefs({ realTimeUpdates: checked })}
              aria-label="Toggle real-time updates"
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">
                {t('consciousness_all_settings.auto_refresh_label')}
              </p>
              <p className="text-xs text-muted-foreground">
                {t('consciousness_all_settings.auto_refresh_desc')}
              </p>
            </div>
            <Switch
              checked={prefs.autoRefresh}
              onCheckedChange={(checked) => updatePrefs({ autoRefresh: checked })}
              aria-label="Toggle auto-refresh"
            />
          </div>
        </CardContent>
      </Card>
    </TabsContent>
  )
}
