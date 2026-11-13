'use client'

/**
 * All Settings — composition root.
 *
 * State, IO and the localStorage prefs persistence live in
 * `useAllSettingsWorkbench`; the six tabs own render. Only the page shell
 * belongs here: the loading skeleton and the Tabs wiring.
 */
import { useState } from 'react'
import { PageContainer } from '@/components/PageContainer'
import {
  Card,
  CardHeader,
  CardContent,
  Skeleton,
  Tabs,
  TabsList,
  TabsTrigger,
} from '@sloughgpt/strui'
import { useAllSettingsWorkbench } from './useAllSettingsWorkbench'
import { AllSettingsCoreTab } from './all-settings-core-tab'
import { AllSettingsDisplayTab } from './all-settings-display-tab'
import { AllSettingsPersonalityTab } from './all-settings-personality-tab'
import { AllSettingsDataTab } from './all-settings-data-tab'
import { AllSettingsNotificationsTab } from './all-settings-notifications-tab'
import { AllSettingsExportTab } from './all-settings-export-tab'

export default function AllSettingsPage() {
  const wb = useAllSettingsWorkbench()
  const { loading, t } = wb
  const [activeTab, setActiveTab] = useState('core')

  if (loading) {
    return (
      <PageContainer title={t('consciousness_all_settings.page_title')}>
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-5 w-40" />
                <Skeleton className="h-3 w-64" />
              </CardHeader>
              <CardContent className="space-y-3">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
              </CardContent>
            </Card>
          ))}
        </div>
      </PageContainer>
    )
  }

  return (
    <PageContainer title={t('consciousness_all_settings.page_title')}>
      <div className="p-6">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            <TabsTrigger value="core">{t('consciousness_all_settings.tab_core')}</TabsTrigger>
            <TabsTrigger value="display">{t('consciousness_all_settings.tab_display')}</TabsTrigger>
            <TabsTrigger value="personality">
              {t('consciousness_all_settings.tab_personality')}
            </TabsTrigger>
            <TabsTrigger value="data">{t('consciousness_all_settings.tab_data')}</TabsTrigger>
            <TabsTrigger value="notifications">
              {t('consciousness_all_settings.tab_notifications')}
            </TabsTrigger>
            <TabsTrigger value="export">{t('consciousness_all_settings.tab_export')}</TabsTrigger>
          </TabsList>

          <AllSettingsCoreTab wb={wb} />
          <AllSettingsDisplayTab wb={wb} />
          <AllSettingsPersonalityTab wb={wb} />
          <AllSettingsDataTab wb={wb} />
          <AllSettingsNotificationsTab wb={wb} />
          <AllSettingsExportTab wb={wb} />
        </Tabs>
      </div>
    </PageContainer>
  )
}
