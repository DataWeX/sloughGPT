'use client'

/** Export tab: settings file export and import. */
import {
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  TabsContent,
} from '@sloughgpt/strui'
import type { AllSettingsWorkbench } from './useAllSettingsWorkbench'

export function AllSettingsExportTab({ wb }: { wb: AllSettingsWorkbench }) {
  const { fileInputRef, handleExport, handleImport, t } = wb

  return (
    <TabsContent value="export" className="mt-6 space-y-6">
      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-base">
              {t('consciousness_all_settings.export_import_title')}
            </CardTitle>
            <CardDescription>{t('consciousness_all_settings.export_import_desc')}</CardDescription>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">{t('consciousness_all_settings.export_label')}</p>
              <p className="text-xs text-muted-foreground">
                {t('consciousness_all_settings.export_desc')}
              </p>
            </div>
            <Button size="sm" variant="secondary" onClick={handleExport}>
              {t('consciousness_all_settings.export_button')}
            </Button>
          </div>
          <div className="border-t border-border/30" />
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">{t('consciousness_all_settings.import_label')}</p>
              <p className="text-xs text-muted-foreground">
                {t('consciousness_all_settings.import_desc')}
              </p>
            </div>
            <div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".json"
                onChange={handleImport}
                className="hidden"
                aria-label="Import settings file"
              />
              <Button size="sm" variant="secondary" onClick={() => fileInputRef.current?.click()}>
                {t('consciousness_all_settings.import_button')}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </TabsContent>
  )
}
