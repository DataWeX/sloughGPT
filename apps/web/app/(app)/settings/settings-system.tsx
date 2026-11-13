'use client'

/** System health, details, version inspector and backup/restore. */
import { VersionBadge, WEB_VERSION } from './settings-ui'
import { SettingsBackupRestoreCard } from '@/components/settings/SettingsBackupRestoreCard'
import { SettingsSystemHealthCard } from '@/components/settings/SettingsSystemHealthCard'
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
  STRUI_VERSION,
  VersionInspector,
} from '@sloughgpt/strui'
import type { SettingsWorkbench } from './useSettingsWorkbench'

export function SettingsSystem({ wb }: { wb: SettingsWorkbench }) {
  const {
    apiOk,
    detailed,
    disk,
    fetchHealth,
    healthError,
    info,
    metrics,
    modelLoaded,
    modelType,
    settings,
    updateSettings,
  } = wb

  return (
    <>
      {/* System health */}
      <SettingsSystemHealthCard
        apiOk={apiOk}
        modelLoaded={modelLoaded}
        modelType={modelType}
        detailed={detailed}
        metrics={metrics}
        disk={disk}
        info={info}
        healthError={healthError}
        onRefresh={fetchHealth}
      />

      {/* System info */}
      {info && (
        <Card>
          <CardHeader>
            <div>
              <CardTitle className="text-base">System information</CardTitle>
              <CardDescription>Detailed platform and environment details</CardDescription>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">
                  Platform
                </p>
                <p className="text-sm font-medium mt-0.5">
                  {info.platform} {info.platform_release}
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">
                  Architecture
                </p>
                <p className="text-sm font-medium mt-0.5">{info.architecture}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">
                  Processor
                </p>
                <p className="text-sm font-medium mt-0.5 truncate">{info.processor}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">
                  CPU cores
                </p>
                <p className="text-sm font-medium mt-0.5">{info.cpu_count}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">
                  Platform version
                </p>
                <p className="text-sm font-medium mt-0.5 font-mono">{info.platform_version}</p>
              </div>
            </div>
          </CardContent>
          <CardFooter className="justify-end">
            <VersionBadge label="torch" version={detailed?.versions?.torch} />
          </CardFooter>
        </Card>
      )}

      {/* Version Inspector */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Version inspector</CardTitle>
          <CardDescription>Strui component ↔ backend feature version alignment</CardDescription>
        </CardHeader>
        <CardContent>
          <VersionInspector backendFeatures={detailed?.versions?.features} />
          <div className="flex items-center justify-between mt-3 pt-3 border-t border-border/40">
            <p className="text-xs text-muted-foreground">Component ↔ feature version sync</p>
            <span className="text-xs text-muted-foreground font-mono">strui {STRUI_VERSION}</span>
          </div>
        </CardContent>
      </Card>

      {/* Backup & restore */}
      <SettingsBackupRestoreCard
        settings={settings as unknown as Record<string, unknown>}
        onImport={(valid) => updateSettings(valid)}
        version={WEB_VERSION}
      />
    </>
  )
}
