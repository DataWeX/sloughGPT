'use client'

/** Process isolation guard and serving profiles. */
import { VersionBadge } from './settings-ui'
import { ServingProfilesCard } from '@/components/ServingProfilesCard'
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
  Skeleton,
  Switch,
} from '@sloughgpt/strui'
import type { SettingsWorkbench } from './useSettingsWorkbench'

export function SettingsProcess({ wb }: { wb: SettingsWorkbench }) {
  const { detailed, handleToggleProcessGuard, processGuard } = wb

  return (
    <>
      {/* Process isolation */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-base">Process isolation</CardTitle>
              <CardDescription>
                Run model inference in a separate process for crash safety
              </CardDescription>
            </div>
            {processGuard && (
              <Switch
                checked={processGuard.enabled}
                onCheckedChange={handleToggleProcessGuard}
                aria-label="Toggle process isolation"
              />
            )}
          </div>
        </CardHeader>
        <CardContent>
          {processGuard ? (
            <div className="space-y-2 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Status</span>
                <span className={processGuard.active ? 'text-success' : 'text-muted-foreground'}>
                  {processGuard.active
                    ? 'Active'
                    : processGuard.enabled
                      ? 'Enabled (not running)'
                      : 'Disabled'}
                </span>
              </div>
              {processGuard.model_id && (
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Model</span>
                  <span className="font-mono text-xs">{processGuard.model_id}</span>
                </div>
              )}
              {processGuard.health && (
                <>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Alive</span>
                    <span
                      className={processGuard.health.alive ? 'text-success' : 'text-destructive'}
                    >
                      {processGuard.health.alive ? 'Yes' : 'No'}
                    </span>
                  </div>
                  {processGuard.health.memory_mb !== undefined && (
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Memory</span>
                      <span className="font-mono text-xs">{processGuard.health.memory_mb} MB</span>
                    </div>
                  )}
                  {processGuard.health.restarts !== undefined && (
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Restarts</span>
                      <span className="font-mono text-xs">{processGuard.health.restarts}</span>
                    </div>
                  )}
                </>
              )}
              <p className="text-xs text-muted-foreground pt-1">
                Process isolation runs the model in a subprocess. If it crashes, the app survives
                and restarts it automatically. Uses ~2 GB more RAM.
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              <Skeleton className="h-4 w-40" />
              <Skeleton className="h-4 w-64" />
              <Skeleton className="h-3 w-full" />
            </div>
          )}
        </CardContent>
        <CardFooter className="justify-end">
          <VersionBadge label="v" version={detailed?.versions?.package} />
        </CardFooter>
      </Card>

      {/* Serving profiles */}
      <ServingProfilesCard />
    </>
  )
}
