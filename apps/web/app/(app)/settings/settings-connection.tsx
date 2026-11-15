'use client'

/** API connection card with test/status, plus external provider URL. */
import { SettingsSlider, VersionBadge, WEB_VERSION } from './settings-ui'
import { SettingsExternalProviderCard } from '@/components/settings/SettingsExternalProviderCard'
import { PUBLIC_API_URL } from '@/lib/config'
import { settingsSchema } from '@/lib/validation-schemas'
import {
  Button,
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
  Input,
  cn,
} from '@sloughgpt/strui'
import type { SettingsWorkbench } from './useSettingsWorkbench'

export function SettingsConnection({ wb }: { wb: SettingsWorkbench }) {
  const {
    connectionTest,
    detailed,
    handleTestConnection,
    setSettingsErrors,
    settings,
    settingsErrors,
    updateSettings,
  } = wb

  return (
    <>
      {/* Connection */}
      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-base">Connection</CardTitle>
            <CardDescription>Service connection and authentication</CardDescription>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label htmlFor="settings-api-url" className="text-sm font-medium">
              API URL
            </label>
            <Input
              id="settings-api-url"
              value={settings.apiUrl}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                updateSettings({ apiUrl: e.target.value })
                if (settingsErrors.apiUrl)
                  setSettingsErrors((prev) => ({ ...prev, apiUrl: undefined }))
              }}
              onBlur={() => {
                if (settings.apiUrl) {
                  const result = settingsSchema.shape.apiUrl.safeParse(settings.apiUrl)
                  if (!result.success) {
                    setSettingsErrors((prev) => ({
                      ...prev,
                      apiUrl: result.error.issues[0]?.message,
                    }))
                  }
                }
              }}
              placeholder={PUBLIC_API_URL}
              className={cn(
                'font-mono text-xs',
                settingsErrors.apiUrl && 'border-destructive ring-destructive/20',
              )}
              aria-label="Service URL"
              aria-invalid={!!settingsErrors.apiUrl}
              aria-describedby={settingsErrors.apiUrl ? 'apiurl-error' : undefined}
            />
            {settingsErrors.apiUrl && (
              <p id="apiurl-error" className="text-xs text-destructive" role="alert">
                {settingsErrors.apiUrl}
              </p>
            )}
            <p className="text-[11px] text-muted-foreground">
              Service address. Changes take effect on next request.
            </p>
          </div>
          <div className="space-y-2">
            <label htmlFor="settings-hf-token" className="text-sm font-medium">
              HuggingFace Token
            </label>
            <Input
              id="settings-hf-token"
              type="password"
              value={settings.hfToken}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                updateSettings({ hfToken: e.target.value })
              }
              placeholder="hf_..."
              className="font-mono text-xs"
              aria-label="HuggingFace API token"
            />
            <p className="text-xs text-muted-foreground">
              Required for loading private HuggingFace models. Stored locally in your browser.
            </p>
          </div>
          <div className="flex items-center gap-2 pt-2">
            <Button
              size="sm"
              variant="outline"
              className="h-9 text-xs"
              onClick={handleTestConnection}
              disabled={connectionTest.status === 'testing'}
            >
              {connectionTest.status === 'testing' ? 'Testing...' : 'Test connection'}
            </Button>
            {connectionTest.status === 'ok' && (
              <span className="text-xs text-success font-medium">
                Connected ({connectionTest.latency}ms)
              </span>
            )}
            {connectionTest.status === 'error' && (
              <span className="text-xs text-destructive font-medium">{connectionTest.error}</span>
            )}
          </div>
        </CardContent>
        <CardFooter className="justify-end">
          <VersionBadge label="v" version={detailed?.versions?.api} />
        </CardFooter>
      </Card>

      {/* Connection banner — the one banner that shows when the backend is
          unreachable, so its timings are LOCAL settings (readable offline) */}
      <Card data-testid="settings-connection-banner">
        <CardHeader>
          <div>
            <CardTitle className="text-base">Connection banner</CardTitle>
            <CardDescription>Backend drop and reconnection notice timing</CardDescription>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <SettingsSlider
            label="Drop notice delay"
            value={settings.bannerGraceMs ?? 5_000}
            onChange={(v) => updateSettings({ bannerGraceMs: v })}
            min={0}
            max={30_000}
            step={500}
            formatValue={(v) => (v === 0 ? 'Immediate' : `${(v / 1000).toFixed(1)}s`)}
          />
          <SettingsSlider
            label="Reconnected confirmation"
            value={settings.bannerHoldMs ?? 4_000}
            onChange={(v) => updateSettings({ bannerHoldMs: v })}
            min={0}
            max={30_000}
            step={500}
            formatValue={(v) => (v === 0 ? 'Off' : `${(v / 1000).toFixed(1)}s`)}
          />
          <p className="text-[11px] text-muted-foreground">
            The red “Backend unreachable” notice waits this long before appearing (blips are never
            shown), then the green confirmation stays this long after the connection recovers.
            Stored locally so they work while the service is down.
          </p>
        </CardContent>
        <CardFooter className="justify-end">
          <VersionBadge label="v" version={WEB_VERSION} />
        </CardFooter>
      </Card>

      {/* External model provider (OpenRouter / OpenAI-compatible) — separate from service API URL + HF token */}
      <SettingsExternalProviderCard version={WEB_VERSION} />
    </>
  )
}
