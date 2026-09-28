'use client'

import { useCallback, useEffect, useState } from 'react'
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
  Button,
  Input,
  Switch,
  Badge,
  Skeleton,
} from '@sloughgpt/strui'
import {
  settingsController,
  type ProviderApiSettings,
  type ProviderApiSettingsUpdate,
} from '@/lib/settings-controller'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'

interface SettingsExternalProviderCardProps {
  version?: string
}

export function SettingsExternalProviderCard({ version }: SettingsExternalProviderCardProps) {
  const addToast = useToastStore((s) => s.addToast)
  const [loaded, setLoaded] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [enabled, setEnabled] = useState(false)
  const [apiUrl, setApiUrl] = useState('')
  const [model, setModel] = useState('gpt-4o-mini')
  const [apiKey, setApiKey] = useState('')
  const [apiKeySet, setApiKeySet] = useState(false)
  const [urlError, setUrlError] = useState<string | undefined>()
  const [saving, setSaving] = useState(false)
  const [registered, setRegistered] = useState<boolean | null>(null)

  const applyProvider = useCallback((p: ProviderApiSettings) => {
    setEnabled(p.enabled)
    setApiUrl(p.api_url)
    setModel(p.model)
    setApiKeySet(p.api_key_set)
    setApiKey('')
    setRegistered(p.registered ?? null)
    setLoaded(true)
  }, [])

  useEffect(() => {
    let cancelled = false
    settingsController
      .getProviderApi()
      .then((p) => {
        if (!cancelled) applyProvider(p)
      })
      .catch((e) => {
        if (!cancelled) {
          setLoadError(extractErrorMessage(e, 'Could not load provider settings'))
          setLoaded(true)
        }
      })
    return () => {
      cancelled = true
    }
  }, [applyProvider])

  const handleSave = async () => {
    setUrlError(undefined)
    const trimmedUrl = apiUrl.trim()
    if (enabled && !/^https?:\/\/.+/.test(trimmedUrl)) {
      setUrlError('Endpoint must be an http(s) URL (e.g. https://openrouter.ai/api/v1)')
      return
    }
    if (enabled && !apiKey && !apiKeySet) {
      addToast('API key required before enabling the external provider', 'error')
      return
    }

    setSaving(true)
    const body: ProviderApiSettingsUpdate = {
      enabled,
      api_url: trimmedUrl,
      model: model.trim() || 'gpt-4o-mini',
    }
    if (apiKey) body.api_key = apiKey

    try {
      const p = await settingsController.updateProviderApi(body)
      applyProvider(p)
      setRegistered(p.registered ?? null)
      if (enabled && p.enabled && p.registered === false) {
        addToast('Saved — provider not registered (check API key)', 'info')
      } else {
        addToast('External provider settings saved', 'success')
      }
    } catch (e: unknown) {
      addToast(extractErrorMessage(e, 'Could not save provider settings'), 'error')
    } finally {
      setSaving(false)
    }
  }

  const handleClearKey = async () => {
    setSaving(true)
    try {
      const p = await settingsController.updateProviderApi({ api_key: '' })
      applyProvider(p)
      setApiKeySet(false)
      addToast('API key cleared', 'success')
    } catch (e: unknown) {
      addToast(extractErrorMessage(e, 'Could not clear API key'), 'error')
    } finally {
      setSaving(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <div>
          <CardTitle className="text-base">External model provider</CardTitle>
          <CardDescription>
            OpenRouter or any OpenAI-compatible endpoint — separate from the service API URL
          </CardDescription>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {!loaded && (
          <div className="space-y-3" aria-busy="true" aria-label="Loading provider settings">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
          </div>
        )}
        {loadError && (
          <p className="text-xs text-destructive" role="alert">
            {loadError}
          </p>
        )}
        {loaded && (
          <>
            <div className="flex items-center justify-between gap-3">
              <div className="space-y-0.5">
                <label htmlFor="settings-provider-enabled" className="text-sm font-medium">
                  Use external provider
                </label>
                <p className="text-[11px] text-muted-foreground">
                  Route chat through the external endpoint when registered.
                </p>
              </div>
              <Switch
                id="settings-provider-enabled"
                checked={enabled}
                onCheckedChange={setEnabled}
                aria-label="Use external provider"
              />
            </div>
            <div className="space-y-2">
              <label htmlFor="settings-provider-url" className="text-sm font-medium">
                Endpoint URL
              </label>
              <Input
                id="settings-provider-url"
                value={apiUrl}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                  setApiUrl(e.target.value)
                  if (urlError) setUrlError(undefined)
                }}
                placeholder="https://openrouter.ai/api/v1"
                className="font-mono text-xs"
                aria-label="External provider endpoint URL"
                aria-invalid={!!urlError}
                aria-describedby={urlError ? 'providerurl-error' : 'providerurl-help'}
              />
              {urlError && (
                <p id="providerurl-error" className="text-xs text-destructive" role="alert">
                  {urlError}
                </p>
              )}
              <p id="providerurl-help" className="text-[11px] text-muted-foreground">
                Base URL for /chat/completions. Not the Man service address above.
              </p>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <label htmlFor="settings-provider-key" className="text-sm font-medium">
                  API key
                </label>
                {apiKeySet && (
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 px-2 text-[11px]"
                    onClick={handleClearKey}
                    disabled={saving}
                  >
                    Clear key
                  </Button>
                )}
              </div>
              <Input
                id="settings-provider-key"
                type="password"
                value={apiKey}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setApiKey(e.target.value)}
                placeholder={apiKeySet ? '•••••••• (saved)' : 'sk-...'}
                className="font-mono text-xs"
                aria-label="External provider API key"
                autoComplete="off"
              />
              <p className="text-[11px] text-muted-foreground">
                {apiKeySet
                  ? 'Key stored server-side — never shown again. Leave blank to keep it.'
                  : 'Required to enable the provider. Stored server-side; never returned to the browser.'}
              </p>
            </div>
            <div className="space-y-2">
              <label htmlFor="settings-provider-model" className="text-sm font-medium">
                Model id
              </label>
              <Input
                id="settings-provider-model"
                value={model}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setModel(e.target.value)}
                placeholder="openai/gpt-4o-mini"
                className="font-mono text-xs"
                aria-label="External provider model id"
              />
              <p className="text-[11px] text-muted-foreground">
                Model identifier sent to the endpoint.
              </p>
            </div>
            <div className="flex items-center gap-2 pt-2">
              <Button size="sm" className="h-9 text-xs" onClick={handleSave} disabled={saving}>
                {saving ? 'Saving...' : 'Save provider'}
              </Button>
              {registered === true && (
                <span className="text-xs text-success font-medium">Registered</span>
              )}
              {enabled && registered === false && (
                <span className="text-xs text-warning font-medium">Not registered</span>
              )}
            </div>
          </>
        )}
      </CardContent>
      <CardFooter className="justify-end">
        {version && (
          <Badge
            variant="outline"
            className="text-[10px] font-mono px-1.5 py-0 h-5 text-muted-foreground border-border/40"
          >
            v {version}
          </Badge>
        )}
      </CardFooter>
    </Card>
  )
}
