'use client'

/** Appearance, palette and language cards. */
import { VersionBadge, WEB_VERSION } from './settings-ui'
import { LOCALES } from '@/hooks/useLocale'
import { PALETTE_IDS, PALETTE_LABELS, THEME_PREVIEWS, type ThemeMode } from '@/lib/theme-storage'
import {
  Button,
  cn,
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@sloughgpt/strui'
import type { SettingsWorkbench } from './useSettingsWorkbench'

export function SettingsAppearance({ wb }: { wb: SettingsWorkbench }) {
  const { locale, palette, setLocale, setPalette, updateSettings, mode, setMode } = wb

  /** Apply a canvas to `<html>` and mirror it into the persisted settings —
   *  `setMode` is the path that actually reaches the DOM, `updateSettings`
   *  keeps the backend-persisted preference in step with it. */
  const applyMode = (next: ThemeMode) => {
    setMode(next)
    updateSettings({ theme: next })
  }

  return (
    <>
      {/* Appearance */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-base">Appearance</CardTitle>
              <CardDescription>Theme preference</CardDescription>
            </div>
            {mode !== 'light' && (
              <Button
                size="sm"
                variant="ghost"
                className="h-8 text-xs text-muted-foreground"
                onClick={() => applyMode('light')}
              >
                Reset
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div role="radiogroup" aria-label="Theme" className="grid grid-cols-3 gap-2.5">
            {THEME_PREVIEWS.map((p) => {
              const selected = mode === p.id
              return (
                <button
                  key={p.id}
                  type="button"
                  role="radio"
                  aria-checked={selected}
                  onClick={() => applyMode(p.id)}
                  className={cn(
                    'relative flex flex-col gap-2 overflow-hidden rounded-lg border p-2 text-left transition-colors',
                    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-card',
                    selected
                      ? 'border-success bg-success/10'
                      : 'border-border/60 hover:border-border hover:bg-muted/40',
                  )}
                >
                  {/* Mini render of this canvas — colours come from
                      THEME_PREVIEWS (picker data), never live vars: only one
                      mode's tokens are live at a time. */}
                  <span
                    aria-hidden="true"
                    className="block h-16 overflow-hidden rounded-md border"
                    style={{ backgroundColor: p.bg, borderColor: p.border }}
                  >
                    <span
                      className="flex items-center gap-1 border-b px-1.5 py-1"
                      style={{ borderColor: p.border }}
                    >
                      <span
                        className="h-1.5 w-1.5 rounded-full"
                        style={{ backgroundColor: p.primary }}
                      />
                      <span
                        className="h-1.5 w-1.5 rounded-full"
                        style={{ backgroundColor: p.border }}
                      />
                      <span
                        className="ml-auto h-1.5 w-5 rounded-sm"
                        style={{ backgroundColor: p.border }}
                      />
                    </span>
                    <span
                      className="flex flex-col gap-1 px-1.5 pt-1"
                      style={{ backgroundColor: p.panel }}
                    >
                      <span className="h-1.5 w-4/5 rounded-sm" style={{ backgroundColor: p.fg }} />
                      <span
                        className="h-1.5 w-1/2 rounded-sm"
                        style={{ backgroundColor: p.border }}
                      />
                      <span
                        className="mt-0.5 inline-flex h-4 items-center self-start rounded-sm px-1.5 text-[11px] font-medium"
                        style={{ backgroundColor: p.primary, color: p.primaryFg }}
                      >
                        Button
                      </span>
                    </span>
                  </span>
                  <span className="text-xs font-medium text-foreground">{p.label}</span>
                  {selected && (
                    <span
                      aria-hidden="true"
                      className="absolute right-0 top-0 h-5 w-5 bg-success"
                      style={{ clipPath: 'polygon(100% 0, 0 0, 100% 100%)' }}
                    />
                  )}
                </button>
              )
            })}
          </div>
        </CardContent>
        <CardFooter className="justify-end">
          <VersionBadge label="v" version={WEB_VERSION} />
        </CardFooter>
      </Card>

      {/* Palette */}
      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-base">Palette</CardTitle>
            <CardDescription>Full-spectrum color palette</CardDescription>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {PALETTE_IDS.map((id) => (
              <Button
                key={id}
                size="sm"
                variant={palette === id ? 'default' : 'outline'}
                className="h-9 text-xs"
                onClick={() => setPalette(id)}
              >
                {PALETTE_LABELS[id]}
              </Button>
            ))}
          </div>
          <div className="mt-3 p-3 rounded-lg border border-border/40 bg-card">
            <div className="flex items-center gap-2 mb-2">
              <div className="h-3 w-3 rounded-full bg-primary" />
              <div className="h-2 w-16 rounded bg-muted" />
            </div>
            <div className="space-y-1.5">
              <div className="h-2 w-full rounded bg-muted/50" />
              <div className="h-2 w-3/4 rounded bg-muted/50" />
            </div>
            <div className="flex gap-1.5 mt-2">
              <div className="h-5 px-2 rounded bg-primary text-[11px] text-primary-foreground flex items-center justify-center">
                Primary
              </div>
              <div className="h-5 px-2 rounded bg-accent text-[11px] text-accent-foreground flex items-center justify-center">
                Accent
              </div>
            </div>
          </div>
        </CardContent>
        <CardFooter className="justify-end">
          <VersionBadge label="v" version={WEB_VERSION} />
        </CardFooter>
      </Card>

      {/* Language */}
      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-base">Language</CardTitle>
            <CardDescription>Interface language</CardDescription>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {LOCALES.map((l) => (
              <Button
                key={l.code}
                size="sm"
                variant={locale === l.code ? 'default' : 'outline'}
                className="h-9 text-xs gap-1.5"
                onClick={() => setLocale(l.code)}
              >
                <span>{l.flag}</span>
                <span>{l.name}</span>
              </Button>
            ))}
          </div>
        </CardContent>
        <CardFooter className="justify-end">
          <VersionBadge label="v" version={WEB_VERSION} />
        </CardFooter>
      </Card>
    </>
  )
}
