'use client'

/** Appearance, palette and language cards. */
import { VersionBadge, WEB_VERSION } from './settings-ui'
import { LOCALES } from '@/hooks/useLocale'
import { PALETTE_IDS, PALETTE_LABELS } from '@/lib/theme-storage'
import {
  Button,
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
  ToggleGroupItem,
  ToggleGroup as ToggleGroupRadix,
} from '@sloughgpt/strui'
import type { SettingsWorkbench } from './useSettingsWorkbench'

export function SettingsAppearance({ wb }: { wb: SettingsWorkbench }) {
  const { locale, palette, setLocale, setPalette, settings, updateSettings } = wb

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
            {settings.theme !== 'light' && (
              <Button
                size="sm"
                variant="ghost"
                className="h-8 text-xs text-muted-foreground"
                onClick={() => updateSettings({ theme: 'light' })}
              >
                Reset
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <ToggleGroupRadix
            type="single"
            value={settings.theme}
            onValueChange={(v) => v && updateSettings({ theme: v as 'light' | 'dark' | 'system' })}
          >
            <ToggleGroupItem value="light">Light</ToggleGroupItem>
            <ToggleGroupItem value="dark">Dark</ToggleGroupItem>
            <ToggleGroupItem value="system">System</ToggleGroupItem>
          </ToggleGroupRadix>
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
              <div className="h-5 px-2 rounded bg-primary text-[8px] text-primary-foreground flex items-center justify-center">
                Button
              </div>
              <div className="h-5 px-2 rounded bg-muted text-[8px] text-muted-foreground flex items-center justify-center">
                Secondary
              </div>
            </div>
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
              <div className="h-5 px-2 rounded bg-primary text-[8px] text-primary-foreground flex items-center justify-center">
                Primary
              </div>
              <div className="h-5 px-2 rounded bg-accent text-[8px] text-accent-foreground flex items-center justify-center">
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
