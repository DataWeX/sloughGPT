'use client'

/** Seven accent auras plus the palette spec. */
import { accentThemes } from './magazine-tokens'
import { SectionHeader, Swatch } from './magazine-ui'
import { Button, Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { MagazinePaletteSpec } from './magazine-palette-spec'

export function MagazineAccentThemes() {
  return (
    <>
      {/* ─── 8. Accent Themes ─────────────────────────────────── */}
      <div className="space-y-4">
        <SectionHeader number="08" title="Accent Themes" />

        <Card className="magazine-card">
          <CardContent className="pt-2.5 px-2.5 pb-2.5">
            <p className="mb-4 text-xs text-muted-foreground">
              Seven accent auras override{' '}
              <code className="font-mono text-[10px] bg-muted px-1 rounded">--primary</code>,{' '}
              <code className="font-mono text-[10px] bg-muted px-1 rounded">--secondary</code>,{' '}
              <code className="font-mono text-[10px] bg-muted px-1 rounded">--muted</code>, and{' '}
              <code className="font-mono text-[10px] bg-muted px-1 rounded">--border</code> via CSS
              class on{' '}
              <code className="font-mono text-[10px] bg-muted px-1 rounded">&lt;html&gt;</code>.
            </p>
            <div className="grid grid-cols-4 gap-1.5 sm:grid-cols-7">
              {accentThemes.map((t) => (
                <Swatch key={t.name} rgb={t.rgb} label={t.label} size="lg" />
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Theme preview row */}
        <Card className="magazine-card">
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-base">Theme in Context</CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            <div className="grid gap-1.5 sm:grid-cols-2">
              {accentThemes.map((t) => (
                <div
                  key={t.name}
                  className="flex items-center gap-3 rounded-lg border border-border/40 p-2.5 hover:bg-muted/20"
                >
                  <div
                    className="h-8 w-8 shrink-0 rounded-md shadow-sm"
                    style={{ backgroundColor: `rgb(${t.rgb})` }}
                  />
                  <div className="min-w-0">
                    <div className="text-xs font-medium">{t.label}</div>
                    <div className="font-mono text-[10px] text-muted-foreground">{t.desc}</div>
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    className="ml-auto shrink-0"
                    style={{
                      borderColor: `rgb(${t.rgb})`,
                      color: `rgb(${t.rgb})`,
                    }}
                  >
                    Apply
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        <MagazinePaletteSpec />
      </div>
    </>
  )
}
