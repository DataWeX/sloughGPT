'use client'

/** Spacing and corner-radius scale. */
import { radiusScale, spacingScale } from './magazine-tokens'
import { SectionHeader } from './magazine-ui'
import { Card, CardContent, CardHeader, CardTitle, Separator, cn } from '@sloughgpt/strui'

export function MagazineSpacingRadius() {
  return (
    <>
      {/* ─── 5. Spacing & Radius ──────────────────────────────── */}
      <div className="space-y-4">
        <SectionHeader number="05" title="Spacing & Radius" />

        <Card className="magazine-card">
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-base">Spacing Scale</CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            <div className="space-y-3">
              {spacingScale.map((s) => (
                <div key={s.token} className="flex items-center gap-3">
                  <span className="w-16 shrink-0 font-mono text-[10px] text-muted-foreground">
                    {s.token}
                  </span>
                  <div className={cn('flex', s.class)}>
                    <div
                      className="h-6 rounded-sm"
                      style={{ backgroundColor: 'rgb(var(--primary))', width: s.px, opacity: 0.7 }}
                    />
                  </div>
                  <span className="font-mono text-[10px] text-muted-foreground">{s.px}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="magazine-card">
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-base">Border Radius</CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            <div className="flex flex-wrap items-end gap-4">
              {radiusScale.map((r) => (
                <div key={r.token} className="flex flex-col items-center gap-1.5">
                  <div
                    className="h-12 w-12 border border-border"
                    style={{
                      borderRadius: r.px,
                      backgroundColor: 'rgb(var(--primary))',
                      opacity: 0.15,
                    }}
                  />
                  <div className="text-center">
                    <div className="font-mono text-[10px] text-muted-foreground">{r.token}</div>
                    <div className="font-mono text-[10px] text-muted-foreground">{r.px}</div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="magazine-card">
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-base">Shadow Depth</CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            <div className="flex flex-wrap items-end gap-6">
              {[
                { token: 'shadow-sm', label: 'SM' },
                { token: 'shadow-md', label: 'MD' },
                { token: 'shadow-lg', label: 'LG' },
                { token: 'shadow-xl', label: 'XL' },
              ].map((s) => (
                <div key={s.token} className="flex flex-col items-center gap-2">
                  <div
                    className={cn('h-16 w-16 rounded-lg border border-border', s.token)}
                    style={{ backgroundColor: 'rgb(var(--card))' }}
                  />
                  <div className="text-center">
                    <div className="text-[10px] font-medium">{s.label}</div>
                    <div className="font-mono text-[10px] text-muted-foreground">{s.token}</div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <Separator />
    </>
  )
}
