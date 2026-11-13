'use client'

/** Type scale reference. */
import { typographyScale } from './magazine-tokens'
import { SectionHeader } from './magazine-ui'
import { Badge, Card, CardContent, CardHeader, CardTitle, Separator, cn } from '@sloughgpt/strui'

export function MagazineTypography() {
  return (
    <>
      {/* ─── 3. Typography Scale ──────────────────────────────── */}
      <div className="space-y-4">
        <SectionHeader number="03" title="Typography Scale" />

        <Card className="magazine-card">
          <CardContent className="pt-2.5 px-2.5 pb-2.5">
            <div className="space-y-4">
              {typographyScale.map((t) => (
                <div
                  key={t.role}
                  className="flex flex-col gap-1 border-b border-border/40 pb-3 last:border-0 last:pb-0"
                >
                  <div className="flex items-baseline justify-between">
                    <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                      {t.role}
                    </span>
                    <span className="font-mono text-[10px] text-muted-foreground">{t.class}</span>
                  </div>
                  <p className={cn(t.class, t.font)}>{t.sample}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="magazine-card">
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-base">Font Families</CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            <div className="space-y-3">
              <div className="flex items-center justify-between border-b border-border/40 pb-2">
                <div>
                  <div
                    className="text-sm font-medium"
                    style={{ fontFamily: 'var(--font-rubik), system-ui, sans-serif' }}
                  >
                    Rubik
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Body text &mdash; system-ui fallback
                  </div>
                </div>
                <Badge variant="secondary">--font-rubik</Badge>
              </div>
              <div className="flex items-center justify-between border-b border-border/40 pb-2">
                <div>
                  <div
                    className="text-sm font-medium"
                    style={{ fontFamily: 'var(--font-lato), system-ui, sans-serif' }}
                  >
                    Lato
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Numeric data &mdash; system-ui fallback
                  </div>
                </div>
                <Badge variant="secondary">--font-lato</Badge>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-mono text-sm font-medium">JetBrains Mono</div>
                  <div className="text-xs text-muted-foreground">
                    Code &mdash; ui-monospace fallback
                  </div>
                </div>
                <Badge variant="secondary">--font-jetbrains-mono</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Separator />
    </>
  )
}
