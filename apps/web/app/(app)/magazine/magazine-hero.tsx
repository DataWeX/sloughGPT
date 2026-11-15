'use client'

/** Hero cover: Noir Violet title, swatch and badges. */
import { Badge, Card } from '@sloughgpt/strui'

export function MagazineHero() {
  return (
    <>
      {/* ─── 1. Hero / Cover ──────────────────────────────────── */}
      <Card className="magazine-card overflow-hidden">
        <div className="relative flex flex-col items-center justify-center px-6 py-12 text-center sm:py-16">
          {/* Background accent glow */}
          <div
            className="pointer-events-none absolute inset-0 opacity-[0.07]"
            style={{
              background:
                'radial-gradient(ellipse at 50% 30%, rgb(var(--primary)) 0%, transparent 70%)',
            }}
          />
          <div className="relative z-10 space-y-4">
            {/* Primary color swatch */}
            <div className="flex justify-center">
              <div
                className="h-20 w-20 rounded-xl shadow-lg transition-transform hover:scale-105"
                style={{ backgroundColor: 'rgb(var(--primary))' }}
                role="img"
                aria-label="Primary violet color swatch"
              />
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-semibold tracking-tight">
                <span style={{ color: 'rgb(var(--primary))' }}>Noir</span>{' '}
                <span style={{ color: 'rgb(var(--foreground))' }}>Violet</span>
              </h1>
              <p className="mt-2 text-sm" style={{ color: 'rgb(var(--muted-foreground))' }}>
                Warm, sophisticated, technical.
              </p>
            </div>
            <div className="flex items-center justify-center gap-2">
              <Badge variant="default">v1.0</Badge>
              <Badge variant="secondary">Design System</Badge>
              <Badge
                variant="outline"
                style={{
                  borderColor: 'rgb(var(--accent))',
                  color: 'rgb(var(--accent))',
                }}
              >
                Accent: Terracotta
              </Badge>
            </div>
            <p
              className="mx-auto max-w-md text-xs"
              style={{ color: 'rgb(var(--muted-foreground))' }}
            >
              A calm confidence. Not sterile, not playful. A tool that respects its user.
            </p>
          </div>
        </div>
      </Card>
    </>
  )
}
