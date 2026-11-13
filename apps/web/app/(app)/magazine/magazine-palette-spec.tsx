'use client'

/** Palette protocol spec: token tables, hex/rgb derivations and contrast invariants. */
import { SectionHeader } from './magazine-ui'
import {
  OS_BLUE_BASES,
  OS_BLUE_ID,
  OS_BLUE_LABEL,
  OS_BLUE_STATES,
  SHADES,
  contrastRatio,
  deriveRamp,
  deriveTokens,
  hexToRgb,
  rgbToHex,
} from '@/lib/os-blue-palette'
import { Card, CardContent } from '@sloughgpt/strui'

export function MagazinePaletteSpec() {
  return (
    <>
      {/* ─── 09 Palette Specification (rendered from the protocol module) ── */}
      <SectionHeader number="09" title={`Palette Specification — ${OS_BLUE_LABEL}`} />

      <Card className="magazine-card" data-testid="palette-spec">
        <CardContent className="pt-2.5 px-2.5 pb-2.5">
          <p className="mb-3 text-xs text-muted-foreground">
            <code className="font-mono text-[10px] bg-muted px-1 rounded">
              {`lib/${OS_BLUE_ID}-palette.ts`}
            </code>{' '}
            is the normative protocol — two base colors, an OKLCH hue-constant derivation, and a
            state contract. The CSS block, switcher swatch, and this page are projections; a
            drift-guard test re-runs the derivation and fails if{' '}
            <code className="font-mono text-[10px] bg-muted px-1 rounded">globals.css</code>{' '}
            diverges from it.
          </p>

          {/* Bases */}
          <div className="mb-3 grid gap-1.5 sm:grid-cols-2">
            {(['light', 'dark'] as const).map((mode) => (
              <div
                key={mode}
                className="flex items-center gap-3 rounded-lg border border-border/40 p-2.5"
              >
                <div
                  className="h-8 w-8 shrink-0 rounded-md shadow-sm"
                  style={{ backgroundColor: OS_BLUE_BASES[mode] }}
                />
                <div className="min-w-0">
                  <div className="text-xs font-medium capitalize">{mode} base</div>
                  <div className="font-mono text-[10px] text-muted-foreground">
                    {OS_BLUE_BASES[mode]} · anchor shade {deriveRamp(mode).anchor}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Ramps */}
          {(['light', 'dark'] as const).map((mode) => (
            <div key={mode} className="mb-3">
              <div className="mb-1.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                {mode} ramp — OKLCH hue-constant
              </div>
              <div className="grid grid-cols-5 gap-1 sm:grid-cols-10">
                {SHADES.map((shade) => {
                  const rgb = deriveRamp(mode).shades[shade]
                  return (
                    <div key={shade} className="flex flex-col items-center gap-1">
                      <div
                        className="h-8 w-full rounded border border-border/40"
                        style={{ backgroundColor: `rgb(${rgb.join(' ')})` }}
                        role="img"
                        aria-label={`${mode} ${shade}: ${rgbToHex(rgb)}`}
                      />
                      <div className="font-mono text-[9px] text-muted-foreground">{shade}</div>
                      <div className="font-mono text-[9px] text-muted-foreground">
                        {rgbToHex(rgb)}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          ))}

          {/* State contract */}
          <div className="mb-3">
            <div className="mb-1.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
              State contract — selected is the base color
            </div>
            <div className="grid gap-1.5 sm:grid-cols-2">
              {(['light', 'dark'] as const).map((mode) => (
                <div
                  key={mode}
                  className="flex items-center gap-2 rounded-lg border border-border/40 p-2.5"
                >
                  <span className="w-12 text-xs capitalize">{mode}</span>
                  {(['selected', 'hover', 'active'] as const).map((state) => {
                    const shade = OS_BLUE_STATES[mode][state]
                    const rgb = deriveRamp(mode).shades[shade]
                    return (
                      <span key={state} className="flex items-center gap-1.5">
                        <span
                          className="h-5 w-5 rounded border border-border/40"
                          style={{ backgroundColor: `rgb(${rgb.join(' ')})` }}
                        />
                        <span className="font-mono text-[9px] text-muted-foreground">
                          {state} {shade} {rgbToHex(rgb)}
                        </span>
                      </span>
                    )
                  })}
                </div>
              ))}
            </div>
          </div>

          {/* Invariants + projections */}
          <div className="grid gap-1.5 sm:grid-cols-2">
            <div className="rounded-lg border border-border/40 p-2.5">
              <div className="mb-1.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                Accessibility invariants
              </div>
              <ul className="space-y-1 font-mono text-[10px] text-muted-foreground">
                <li>
                  white on light primary —{' '}
                  {contrastRatio([255, 255, 255], hexToRgb(OS_BLUE_BASES.light)).toFixed(2)}:1
                </li>
                <li>dark fg on dark primary — {darkPrimaryContrast().toFixed(2)}:1</li>
              </ul>
            </div>
            <div className="rounded-lg border border-border/40 p-2.5">
              <div className="mb-1.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                Projections
              </div>
              <ul className="space-y-1 font-mono text-[10px] text-muted-foreground">
                <li>globals.css — static blocks (drift-guarded)</li>
                <li>theme-storage.ts — switcher + settings swatch</li>
                <li>this page — spec rendered from the module</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </>
  )
}

/* Contrast ratios for the palette-spec invariants, computed from the module. */
function darkPrimaryContrast(): number {
  const tokens = deriveTokens('dark')
  const fg = tokens['primary-foreground'].split(' ').map(Number) as [number, number, number]
  return contrastRatio(fg, hexToRgb(OS_BLUE_BASES.dark))
}
