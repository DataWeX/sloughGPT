'use client'

/** Dark mode token preview. */
import { SectionHeader, Swatch } from './magazine-ui'
import { Separator } from '@sloughgpt/strui'

export function MagazineDarkPreview() {
  return (
    <>
      {/* ─── 7. Dark Mode Preview ─────────────────────────────── */}
      <div className="space-y-4">
        <SectionHeader number="07" title="Dark Mode Preview" />

        <div
          className="magazine-card rounded-lg border p-6"
          style={{
            backgroundColor: 'rgb(17, 15, 24)',
            borderColor: 'rgb(52, 46, 72)',
          }}
        >
          <div className="space-y-4">
            <div className="flex items-baseline justify-between">
              <h3 className="text-base font-medium" style={{ color: 'rgb(238, 234, 248)' }}>
                Dark Mode Palette
              </h3>
              <span className="font-mono text-[10px]" style={{ color: 'rgb(150, 140, 172)' }}>
                html.dark
              </span>
            </div>
            <div className="grid grid-cols-3 gap-1.5 sm:grid-cols-6">
              {[
                { label: 'Background', rgb: '17 15 24' },
                { label: 'Card', rgb: '28 25 38' },
                { label: 'Primary', rgb: '192 170 244' },
                { label: 'Accent', rgb: '240 176 130' },
                { label: 'Success', rgb: '72 192 140' },
                { label: 'Border', rgb: '52 46 72' },
              ].map((c) => (
                <Swatch key={c.label} rgb={c.rgb} label={c.label} size="sm" />
              ))}
            </div>
            {/* Sample dark UI elements */}
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                className="rounded-md px-3 py-1.5 text-xs font-medium transition-colors"
                style={{
                  backgroundColor: 'rgb(192, 170, 244)',
                  color: 'rgb(25, 22, 36)',
                }}
              >
                Primary Button
              </button>
              <button
                type="button"
                className="rounded-md border px-3 py-1.5 text-xs font-medium transition-colors"
                style={{
                  borderColor: 'rgb(52, 46, 72)',
                  color: 'rgb(238, 234, 248)',
                }}
              >
                Outline Button
              </button>
              <span
                className="rounded-md px-2 py-0.5 text-[10px] font-medium"
                style={{
                  backgroundColor: 'rgb(72, 192, 140)',
                  color: 'rgb(25, 22, 36)',
                }}
              >
                Success Badge
              </span>
              <span
                className="rounded-md px-2 py-0.5 text-[10px] font-medium"
                style={{
                  backgroundColor: 'rgb(240, 176, 130)',
                  color: 'rgb(25, 22, 36)',
                }}
              >
                Accent Badge
              </span>
            </div>
            <p className="text-xs" style={{ color: 'rgb(150, 140, 172)' }}>
              Body text in muted foreground. Warm, deep charcoal-violet with vibrant lilac accents.
            </p>
          </div>
        </div>
      </div>

      <Separator />
    </>
  )
}
