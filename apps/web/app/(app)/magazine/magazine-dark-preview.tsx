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
          className="magazine-card magazine-dark-preview rounded-lg border p-6"
          style={{
            backgroundColor: 'rgb(var(--background))',
            borderColor: 'rgb(var(--border))',
          }}
        >
          <div className="space-y-4">
            <div className="flex items-baseline justify-between">
              <h3 className="text-base font-medium" style={{ color: 'rgb(var(--foreground))' }}>
                Dark Mode Palette
              </h3>
              <span
                className="font-mono text-[10px]"
                style={{ color: 'rgb(var(--muted-foreground))' }}
              >
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
                  backgroundColor: 'rgb(var(--primary))',
                  color: 'rgb(var(--primary-foreground))',
                }}
              >
                Primary Button
              </button>
              <button
                type="button"
                className="rounded-md border px-3 py-1.5 text-xs font-medium transition-colors"
                style={{
                  borderColor: 'rgb(var(--border))',
                  color: 'rgb(var(--foreground))',
                }}
              >
                Outline Button
              </button>
              <span
                className="rounded-md px-2 py-0.5 text-[10px] font-medium"
                style={{
                  backgroundColor: 'rgb(var(--success))',
                  color: 'rgb(var(--primary-foreground))',
                }}
              >
                Success Badge
              </span>
              <span
                className="rounded-md px-2 py-0.5 text-[10px] font-medium"
                style={{
                  backgroundColor: 'rgb(var(--accent))',
                  color: 'rgb(var(--primary-foreground))',
                }}
              >
                Accent Badge
              </span>
            </div>
            <p className="text-xs" style={{ color: 'rgb(var(--muted-foreground))' }}>
              Body text in muted foreground. Warm, deep charcoal-violet with vibrant lilac accents.
            </p>
          </div>
        </div>
      </div>

      <Separator />
    </>
  )
}
