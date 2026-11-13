'use client'

/** Light and dark token swatch grids. */
import { darkColors, lightColors } from './magazine-tokens'
import { SectionHeader, Swatch } from './magazine-ui'
import { Card, CardContent, CardHeader, CardTitle, Separator } from '@sloughgpt/strui'

export function MagazineColorPalette() {
  return (
    <>
      {/* ─── 2. Color Palette ─────────────────────────────────── */}
      <div className="space-y-4">
        <SectionHeader number="02" title="Color Palette" />

        <Card className="magazine-card">
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-base">Light Mode</CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            <div className="grid grid-cols-4 gap-1.5 sm:grid-cols-6 md:grid-cols-9">
              {lightColors.map((c) => (
                <Swatch key={c.name} rgb={c.rgb} label={c.label} description={c.desc} size="md" />
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="magazine-card">
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-base">Dark Mode</CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            <div className="grid grid-cols-4 gap-1.5 sm:grid-cols-6 md:grid-cols-9">
              {darkColors.map((c) => (
                <Swatch key={c.name} rgb={c.rgb} label={c.label} description={c.desc} size="md" />
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <Separator />
    </>
  )
}
