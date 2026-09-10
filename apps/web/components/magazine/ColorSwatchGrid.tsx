'use client'

import { Card, CardContent, CardHeader, CardTitle, cn } from '@sloughgpt/strui'

interface ColorToken {
  name: string
  rgb: string
  label: string
  desc: string
}

interface ColorSwatchGridProps {
  title: string
  colors: ColorToken[]
}

function Swatch({ rgb, label, description }: { rgb: string; label: string; description?: string }) {
  return (
    <div className="flex flex-col items-center gap-1.5">
      <div
        className={cn('h-12 w-12 rounded-md border border-border/40 shadow-sm transition-transform hover:scale-110')}
        style={{ backgroundColor: `rgb(${rgb})` }}
        role="img"
        aria-label={`${label}: rgb(${rgb})`}
      />
      <div className="text-center">
        <div className="text-[10px] font-medium">{label}</div>
        <div className="font-mono text-[10px] text-muted-foreground">{rgb}</div>
        {description && <div className="text-[10px] text-muted-foreground">{description}</div>}
      </div>
    </div>
  )
}

export function ColorSwatchGrid({ title, colors }: ColorSwatchGridProps) {
  return (
    <Card className="magazine-card">
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="px-2.5 pb-2.5">
        <div className="grid grid-cols-4 gap-1.5 sm:grid-cols-6 md:grid-cols-9">
          {colors.map((c) => (
            <Swatch key={c.name} rgb={c.rgb} label={c.label} description={c.desc} />
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
