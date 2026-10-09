'use client'

import { Card, CardContent, cn } from '@sloughgpt/strui'

interface TypographyEntry {
  role: string
  class: string
  sample: string
  font: string
  weight: string
}

interface TypographyScaleProps {
  entries: TypographyEntry[]
}

export function TypographyScale({ entries }: TypographyScaleProps) {
  return (
    <Card className="magazine-card">
      <CardContent className="pt-2.5 px-2.5 pb-2.5">
        <div className="space-y-4">
          {entries.map((t) => (
            <div key={t.role} className="flex flex-col gap-1 border-b border-border/40 pb-3 last:border-0 last:pb-0">
              <div className="flex items-baseline justify-between">
                <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                  {t.role}
                </span>
                <span className="font-mono text-[10px] text-muted-foreground">
                  {t.class}
                </span>
              </div>
              <p className={cn(t.class, t.font)}>
                {t.sample}
              </p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
