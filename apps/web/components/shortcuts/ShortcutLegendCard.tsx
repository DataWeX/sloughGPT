'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'
import { Info } from 'lucide-react'

export interface LegendItem {
  key: string
  description: string
}

export interface ShortcutLegendCardProps {
  title?: string
  items: LegendItem[]
}

export function ShortcutLegendCard({ title = 'Keyboard Shortcuts', items }: ShortcutLegendCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs flex items-center gap-2">
          <Info className="h-3.5 w-3.5" />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-xs text-muted-foreground mb-3">
          Keyboard shortcuts available across the application. Shortcuts are disabled when focus is in input fields.
        </p>
        <div className="space-y-1.5">
          {items.map((item, i) => (
            <div key={i} className="flex items-center gap-2 py-1">
              <kbd className="px-2 py-1 text-xs font-mono bg-muted border rounded shadow-sm shrink-0">
                {item.key}
              </kbd>
              <span className="text-xs text-muted-foreground">{item.description}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
