'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'
import type { LucideIcon } from 'lucide-react'

export interface Shortcut {
  keys: string[]
  label: string
  icon: LucideIcon
}

export interface ShortcutCategoryCardProps {
  title: string
  shortcuts: Shortcut[]
}

export function ShortcutCategoryCard({ title, shortcuts }: ShortcutCategoryCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-1">
          {shortcuts.map((s, i) => (
            <div key={i} className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-muted/50">
              <div className="flex items-center gap-2">
                <s.icon className="h-3.5 w-3.5 text-muted-foreground" />
                <span className="text-sm">{s.label}</span>
              </div>
              <div className="flex items-center gap-0.5">
                {s.keys.map((key, ki) => (
                  <span key={ki}>
                    <kbd className="px-2 py-1 text-xs font-mono bg-muted border rounded shadow-sm">
                      {key}
                    </kbd>
                    {ki < s.keys.length - 1 && (
                      <span className="text-muted-foreground text-xs mx-0.5">+</span>
                    )}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
