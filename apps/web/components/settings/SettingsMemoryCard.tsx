'use client'

import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle, Textarea } from '@sloughgpt/strui'

interface SettingsMemoryCardProps {
  customContext: string
  onChange: (value: string) => void
  version?: string
}

export function SettingsMemoryCard({ customContext, onChange, version }: SettingsMemoryCardProps) {
  return (
    <Card>
      <CardHeader>
        <div>
          <CardTitle className="text-base">Memory</CardTitle>
          <CardDescription>Custom instructions included with every prompt</CardDescription>
        </div>
      </CardHeader>
      <CardContent>
        <Textarea
          className="min-h-[120px]"
          placeholder="e.g., You are a helpful coding assistant. Keep responses concise..."
          value={customContext}
          onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => onChange(e.target.value)}
          aria-label="Custom instructions"
        />
      </CardContent>
      <CardFooter className="justify-end">
        {version && <span className="text-[10px] font-mono px-1.5 py-0 h-5 text-muted-foreground border border-border/50 rounded text-xs">v{version}</span>}
      </CardFooter>
    </Card>
  )
}
