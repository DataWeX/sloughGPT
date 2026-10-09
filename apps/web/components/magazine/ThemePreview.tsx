'use client'

import { Card, CardContent, CardHeader, CardTitle, Button, cn } from '@sloughgpt/strui'

interface AccentTheme {
  name: string
  label: string
  rgb: string
}

interface ThemePreviewProps {
  themes: AccentTheme[]
  onApply?: (themeName: string) => void
}

export function ThemePreview({ themes, onApply }: ThemePreviewProps) {
  return (
    <Card className="magazine-card">
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-base">Theme in Context</CardTitle>
      </CardHeader>
      <CardContent className="px-2.5 pb-2.5">
        <div className="grid gap-1.5 sm:grid-cols-2">
          {themes.map((t) => (
            <div
              key={t.name}
              className="flex items-center gap-3 rounded-lg border border-border/40 p-2.5 hover:bg-muted/20"
            >
              <div
                className="h-8 w-8 shrink-0 rounded-md shadow-sm"
                style={{ backgroundColor: `rgb(${t.rgb})` }}
              />
              <div className="min-w-0">
                <div className="text-xs font-medium">{t.label}</div>
                <div className="font-mono text-[10px] text-muted-foreground">
                  html.theme-{t.name}
                </div>
              </div>
              <Button
                size="sm"
                variant="outline"
                className="ml-auto shrink-0"
                style={{
                  borderColor: `rgb(${t.rgb})`,
                  color: `rgb(${t.rgb})`,
                }}
                onClick={() => onApply?.(t.name)}
              >
                Apply
              </Button>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
