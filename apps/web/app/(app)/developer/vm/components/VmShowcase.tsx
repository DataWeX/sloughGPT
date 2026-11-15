'use client'

// Pre-run showcase (card 20261102_014255): one entry per sample program
// showing what it does and its REAL captured output — rendered only before
// the first run so first paint is legible. Run replaces it with the live
// result (the parent gates on result === null); clicking an entry loads that
// program into the editor. Entries with no textual output carry an explicit
// 'sample' badge — never presented as live results.

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { SHOWCASE } from '../vm-data'

interface VmShowcaseProps {
  onLoad: (name: string) => void
}

export function VmShowcase({ onLoad }: VmShowcaseProps) {
  return (
    <Card data-testid="vm-showcase">
      <CardHeader>
        <div className="flex items-baseline justify-between gap-2 flex-wrap">
          <CardTitle className="text-base">What the programs do</CardTitle>
          <p className="text-xs text-muted-foreground">
            Real captured output — click one to load it into the editor, then Run.
          </p>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
          {SHOWCASE.map((entry) => (
            <button
              key={entry.name}
              type="button"
              aria-label={`Load ${entry.name} program`}
              onClick={() => onLoad(entry.name)}
              className="group text-left rounded-lg border border-border bg-card p-3 flex flex-col gap-2 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
            >
              <span className="flex items-center justify-between gap-2">
                <span className="font-mono text-sm font-medium">{entry.name}</span>
                <span
                  className={
                    entry.kind === 'captured'
                      ? 'text-xs px-1.5 py-0.5 rounded bg-success/10 text-success'
                      : 'text-xs px-1.5 py-0.5 rounded bg-warning/10 text-warning'
                  }
                >
                  {entry.kind}
                </span>
              </span>
              <span className="text-xs text-muted-foreground">{entry.description}</span>
              <pre className="text-xs font-mono bg-muted/30 p-2 rounded overflow-x-auto whitespace-pre-wrap max-h-20 overflow-y-auto">
                {entry.output}
              </pre>
              {entry.note && <span className="text-xs text-muted-foreground">{entry.note}</span>}
            </button>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
