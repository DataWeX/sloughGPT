'use client'

/** Hover, switch and checkbox state demos. */
import { useState } from 'react'
import { SectionHeader } from './magazine-ui'
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Input,
  Separator,
  cn,
} from '@sloughgpt/strui'

export function MagazineInteractiveStates() {
  const [interactiveHover, setInteractiveHover] = useState(false)

  return (
    <>
      {/* ─── 6. Interactive States ────────────────────────────── */}
      <div className="space-y-4">
        <SectionHeader number="06" title="Interactive States" />

        <Card className="magazine-card">
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-base">Button States</CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            <div className="flex flex-wrap items-center gap-3">
              <Button size="sm">Default</Button>
              <Button size="sm" className="bg-primary/90">
                Hover
              </Button>
              <Button size="sm" className="ring-2 ring-ring ring-offset-2">
                Focus
              </Button>
              <Button size="sm" className="active:scale-[0.98]">
                Active
              </Button>
              <Button size="sm" disabled className="opacity-40 pointer-events-none">
                Disabled
              </Button>
            </div>
            <p className="mt-3 text-xs text-muted-foreground">
              Focus:{' '}
              <code className="font-mono text-[10px] bg-muted px-1 rounded">
                ring-2 ring-ring ring-offset-2
              </code>
              . Active:{' '}
              <code className="font-mono text-[10px] bg-muted px-1 rounded">scale-[0.98]</code>{' '}
              press feedback. Disabled:{' '}
              <code className="font-mono text-[10px] bg-muted px-1 rounded">
                opacity-40 pointer-events-none
              </code>
              .
            </p>
          </CardContent>
        </Card>

        <Card className="magazine-card">
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-base">Hover Preview</CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            <div className="grid gap-1.5 sm:grid-cols-2">
              <button
                type="button"
                className={cn(
                  'rounded-md border p-4 text-left text-sm transition-all duration-200',
                  'hover:border-primary/50 hover:shadow-md hover:-translate-y-0.5',
                  'focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
                  interactiveHover
                    ? 'border-primary/50 shadow-md -translate-y-0.5'
                    : 'border-border',
                )}
                onMouseEnter={() => setInteractiveHover(true)}
                onMouseLeave={() => setInteractiveHover(false)}
                aria-label="Hoverable card demo"
              >
                <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Hover Me
                </div>
                <div className="mt-1">Hover card with elevation change.</div>
              </button>
              <div className="rounded-md border border-border p-4">
                <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Focus Ring
                </div>
                <div className="mt-2">
                  <Input
                    placeholder="Tab here to see focus ring..."
                    aria-label="Focus ring demo"
                    className="focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                  />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Separator />
    </>
  )
}
