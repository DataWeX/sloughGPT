'use client'

/** Gallery of the strui component set. */
import { useState } from 'react'
import { SectionHeader } from './magazine-ui'
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Checkbox,
  Input,
  Progress,
  Separator,
  Switch,
} from '@sloughgpt/strui'

export function MagazineComponentGallery() {
  const [switchOn, setSwitchOn] = useState(false)
  const [checkboxOn, setCheckboxOn] = useState(false)

  return (
    <>
      {/* ─── 4. Component Gallery ─────────────────────────────── */}
      <div className="space-y-4">
        <SectionHeader number="04" title="Component Gallery" />

        {/* Buttons */}
        <Card className="magazine-card">
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-base">Buttons</CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <Button size="sm">Primary</Button>
                <Button size="sm" variant="secondary">
                  Secondary
                </Button>
                <Button size="sm" variant="destructive">
                  Destructive
                </Button>
                <Button size="sm" variant="outline">
                  Outline
                </Button>
                <Button size="sm" variant="ghost">
                  Ghost
                </Button>
                <Button size="sm" variant="link">
                  Link
                </Button>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Button size="sm" disabled>
                  Disabled
                </Button>
                <Button size="sm" variant="outline" disabled>
                  Disabled
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                All buttons use{' '}
                <code className="font-mono text-[10px] bg-muted px-1 rounded">h-9</code> default,{' '}
                <code className="font-mono text-[10px] bg-muted px-1 rounded">h-11</code> for
                primary CTAs. Touch target minimum: 24x24px.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Badges */}
        <Card className="magazine-card">
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-base">Badges</CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="default">Default</Badge>
              <Badge variant="secondary">Secondary</Badge>
              <Badge variant="destructive">Destructive</Badge>
              <Badge variant="outline">Outline</Badge>
            </div>
          </CardContent>
        </Card>

        {/* Form Elements */}
        <Card className="magazine-card">
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-base">Form Elements</CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            <div className="space-y-4">
              <div className="grid gap-1.5 sm:grid-cols-2">
                <div className="space-y-2">
                  <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Input
                  </label>
                  <Input placeholder="Enter text..." aria-label="Example input" />
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Input (disabled)
                  </label>
                  <Input placeholder="Disabled input" disabled aria-label="Disabled input" />
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-4">
                <div className="flex items-center gap-2">
                  <Switch
                    checked={switchOn}
                    onCheckedChange={setSwitchOn}
                    aria-label="Example switch"
                  />
                  <span className="text-sm">Switch</span>
                </div>
                <div className="flex items-center gap-2">
                  <Checkbox
                    checked={checkboxOn}
                    onCheckedChange={(v) => setCheckboxOn(v === true)}
                    aria-label="Example checkbox"
                  />
                  <span className="text-sm">Checkbox</span>
                </div>
              </div>
              <div className="space-y-2">
                <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Progress
                </label>
                <Progress value={64} className="h-2" aria-label="Progress: 64%" />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Cards */}
        <Card className="magazine-card">
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-base">Cards</CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            <div className="grid gap-1.5 sm:grid-cols-3">
              <div className="rounded-lg border border-border/40 p-2.5 hover:bg-muted/20 shadow-sm">
                <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Shadow SM
                </div>
                <div className="mt-1 text-sm">Subtle elevation for cards at rest.</div>
              </div>
              <div className="rounded-lg border border-border/40 p-2.5 hover:bg-muted/20 shadow-md">
                <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Shadow MD
                </div>
                <div className="mt-1 text-sm">Hover states, dropdowns.</div>
              </div>
              <div className="rounded-lg border border-border/40 p-2.5 hover:bg-muted/20 shadow-lg">
                <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Shadow LG
                </div>
                <div className="mt-1 text-sm">Modals, popovers.</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Separator />
    </>
  )
}
