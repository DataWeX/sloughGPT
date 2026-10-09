'use client'

import { Card, CardContent, CardHeader, CardTitle, Button, Badge, Input, Switch, Progress } from '@sloughgpt/strui'

interface ComponentGalleryCardProps {
  switchOn: boolean
  onSwitchChange: (value: boolean) => void
}

export function ComponentGalleryCard({ switchOn, onSwitchChange }: ComponentGalleryCardProps) {
  return (
    <Card className="magazine-card">
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-base">Component Gallery</CardTitle>
      </CardHeader>
      <CardContent className="px-2.5 pb-2.5 space-y-4">
        <div>
          <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground mb-2">Buttons</div>
          <div className="flex flex-wrap items-center gap-2">
            <Button size="sm">Primary</Button>
            <Button size="sm" variant="secondary">Secondary</Button>
            <Button size="sm" variant="destructive">Destructive</Button>
            <Button size="sm" variant="outline">Outline</Button>
            <Button size="sm" variant="ghost">Ghost</Button>
          </div>
        </div>
        <div>
          <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground mb-2">Badges</div>
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="default">Default</Badge>
            <Badge variant="secondary">Secondary</Badge>
            <Badge variant="destructive">Destructive</Badge>
            <Badge variant="outline">Outline</Badge>
          </div>
        </div>
        <div>
          <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground mb-2">Form Elements</div>
          <div className="space-y-3">
            <Input placeholder="Enter text..." aria-label="Gallery input" />
            <div className="flex items-center gap-2">
              <Switch checked={switchOn} onCheckedChange={onSwitchChange} aria-label="Gallery switch" />
              <span className="text-sm">Switch</span>
            </div>
            <Progress value={64} className="h-2" aria-label="Progress: 64%" />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
