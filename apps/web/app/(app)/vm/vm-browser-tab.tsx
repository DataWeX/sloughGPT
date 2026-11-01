'use client'

/** Browser VM tab: the v86 Linux terminal. */
import { Card, CardContent } from '@sloughgpt/strui'
import { V86TerminalPanel } from '@/components/shell/V86TerminalPanel'

export function BrowserTab() {
  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="p-3">
          <div className="flex items-center gap-3">
            <p className="text-xs text-muted-foreground">
              Real Linux running in your browser via v86. Boot into a Buildroot image with BusyBox,
              Python, and the Dait shell.
            </p>
          </div>
        </CardContent>
      </Card>
      <V86TerminalPanel className="h-[calc(100vh-14rem)]" />
    </div>
  )
}
