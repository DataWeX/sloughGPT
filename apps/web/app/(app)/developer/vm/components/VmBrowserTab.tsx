'use client'

// Browser VM tab content: v86 intro card + the terminal panel. Extracted from
// page.tsx by the page-decomposition refactor.

import { Card, CardContent } from '@sloughgpt/strui'
import { V86TerminalPanel } from '@/components/shell/V86TerminalPanel'

export function VmBrowserTab() {
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
