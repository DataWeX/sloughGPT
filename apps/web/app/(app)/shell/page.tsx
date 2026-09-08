'use client'

import { useState } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'
import { TerminalPanel } from '@/components/shell/TerminalPanel'
import { V86TerminalPanel } from '@/components/shell/V86TerminalPanel'

type ShellMode = 'backend' | 'v86'

export default function ShellPage() {
  const [mode, setMode] = useState<ShellMode>('backend')

  return (
    <PageContainer title="Shell">
      <Card className="h-[calc(100vh-8rem)]">
        <CardHeader className="pb-2 pt-2.5 px-2.5">
          <CardTitle className="flex items-center justify-between text-[11px] font-medium">
            <span>Dait Shell</span>
            <div className="flex gap-0.5">
              <Button
                variant={mode === 'backend' ? 'default' : 'ghost'}
                size="sm"
                className="h-6 text-[10px]"
                onClick={() => setMode('backend')}
              >
                Backend
              </Button>
              <Button
                variant={mode === 'v86' ? 'default' : 'ghost'}
                size="sm"
                className="h-6 text-[10px]"
                onClick={() => setMode('v86')}
              >
                Browser VM
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="h-[calc(100%-3rem)] px-2.5 pb-2.5">
          {mode === 'backend' ? (
            <TerminalPanel className="h-full" />
          ) : (
            <V86TerminalPanel className="h-full" />
          )}
        </CardContent>
      </Card>
    </PageContainer>
  )
}
