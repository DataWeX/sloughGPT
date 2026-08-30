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
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Dait Shell</span>
            <div className="flex gap-2">
              <Button
                variant={mode === 'backend' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setMode('backend')}
              >
                Backend
              </Button>
              <Button
                variant={mode === 'v86' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setMode('v86')}
              >
                Browser VM
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="h-[calc(100%-3rem)]">
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
