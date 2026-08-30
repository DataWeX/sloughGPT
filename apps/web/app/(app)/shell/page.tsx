'use client'

import { PageContainer } from '@/components/PageContainer'
import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'
import { TerminalPanel } from '@/components/shell/TerminalPanel'

export default function ShellPage() {
  return (
    <PageContainer title="Shell">
      <Card className="h-[calc(100vh-8rem)]">
        <CardHeader>
          <CardTitle>Dait Shell</CardTitle>
        </CardHeader>
        <CardContent className="h-[calc(100%-3rem)]">
          <TerminalPanel className="h-full" />
        </CardContent>
      </Card>
    </PageContainer>
  )
}
