'use client'
export const dynamic = 'force-dynamic'

/**
 * Developer — composition root.
 *
 * The seven tool tabs are self-contained sibling components (each fetches
 * through the shared query cache), so this file is only the shell: tab
 * state, the startup-stage header chip, and the Tabs wiring.
 */
import { useState } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Tabs, TabsList, TabsTrigger, TabsContent, StatusDot } from '@sloughgpt/strui'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { revalidateAll } from '@/lib/cache'
import { ShellTab } from './dev-shell-tab'
import { FilesTab } from './dev-files-tab'
import { VoiceTab } from './dev-voice-tab'
import { ApiTab } from './dev-api-tab'
import { QuickActionsTab } from './dev-quick-actions-tab'
import { StartupTab } from './dev-startup-tab'

export default function DeveloperPage() {
  const [tab, setTab] = useState<string>('shell')
  // `r` refresh revalidates the shared query cache and nothing else. Every tab
  // that fetches reads through that cache (files/models via their own hooks,
  // voice/startup via the queries below), so cached data is re-read *in place*:
  // no remounts, no `key` bumps — terminal scrollback, typed input and scroll
  // positions all survive. Shell/api/quick fetch nothing at all.
  useRefreshShortcut(() => revalidateAll())
  const { startupStage, startupElapsed } = useLiveStatus()

  return (
    <PageContainer
      title="Developer"
      subtitle="Shell, files, voice & API tools"
      headerRight={
        <div className="flex items-center gap-3">
          {startupStage && startupStage !== 'ready' && startupStage !== 'background' && (
            <div className="flex items-center gap-2 rounded-lg border border-warning/20 bg-warning/[0.06] px-3 py-1.5">
              <StatusDot tone="warning" pulse />
              <span className="text-[10px] font-medium text-warning uppercase tracking-wider">
                {startupStage} {startupElapsed > 0 && `· ${startupElapsed.toFixed(0)}s`}
              </span>
            </div>
          )}
          {startupStage === 'ready' && (
            <div className="flex items-center gap-2 rounded-lg border border-success/20 bg-success/[0.06] px-3 py-1.5">
              <StatusDot tone="success" />
              <span className="text-[10px] font-medium text-success uppercase tracking-wider">
                Ready
              </span>
            </div>
          )}
        </div>
      }
    >
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList aria-label="Developer tools">
          <TabsTrigger value="shell">Terminal</TabsTrigger>
          <TabsTrigger value="files">Files</TabsTrigger>
          <TabsTrigger value="voice">Voice</TabsTrigger>
          <TabsTrigger value="api">API</TabsTrigger>
          <TabsTrigger value="quick">Quick Actions</TabsTrigger>
          <TabsTrigger value="startup">Startup</TabsTrigger>
        </TabsList>

        <TabsContent value="shell">
          <ShellTab />
        </TabsContent>
        <TabsContent value="files">
          <FilesTab />
        </TabsContent>
        <TabsContent value="voice">
          <VoiceTab />
        </TabsContent>
        <TabsContent value="api">
          <ApiTab />
        </TabsContent>
        <TabsContent value="quick">
          <QuickActionsTab />
        </TabsContent>
        <TabsContent value="startup">
          <StartupTab />
        </TabsContent>
      </Tabs>
    </PageContainer>
  )
}
