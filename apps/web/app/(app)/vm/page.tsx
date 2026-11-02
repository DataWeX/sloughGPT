'use client'

// VM console — thin composition. All state, chatDB hydration, the training
// poller, and the run/launch handlers live in ./useVmPage; the assembly and
// browser tab markup lives in ./components/* (extracted by the
// page-decomposition refactor).

import { PageContainer } from '@/components/PageContainer'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@sloughgpt/strui'
import { useVmPage } from './useVmPage'
import { VmAssemblyTab } from './components/VmAssemblyTab'
import { VmBrowserTab } from './components/VmBrowserTab'

export default function VMPage() {
  const vm = useVmPage()

  return (
    <PageContainer
      title="VM Console"
      subtitle="x86-32 assembly sandbox + browser Linux"
      maxWidth="max-w-6xl"
    >
      <Tabs value={vm.vmMode} onValueChange={(v) => vm.setVmMode(v as 'assembly' | 'browser')}>
        <TabsList aria-label="VM mode">
          <TabsTrigger value="assembly">Assembly Sandbox</TabsTrigger>
          <TabsTrigger value="browser">Browser VM</TabsTrigger>
        </TabsList>

        <TabsContent value="assembly">
          <VmAssemblyTab vm={vm} />
        </TabsContent>

        <TabsContent value="browser">
          <VmBrowserTab />
        </TabsContent>
      </Tabs>
    </PageContainer>
  )
}
