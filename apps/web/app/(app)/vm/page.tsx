'use client'

/**
 * VM Console — composition root.
 *
 * State and IO live in `useVmWorkbench`; rendering lives in the `vm-*`
 * siblings next to this file. Only the tab shell belongs here.
 */
import { PageContainer } from '@/components/PageContainer'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@sloughgpt/strui'
import { useVmWorkbench } from './useVmWorkbench'
import { AssemblyTab } from './vm-assembly-tab'
import { BrowserTab } from './vm-browser-tab'

export default function VMPage() {
  const vm = useVmWorkbench()

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
          <AssemblyTab vm={vm} />
        </TabsContent>

        <TabsContent value="browser">
          <BrowserTab />
        </TabsContent>
      </Tabs>
    </PageContainer>
  )
}
