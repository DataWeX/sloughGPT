'use client'

// VM console tab — thin composition. Folded from app/(app)/vm/page.tsx by
// route-map card 6/7 (20261101_154057): the /vm page is gone, the console
// now lives as the seventh tab of /developer. All state, chatDB hydration,
// the training poller, and the run/launch handlers live in ./vm/useVmPage;
// the assembly and browser tab markup lives in ./vm/components/*.
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@sloughgpt/strui'
import { useVmPage } from './vm/useVmPage'
import { VmAssemblyTab } from './vm/components/VmAssemblyTab'
import { VmBrowserTab } from './vm/components/VmBrowserTab'

export function VmTab() {
  const vm = useVmPage()

  return (
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
  )
}
