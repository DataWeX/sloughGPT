'use client'
export const dynamic = 'force-dynamic'

/**
 * Agents — composition root.
 *
 * State and IO live in `useAgentsWorkbench`; rendering lives in the
 * `agents-*` siblings next to this file. Only the page shell (KPI strip,
 * header actions and the panel wiring) belongs here.
 *
 * Panels receive the workbench object by spread — each destructures the
 * names it renders, so a panel's prop surface stays explicit in its own
 * signature while the page never has to keep five prop lists in sync.
 */
import { PageContainer } from '@/components/PageContainer'
import { Button, KpiGrid, StatCard } from '@sloughgpt/strui'
import { useAgentsWorkbench } from './useAgentsWorkbench'
import { AVAILABLE_TOOLS } from './agents-constants'
import { NewAgentCard } from './agents-new-card'
import { AgentsListCard } from './agents-list-card'
import { OrchestrationCard } from './agents-orchestration-card'
import { RunHistoryCard } from './agents-run-history'
import { DeleteAgentDialog } from './agents-delete-dialog'

export default function AgentsPage() {
  const wb = useAgentsWorkbench()
  const { agents, toolCount, importInputRef, handleImport, handleExport } = wb

  const headerRight = (
    <div className="flex items-center gap-2">
      <input
        ref={importInputRef}
        type="file"
        accept=".json"
        className="hidden"
        onChange={handleImport}
      />
      <Button
        variant="outline"
        size="sm"
        className="h-6 text-[10px]"
        onClick={handleExport}
        disabled={agents.length === 0}
      >
        Export
      </Button>
      <Button
        variant="outline"
        size="sm"
        className="h-6 text-[10px]"
        onClick={() => importInputRef.current?.click()}
      >
        Import
      </Button>
    </div>
  )

  return (
    <PageContainer title="Agents" headerRight={headerRight}>
      {/* Stats */}
      <KpiGrid columns={3}>
        <StatCard label="Total Agents" value={agents.length} />
        <StatCard label="Tool Assignments" value={toolCount} />
        <StatCard label="Available Tools" value={AVAILABLE_TOOLS.length} />
      </KpiGrid>

      {/* Create */}
      <NewAgentCard {...wb} />

      {/* Roster */}
      <AgentsListCard {...wb} />

      {/* Orchestration */}
      <OrchestrationCard {...wb} />

      {/* Run history */}
      <RunHistoryCard {...wb} />

      {/* Delete confirmation */}
      <DeleteAgentDialog {...wb} />
    </PageContainer>
  )
}
