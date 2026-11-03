'use client'

/**
 * Agent roster: search, bulk actions, inline edit and per-agent execute.
 */
import type { AgentsWorkbench } from './useAgentsWorkbench'
import { AVAILABLE_TOOLS } from './agents-constants'
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Checkbox,
  EmptyCard,
  IconCopy,
  IconRefresh,
  Input,
  Skeleton,
  cn,
} from '@sloughgpt/strui'
import { IconPlus, IconTrash } from '@/components/icons/NavIcons'

export function AgentsListCard({
  agents,
  loading,
  router,
  editingId,
  editName,
  setEditName,
  editDesc,
  setEditDesc,
  editInstructions,
  setEditInstructions,
  editTools,
  setEditTools,
  execAgentId,
  setExecAgentId,
  execPrompt,
  setExecPrompt,
  execResult,
  setExecResult,
  execToolsUsed,
  setPendingDelete,
  selectedIds,
  execRunning,
  agentSearch,
  setAgentSearch,
  editErrors,
  execErrors,
  fetchAgents,
  toggleTool,
  startEdit,
  cancelEdit,
  handleUpdate,
  handleExecute,
  toggleSelect,
  selectAll,
  handleBulkDelete,
  handleBulkExport,
  handleClone,
  filteredAgents,
  setEditErrors,
  setExecErrors,
}: {
  agents: AgentsWorkbench['agents']
  loading: AgentsWorkbench['loading']
  router: AgentsWorkbench['router']
  editingId: AgentsWorkbench['editingId']
  editName: AgentsWorkbench['editName']
  setEditName: AgentsWorkbench['setEditName']
  editDesc: AgentsWorkbench['editDesc']
  setEditDesc: AgentsWorkbench['setEditDesc']
  editInstructions: AgentsWorkbench['editInstructions']
  setEditInstructions: AgentsWorkbench['setEditInstructions']
  editTools: AgentsWorkbench['editTools']
  setEditTools: AgentsWorkbench['setEditTools']
  execAgentId: AgentsWorkbench['execAgentId']
  setExecAgentId: AgentsWorkbench['setExecAgentId']
  execPrompt: AgentsWorkbench['execPrompt']
  setExecPrompt: AgentsWorkbench['setExecPrompt']
  execResult: AgentsWorkbench['execResult']
  setExecResult: AgentsWorkbench['setExecResult']
  execToolsUsed: AgentsWorkbench['execToolsUsed']
  setPendingDelete: AgentsWorkbench['setPendingDelete']
  selectedIds: AgentsWorkbench['selectedIds']
  execRunning: AgentsWorkbench['execRunning']
  agentSearch: AgentsWorkbench['agentSearch']
  setAgentSearch: AgentsWorkbench['setAgentSearch']
  editErrors: AgentsWorkbench['editErrors']
  execErrors: AgentsWorkbench['execErrors']
  fetchAgents: AgentsWorkbench['fetchAgents']
  toggleTool: AgentsWorkbench['toggleTool']
  startEdit: AgentsWorkbench['startEdit']
  cancelEdit: AgentsWorkbench['cancelEdit']
  handleUpdate: AgentsWorkbench['handleUpdate']
  handleExecute: AgentsWorkbench['handleExecute']
  toggleSelect: AgentsWorkbench['toggleSelect']
  selectAll: AgentsWorkbench['selectAll']
  handleBulkDelete: AgentsWorkbench['handleBulkDelete']
  handleBulkExport: AgentsWorkbench['handleBulkExport']
  handleClone: AgentsWorkbench['handleClone']
  filteredAgents: AgentsWorkbench['filteredAgents']
  setEditErrors: AgentsWorkbench['setEditErrors']
  setExecErrors: AgentsWorkbench['setExecErrors']
}) {
  return (
    <Card>
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <div className="flex items-center justify-between">
          <CardTitle className="text-[11px] font-medium">Agents</CardTitle>
          <Button
            size="sm"
            variant="ghost"
            onClick={fetchAgents}
            disabled={loading}
            className="h-6 text-[10px]"
          >
            <IconRefresh className="h-4 w-4 mr-1" />
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent className="px-2.5 pb-2.5 space-y-2">
        {loading ? (
          <div className="space-y-2">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="flex items-center gap-1.5 p-2.5 rounded-lg border border-border/40"
              >
                <Skeleton className="h-6 w-6 rounded shrink-0 bg-muted/20" />
                <div className="flex-1 space-y-1">
                  <Skeleton className="h-3.5 w-24 bg-muted/20 rounded-lg" />
                  <Skeleton className="h-3 w-40 bg-muted/20 rounded-lg" />
                </div>
                <Skeleton className="h-6 w-14 rounded-full bg-muted/20" />
              </div>
            ))}
          </div>
        ) : agents.length === 0 ? (
          <EmptyCard
            message="No agents yet"
            description="Create your first agent above. Use templates for quick setup."
            icon={<IconPlus className="h-5 w-5" />}
            action={null}
          />
        ) : (
          <>
            {agents.length > 2 && (
              <input
                type="text"
                aria-label="Search agents"
                value={agentSearch}
                onChange={(e) => setAgentSearch(e.target.value)}
                placeholder="Search agents..."
                className="h-7 w-full max-w-xs rounded-md border border-border/40 bg-background px-2.5 text-[11px] placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary/30"
              />
            )}
            {filteredAgents.length > 0 && (
              <div className="flex items-center gap-2 text-[10px]">
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <Checkbox
                    checked={
                      selectedIds.size === filteredAgents.length && filteredAgents.length > 0
                    }
                    onCheckedChange={selectAll}
                    aria-label="Select all agents"
                    className="rounded border-border"
                  />
                  <span className="text-muted-foreground">Select all</span>
                </label>
                {selectedIds.size > 0 && (
                  <>
                    <span className="text-muted-foreground">·</span>
                    <span className="text-muted-foreground">{selectedIds.size} selected</span>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={handleBulkExport}
                      className="h-6 text-[10px]"
                    >
                      Export
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={handleBulkDelete}
                      className="h-6 text-[10px] text-destructive hover:text-destructive"
                    >
                      Delete
                    </Button>
                  </>
                )}
              </div>
            )}
            {filteredAgents.length === 0 ? (
              <div className="text-center py-4">
                <p className="text-[10px] text-muted-foreground/60">
                  No agents matching &quot;{agentSearch}&quot;
                </p>
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-6 text-[10px] mt-2"
                  onClick={() => setAgentSearch('')}
                >
                  Clear search
                </Button>
              </div>
            ) : (
              filteredAgents.map((agent) => (
                <div
                  key={agent.id}
                  className={cn(
                    'rounded-lg border space-y-2 transition-colors',
                    execAgentId === agent.id
                      ? 'p-2.5 bg-primary/[0.08] border-primary/40'
                      : 'border-border/40 p-2.5 hover:bg-muted/20',
                  )}
                >
                  {editingId === agent.id ? (
                    <div className="space-y-2">
                      <div>
                        <Input
                          id="edit-agent-name"
                          value={editName}
                          onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                            setEditName(e.target.value)
                            if (editErrors.name)
                              setEditErrors((prev) => ({ ...prev, name: undefined }))
                          }}
                          placeholder="Name"
                          aria-label="Agent name"
                          className={
                            editErrors.name ? 'border-destructive ring-destructive/20' : ''
                          }
                          aria-invalid={!!editErrors.name}
                          aria-describedby={editErrors.name ? 'edit-name-error' : undefined}
                        />
                        {editErrors.name && (
                          <p
                            id="edit-name-error"
                            className="text-[10px] text-destructive mt-1"
                            role="alert"
                          >
                            {editErrors.name}
                          </p>
                        )}
                      </div>
                      <Input
                        value={editDesc}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                          setEditDesc(e.target.value)
                        }
                        placeholder="Description"
                        aria-label="Agent description"
                      />
                      <textarea
                        className="w-full rounded-lg border border-input bg-background px-3 py-2 text-[11px] placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring min-h-[60px]"
                        value={editInstructions}
                        onChange={(e) => setEditInstructions(e.target.value)}
                        aria-label="Edit agent instructions"
                      />
                      <div className="flex flex-wrap gap-1">
                        {AVAILABLE_TOOLS.map((t) => (
                          <button
                            key={t}
                            type="button"
                            onClick={() => toggleTool(t, editTools, setEditTools)}
                            aria-pressed={editTools.includes(t)}
                            className={cn(
                              'rounded-full px-3 py-1 text-[10px] font-medium border transition-colors',
                              editTools.includes(t)
                                ? 'bg-primary text-primary-foreground border-primary'
                                : 'bg-background text-muted-foreground border-border hover:border-primary/50',
                            )}
                          >
                            {t.replace(/_/g, ' ')}
                          </button>
                        ))}
                      </div>
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          onClick={() => handleUpdate(agent.id)}
                          className="h-7 text-[11px]"
                        >
                          Save
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={cancelEdit}
                          className="h-6 text-[10px]"
                        >
                          Cancel
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-2 min-w-0 flex-1">
                          <Checkbox
                            checked={selectedIds.has(agent.id)}
                            onCheckedChange={() => toggleSelect(agent.id)}
                            aria-label={`Select agent ${agent.name}`}
                            className="mt-1 rounded border-border"
                          />
                          <div className="min-w-0 flex-1">
                            <p
                              className="text-[11px] font-medium cursor-pointer hover:text-primary transition-colors"
                              onClick={() => router.push(`/agents/${agent.id}`)}
                            >
                              {agent.name}
                            </p>
                            {agent.description && (
                              <p className="text-[10px] text-muted-foreground mt-0.5">
                                {agent.description}
                              </p>
                            )}
                          </div>
                        </div>
                        <div className="flex gap-1 shrink-0 ml-2">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => handleClone(agent)}
                            aria-label={`Clone ${agent.name}`}
                            className="h-6 text-[10px]"
                          >
                            <IconCopy className="h-4 w-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => startEdit(agent)}
                            className="h-6 text-[10px]"
                          >
                            Edit
                          </Button>
                          <Button
                            size="sm"
                            variant="destructive"
                            onClick={() => setPendingDelete(agent)}
                            aria-label={`Delete ${agent.name}`}
                            className="h-6 text-[10px]"
                          >
                            <IconTrash className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                      {agent.tools.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {agent.tools.map((t) => (
                            <span
                              key={t}
                              className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium"
                            >
                              {t.replace(/_/g, ' ')}
                            </span>
                          ))}
                        </div>
                      )}
                      {agent.instructions && (
                        <p className="text-[10px] text-muted-foreground/70 line-clamp-2">
                          {agent.instructions}
                        </p>
                      )}
                      {/* Inline execute */}
                      <div className="pt-1">
                        {execAgentId === agent.id ? (
                          <div className="space-y-2">
                            <div>
                              <Input
                                id="exec-prompt"
                                placeholder="What should this agent do?"
                                value={execPrompt}
                                onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                                  setExecPrompt(e.target.value)
                                  if (execErrors.prompt) setExecErrors({})
                                }}
                                onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                                  if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault()
                                    handleExecute(agent.id)
                                  }
                                }}
                                className={
                                  execErrors.prompt ? 'border-destructive ring-destructive/20' : ''
                                }
                                aria-invalid={!!execErrors.prompt}
                                aria-describedby={
                                  execErrors.prompt ? 'exec-prompt-error' : undefined
                                }
                              />
                              {execErrors.prompt && (
                                <p
                                  id="exec-prompt-error"
                                  className="text-[10px] text-destructive mt-1"
                                  role="alert"
                                >
                                  {execErrors.prompt}
                                </p>
                              )}
                            </div>
                            <div className="flex gap-2">
                              <Button
                                size="sm"
                                onClick={() => handleExecute(agent.id)}
                                disabled={execRunning || !execPrompt.trim()}
                              >
                                {execRunning ? 'Running...' : 'Execute'}
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => {
                                  setExecAgentId(null)
                                  setExecResult(null)
                                  setExecPrompt('')
                                }}
                                className="h-6 text-[10px]"
                              >
                                Close
                              </Button>
                            </div>
                            {execResult && (
                              <div className="rounded-lg bg-muted p-3">
                                {execToolsUsed.length > 0 && (
                                  <div className="mb-2 flex flex-wrap gap-1">
                                    {execToolsUsed.map((t, i) => (
                                      <span
                                        key={i}
                                        className="text-[9px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium"
                                      >
                                        {t.tool.replace(/_/g, ' ')}
                                      </span>
                                    ))}
                                  </div>
                                )}
                                <p className="text-[10px] font-medium text-muted-foreground mb-1">
                                  Response
                                </p>
                                <p className="text-[11px] whitespace-pre-wrap">{execResult}</p>
                              </div>
                            )}
                          </div>
                        ) : (
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => {
                              setExecAgentId(agent.id)
                              setExecPrompt('')
                              setExecResult(null)
                            }}
                          >
                            Run
                          </Button>
                        )}
                      </div>
                    </>
                  )}
                </div>
              ))
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}
