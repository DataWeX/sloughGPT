'use client'

import { useState, useCallback , useEffect} from 'react'
import { ConfirmDialog } from '@/components/ConfirmDialog'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label, Textarea } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { useToastStore } from '@/lib/toast-store'
import { sessionController, type SessionInspector } from '@/lib/session-controller'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'

export default function SessionPage() {
  const addToast = useToastStore(s => s.addToast)
  const [sessionId, setSessionId] = useState('')

  const [currentSession, setCurrentSession] = useState<Record<string, unknown> | null>(null)
  const [fetchedMessages, setFetchedMessages] = useState<Array<{ role: string; content: string }> | null>(null)
  const [inspector, setInspector] = useState<SessionInspector | null>(null)
  const [loading, setLoading] = useState(false)
  const [regenerating, setRegenerating] = useState(false)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null)

  const handleInspect = useCallback(async () => {
    if (!sessionId.trim()) return
    setLoading(true)
    setInspector(null)
    try {
      const result = await sessionController.getInspector(sessionId)
      setInspector(result)
    } catch (e) {
      addToast(`Could not load session: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setLoading(false)
    }
  }, [sessionId, addToast])

  const handleRegenerate = useCallback(async () => {
    if (!sessionId.trim()) return
    setRegenerating(true)
    try {
      await sessionController.regenerate(sessionId)
      addToast('Regeneration started', 'success')
      void handleInspect()
    } catch (e) {
      addToast(`Regeneration failed: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setRegenerating(false)
    }
  }, [sessionId, addToast, handleInspect])


  const [sessions, setSessions] = useState<{ id: string; name?: string; created_at?: string }[]>([])
  const [sessionsLoading, setSessionsLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<{ id: string; name?: string; created_at?: string }[] | null>(null)
  const [showCreate, setShowCreate] = useState(false)
  const [createSoul, setCreateSoul] = useState('')

  const loadSessions = useCallback(async () => {
    setSessionsLoading(true)
    try {
      const items = await sessionController.list()
      setSessions(items.slice(0, 50))
    } catch { /* silent */ }
    finally { setSessionsLoading(false) }
  }, [])

  useRefreshShortcut(() => void loadSessions())

  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) { setSearchResults(null); return }
    try {
      const items = await sessionController.search(searchQuery, 20)
      setSearchResults(items)
    } catch {
      addToast('Could not search sessions', 'error')
    }
  }, [searchQuery, addToast])

  const handleCreate = useCallback(async () => {
    try {
      const resp = await sessionController.create(createSoul || 'new session')
      addToast('Session created', 'success')
      setSessionId(resp.id)
      setShowCreate(false)
      void loadSessions()
      void handleInspect()
    } catch {
      addToast('Could not create session', 'error')
    }
  }, [createSoul, addToast, loadSessions, handleInspect])

  const handleDelete = useCallback(async (id: string) => {
    setPendingDeleteId(id)
    setConfirmOpen(true)
  }, [])

  const confirmDelete = useCallback(async () => {
    if (!pendingDeleteId) return
    try {
      await sessionController.delete(pendingDeleteId)
      addToast('Session deleted', 'success')
      if (sessionId === pendingDeleteId) { setInspector(null); setSessionId('') }
      void loadSessions()
    } catch {
      addToast('Could not delete session', 'error')
    } finally {
      setConfirmOpen(false)
      setPendingDeleteId(null)
    }
  }, [pendingDeleteId, sessionId, addToast, loadSessions])

  const handleLoadArchived = useCallback(async () => {
    try {
      const items = await sessionController.listArchived()
      setSessions(items.slice(0, 50))
    } catch {
      addToast('Could not load archived sessions', 'error')
    }
  }, [addToast])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'r' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); void loadSessions() }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [loadSessions])

  return (
    <PageContainer
      title="Session Inspector"
      subtitle="Debug and inspect chat session state"
      headerRight={
        <div className="flex gap-1">
          <Button size="sm" variant="ghost" onClick={() => { void loadSessions(); setShowCreate(!showCreate) }}>Sessions</Button>
          <Button size="sm" variant="ghost" onClick={() => void handleLoadArchived()}>Archived</Button>
          <Button size="sm" variant="ghost" onClick={async () => { try { const s = await sessionController.getCurrent(); setCurrentSession(s as unknown as Record<string, unknown>) } catch { /* */ } }}>Current</Button>
          <Button size="sm" variant="ghost" onClick={async () => { if (!sessionId.trim()) return; try { const m = await sessionController.fetchMessages(sessionId); setFetchedMessages(m) } catch { addToast('Could not fetch messages', 'error') } }}>Messages</Button>
          {inspector && <Button size="sm" variant="ghost" onClick={() => void handleInspect()}>Refresh</Button>}
        </div>
      }
    >
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Load Session</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="space-y-2">
              <Label className="text-xs">Session ID</Label>
              <Input
                value={sessionId}
                onChange={e => setSessionId(e.target.value)}
                placeholder="Enter session ID..."
                className="h-8 text-xs font-mono"
                onKeyDown={e => e.key === 'Enter' && void handleInspect()}
              />
            </div>
            <div className="flex gap-2">
              <Button onClick={() => void handleInspect()} disabled={loading || !sessionId.trim()} className="flex-1">
                {loading ? 'Loading...' : 'Inspect'}
              </Button>
              <Button onClick={() => void handleRegenerate()} disabled={regenerating || !sessionId.trim()} variant="outline" className="flex-1">
                {regenerating ? 'Regenerating...' : 'Regenerate Last Response'}
              </Button>
            </div>
          </CardContent>
        </Card>

  
      {showCreate && (
        <Card>
          <CardContent className="p-3 flex items-center gap-2">
            <Input value={createSoul} onChange={e => setCreateSoul(e.target.value)} placeholder="Soul name (optional)" className="h-8 text-xs flex-1" />
            <Button size="sm" onClick={() => void handleCreate()}>Create Session</Button>
          </CardContent>
        </Card>
      )}

      {(sessions.length > 0 || searchResults) && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center justify-between">
              Sessions ({(searchResults ?? sessions).length})
              <div className="flex gap-1">
                <Input value={searchQuery} onChange={e => setSearchQuery(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') void handleSearch() }} placeholder="Search..." className="h-7 text-xs w-40" />
                {searchResults && <Button size="sm" variant="ghost" onClick={() => { setSearchResults(null); setSearchQuery('') }}>Clear</Button>}
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1 max-h-60 overflow-auto">
              {(searchResults ?? sessions).map(s => (
                <div key={s.id} className="flex items-center justify-between rounded px-2 py-1 hover:bg-muted/30 transition-colors text-xs">
                  <button type="button" className="text-left flex-1 truncate font-mono" onClick={() => { setSessionId(s.id); void handleInspect() }}>
                    {s.name || s.id}{s.created_at ? ` — ${new Date(s.created_at).toLocaleDateString()}` : ''}
                  </button>
                  <Button variant="ghost" size="sm" className="text-[10px] h-6 text-destructive" onClick={() => void handleDelete(s.id)}>Delete</Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {inspector && (
          <>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                { label: 'Messages', value: inspector.session.message_count },
                { label: 'Knowledge Facts', value: inspector.knowledge.total_facts },
                { label: 'Feedback', value: inspector.feedback.total },
                { label: 'Inspect Time', value: `${inspector.elapsed_ms}ms` },
              ].map(s => (
                <div key={s.label} className="rounded-md bg-muted/30 p-3 text-center">
                  <div className="text-[10px] text-muted-foreground">{s.label}</div>
                  <div className="text-lg font-mono font-medium">{s.value}</div>
                </div>
              ))}
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Workspace</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid grid-cols-2 gap-2">
                    <div className="rounded bg-muted/30 p-2 text-center">
                      <div className="text-[10px] text-muted-foreground">Episodic Memory</div>
                      <div className="text-xs font-mono">{inspector.workspace.episodic_count}</div>
                    </div>
                    <div className="rounded bg-muted/30 p-2 text-center">
                      <div className="text-[10px] text-muted-foreground">Sensory Buffer</div>
                      <div className="text-xs font-mono">{inspector.workspace.sensory_buffer_size}</div>
                    </div>
                  </div>
                  {inspector.workspace.working_memory.length > 0 && (
                    <div>
                      <p className="text-[10px] text-muted-foreground mb-1">Working Memory</p>
                      <div className="flex flex-wrap gap-1">
                        {inspector.workspace.working_memory.map((m, i) => (
                          <span key={i} className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary">{m}</span>
                        ))}
                      </div>
                    </div>
                  )}
                  {inspector.workspace.semantic_keys.length > 0 && (
                    <div>
                      <p className="text-[10px] text-muted-foreground mb-1">Semantic Keys</p>
                      <div className="flex flex-wrap gap-1">
                        {inspector.workspace.semantic_keys.map((k, i) => (
                          <span key={i} className="rounded bg-muted px-1.5 py-0.5 text-[10px]">{k}</span>
                        ))}
                      </div>
                    </div>
                  )}
                  {inspector.workspace.system_prompt && (
                    <div>
                      <p className="text-[10px] text-muted-foreground mb-1">System Prompt (truncated)</p>
                      <pre className="max-h-[100px] overflow-y-auto rounded bg-muted/30 p-2 text-[10px] whitespace-pre-wrap">{inspector.workspace.system_prompt}</pre>
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Modes & Traits</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {Object.keys(inspector.modes).length > 0 && (
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(inspector.modes).map(([k, v]) => (
                        <div key={k} className="rounded bg-muted/30 p-2">
                          <div className="text-[10px] text-muted-foreground capitalize">{k}</div>
                          <div className="text-xs font-medium">{v}</div>
                        </div>
                      ))}
                    </div>
                  )}
                  {Object.keys(inspector.traits).length > 0 && (
                    <div>
                      <p className="text-[10px] text-muted-foreground mb-1">Traits</p>
                      <pre className="max-h-[150px] overflow-y-auto rounded bg-muted/30 p-2 text-[10px]">{JSON.stringify(inspector.traits, null, 2)}</pre>
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Knowledge</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="rounded bg-muted/30 p-2 text-center">
                    <div className="text-[10px] text-muted-foreground">Total Facts</div>
                    <div className="text-lg font-mono font-medium">{inspector.knowledge.total_facts}</div>
                  </div>
                  {inspector.knowledge.topics.length > 0 && (
                    <div>
                      <p className="text-[10px] text-muted-foreground mb-1">Topics</p>
                      <div className="flex flex-wrap gap-1">
                        {inspector.knowledge.topics.map((t, i) => (
                          <span key={i} className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary">{t}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Feedback</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid grid-cols-2 gap-2">
                    <div className="rounded bg-muted/30 p-2 text-center">
                      <div className="text-[10px] text-muted-foreground">Thumbs Up</div>
                      <div className="text-lg font-mono font-medium text-green-500">{inspector.feedback.thumbs_up}</div>
                    </div>
                    <div className="rounded bg-muted/30 p-2 text-center">
                      <div className="text-[10px] text-muted-foreground">Thumbs Down</div>
                      <div className="text-lg font-mono font-medium text-red-500">{inspector.feedback.thumbs_down}</div>
                    </div>
                  </div>
                  <div className="rounded bg-muted/30 p-2 text-center">
                    <div className="text-[10px] text-muted-foreground">Approval Rate</div>
                    <div className="text-sm font-mono font-medium">
                      {inspector.feedback.total > 0
                        ? `${((inspector.feedback.thumbs_up / inspector.feedback.total) * 100).toFixed(1)}%`
                        : 'N/A'}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {inspector.session.messages.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Recent Messages ({inspector.session.messages.length})</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 max-h-[300px] overflow-y-auto">
                    {inspector.session.messages.map((msg, i) => (
                      <div key={i} className={`rounded p-2 text-xs ${msg.role === 'assistant' ? 'bg-primary/5' : 'bg-muted/30'}`}>
                        <div className="flex items-center gap-2 mb-1">
                          <span className={`text-[10px] font-medium ${msg.role === 'assistant' ? 'text-primary' : 'text-muted-foreground'}`}>
                            {msg.role}
                          </span>
                        </div>
                        <p className="whitespace-pre-wrap">{msg.content.slice(0, 500)}{msg.content.length > 500 ? '...' : ''}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </>
        )}
      </div>

      {currentSession && (
        <Card>
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <CardTitle className="text-base">Current Session</CardTitle>
            <Button size="sm" variant="ghost" onClick={() => setCurrentSession(null)}>Close</Button>
          </CardHeader>
          <CardContent>
            <pre className="rounded bg-muted p-3 text-xs overflow-auto max-h-80 whitespace-pre-wrap">{JSON.stringify(currentSession, null, 2)}</pre>
          </CardContent>
        </Card>
      )}
      {fetchedMessages && (
        <Card>
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <CardTitle className="text-base">Messages ({fetchedMessages.length})</CardTitle>
            <Button size="sm" variant="ghost" onClick={() => setFetchedMessages(null)}>Close</Button>
          </CardHeader>
          <CardContent>
            <div className="space-y-1 max-h-80 overflow-auto">
              {fetchedMessages.map((m, i) => (
                <div key={i} className="rounded bg-muted/30 px-2 py-1 text-xs">
                  <span className="font-medium">{m.role}:</span> {m.content}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <ConfirmDialog
        open={confirmOpen}
        onOpenChange={setConfirmOpen}
        title="Delete session"
        description="Delete this session? This cannot be undone."
        confirmLabel="Delete"
        onConfirm={() => void confirmDelete()}
      />
    </PageContainer>
  )
}
