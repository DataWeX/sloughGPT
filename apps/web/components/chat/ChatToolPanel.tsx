'use client'

import { useState, useRef, useEffect } from 'react'

import { Button } from '@/components/ui/button'
import { IconX, IconBrain, IconEye, IconHeart, IconSettings, IconDocument, IconModel, IconCheck, IconRefresh } from '@/components/ui'
import { cn } from '@/lib/cn'
import { KNOWLEDGE_STORAGE_KEY } from '@/lib/config'

type TabId = 'knowledge' | 'vision' | 'learner' | 'checkpoints' | 'agents'

interface KnowledgeItem {
  id: string
  content: string
  timestamp: number
}

interface Checkpoint {
  name: string
  loss?: number
  traits?: string[]
  is_loaded?: boolean
  eval_verdict?: string
}

interface LearnerInfo {
  total_tokens_ingested: number
  train_steps_completed: number
  current_loss?: number
  loss_history?: Array<{ step: number; loss: number; tokens: number; timestamp: number }>
  n_embed?: number
  n_layer?: number
  arch?: string
}

interface Agent {
  id: string
  name: string
  description?: string
  instructions: string
}

interface ChatToolPanelProps {
  open: boolean
  onClose: () => void
  // Vision
  visionImagesLearned?: number
  visionTrained?: boolean
  visionStatus?: string
  visionCaptionHistory?: string[]
  visionVocabSize?: number
  // Learner
  learnerInfo: LearnerInfo | null
  learnerTraining: boolean
  onTrainStep: () => Promise<void>
  // Checkpoints
  checkpoints: Checkpoint[]
  onLoadCheckpoint?: (name: string) => Promise<void>
  currentCheckpoint?: string
  // Agents
  agents: Agent[]
  currentAgent: Agent | null
  onSelectAgent: (agent: Agent | null) => void
  // Models
  availableModels: string[]
  currentModel: string
  onSelectModel: (model: string) => void
  modelInfoMap?: Record<string, { cached?: boolean; size_gb?: number }>
  // Personality
  souls: { name: string; traits?: string[]; description?: string }[]
  currentSoulName?: string
  onSwitchSoul?: (name: string) => void
  // General
  onOpenSettings: () => void
  onOpenShortcuts: () => void
  onOpenConversationViewer: () => void
}

const tabs: { id: TabId; label: string; icon: typeof IconBrain }[] = [
  { id: 'knowledge', label: 'Knowledge', icon: IconDocument },
  { id: 'vision', label: 'Vision', icon: IconEye },
  { id: 'learner', label: 'Learner', icon: IconBrain },
  { id: 'agents', label: 'Agents', icon: IconHeart },
  { id: 'checkpoints', label: 'Models', icon: IconModel },
]

export function ChatToolPanel({
  open,
  onClose,
  learnerInfo,
  learnerTraining,
  onTrainStep,
  checkpoints,
  onLoadCheckpoint,
  currentCheckpoint,
  agents,
  currentAgent,
  onSelectAgent,
  availableModels,
  currentModel,
  onSelectModel,
  modelInfoMap,
  souls,
  currentSoulName,
  onSwitchSoul,
  onOpenSettings,
  onOpenShortcuts,
  onOpenConversationViewer,
  visionImagesLearned,
  visionTrained,
  visionStatus,
  visionCaptionHistory,
  visionVocabSize,
}: ChatToolPanelProps) {
  const [activeTab, setActiveTab] = useState<TabId>('knowledge')
  const [knowledge, setKnowledge] = useState<KnowledgeItem[]>(() => {
    if (typeof window === 'undefined') return []
    try {
      const raw = localStorage.getItem(KNOWLEDGE_STORAGE_KEY)
      return raw ? JSON.parse(raw) : []
    } catch { return [] }
  })
  const [showAddKnowledge, setShowAddKnowledge] = useState(false)
  const [newKnowledge, setNewKnowledge] = useState('')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editText, setEditText] = useState('')
  const [backendContentIds, setBackendContentIds] = useState<Map<string, string>>(new Map())
  const syncTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Fetch backend knowledge on mount and merge with local items
  useEffect(() => {
    if (!open) return
    let cancelled = false
    import('@/lib/knowledge-controller').then(({ knowledgeController }) => {
      knowledgeController.list().then(backendItems => {
        if (cancelled) return
        const idMap = new Map<string, string>()
        for (const item of backendItems) {
          idMap.set(item.content, item.id)
        }
        setBackendContentIds(idMap)
        // Merge — add backend items not already in localStorage
        const raw = localStorage.getItem(KNOWLEDGE_STORAGE_KEY)
        const local: KnowledgeItem[] = raw ? JSON.parse(raw) : []
        const existingContent = new Set(local.map(k => k.content))
        const needsSave = backendItems.some(item => !existingContent.has(item.content))
        if (needsSave) {
          for (const item of backendItems) {
            if (!existingContent.has(item.content)) {
              local.push({
                id: `know_b_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
                content: item.content,
                timestamp: Date.now(),
              })
            }
          }
          setKnowledge(local)
          localStorage.setItem(KNOWLEDGE_STORAGE_KEY, JSON.stringify(local))
        }
      }).catch(() => {})
    }).catch(() => {})
    return () => { cancelled = true }
  }, [open])

  const syncToBackend = (items: KnowledgeItem[]) => {
    if (syncTimerRef.current) clearTimeout(syncTimerRef.current)
    syncTimerRef.current = setTimeout(async () => {
      try {
        const { knowledgeController } = await import('@/lib/knowledge-controller')
        await knowledgeController.batchIngest(
          items.map(k => ({ content: k.content, source: 'injected' }))
        )
        // Re-fetch to learn backend IDs for future deletes
        const updated = await knowledgeController.list()
        const idMap = new Map<string, string>()
        for (const item of updated) idMap.set(item.content, item.id)
        setBackendContentIds(idMap)
      } catch {
        // offline — fine, next sync will catch up
      }
    }, 2000)
  }

  const saveKnowledge = (items: KnowledgeItem[]) => {
    setKnowledge(items)
    localStorage.setItem(KNOWLEDGE_STORAGE_KEY, JSON.stringify(items))
    syncToBackend(items)
  }

  const addKnowledge = () => {
    if (!newKnowledge.trim()) return
    const item: KnowledgeItem = {
      id: `know_${Date.now()}`,
      content: newKnowledge.trim(),
      timestamp: Date.now(),
    }
    saveKnowledge([...knowledge, item])
    setNewKnowledge('')
    setShowAddKnowledge(false)
  }

  const removeKnowledge = (id: string) => {
    const item = knowledge.find(k => k.id === id)
    saveKnowledge(knowledge.filter(k => k.id !== id))
    // Also remove from backend if synced
    if (item) {
      const backendId = backendContentIds.get(item.content)
      if (backendId) {
        import('@/lib/knowledge-controller').then(({ knowledgeController }) => {
          knowledgeController.delete(backendId).catch(() => {})
        }).catch(() => {})
      }
    }
  }

  return (
    <div
      id="chat-tool-panel"
      className={cn(
        'border-l border-border/50 bg-background overflow-hidden transition-all duration-200 flex flex-col',
        open ? 'w-72 min-w-[16rem]' : 'w-0 min-w-0',
      )}
    >
      {open && (
        <>
          {/* Header */}
          <div className="flex items-center justify-between px-3 py-2 border-b border-border/50 shrink-0">
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              {tabs.find(t => t.id === activeTab)?.label}
            </span>
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onClose} aria-label="Close tools panel">
              <IconX className="h-3.5 w-3.5" />
            </Button>
          </div>

          {/* Tab bar */}
          <div className="flex border-b border-border/50 shrink-0" role="tablist" aria-label="Tools">
            {tabs.map(tab => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  role="tab"
                  aria-selected={activeTab === tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={cn(
                    'flex-1 flex flex-col items-center gap-0.5 py-1.5 text-[10px] font-medium transition-colors',
                    activeTab === tab.id
                      ? 'text-foreground border-b-2 border-primary'
                      : 'text-muted-foreground hover:text-foreground',
                  )}
                  title={tab.label}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {tab.label}
                </button>
              )
            })}
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-3 space-y-3 text-xs scrollbar-thin">
            {activeTab === 'knowledge' && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">
                    {knowledge.length} snippet{knowledge.length !== 1 ? 's' : ''}
                  </span>
                  <Button variant="outline" size="sm" className="h-6 text-[10px] px-2" onClick={() => setShowAddKnowledge(true)}>
                    + Add
                  </Button>
                </div>

                {showAddKnowledge && (
                  <div className="space-y-1">
                    <textarea
                      className="w-full p-2 text-xs border border-input rounded-lg resize-none h-16 bg-background focus:outline-none focus:ring-1 focus:ring-primary/40"
                      placeholder="Enter a fact the AI should know..."
                      value={newKnowledge}
                      onChange={e => setNewKnowledge(e.target.value)}
                      autoFocus
                    />
                    <div className="flex gap-1">
                      <Button size="sm" className="h-6 text-[10px] flex-1" onClick={addKnowledge}>Save</Button>
                      <Button variant="outline" size="sm" className="h-6 text-[10px]" onClick={() => { setShowAddKnowledge(false); setNewKnowledge('') }}>Cancel</Button>
                    </div>
                  </div>
                )}

                {knowledge.length === 0 ? (
                  <p className="text-xs text-muted-foreground text-center py-4">
                    No knowledge stored. Add facts the AI should reference.
                  </p>
                ) : (
                  <ul className="space-y-1 max-h-60 overflow-y-auto">
                    {knowledge.map((item) => (
                      <li key={item.id} className="p-2 rounded bg-muted/30 border border-border/40 text-xs leading-relaxed group relative">
                        {editingId === item.id ? (
                          <div className="space-y-1">
              <textarea
                  value={editText}
                  onChange={(e) => setEditText(e.target.value)}
                  className="w-full h-16 resize-none rounded border border-border/60 bg-muted/30 p-2 text-xs focus:outline-none focus:ring-1 focus:ring-primary/40"
                  aria-label="Edit knowledge snippet"
                />
                            <div className="flex gap-1">
                              <Button size="sm" className="h-5 text-[10px] px-2 flex-1" onClick={() => {
                                saveKnowledge(knowledge.map(k => k.id === item.id ? { ...k, content: editText } : k))
                                setEditingId(null)
                              }}>Save</Button>
                              <Button variant="outline" size="sm" className="h-5 text-[10px] px-2" onClick={() => setEditingId(null)}>Cancel</Button>
                            </div>
                          </div>
                        ) : (
                          <>
                            <span>{item.content.length > 200 ? item.content.slice(0, 200) + '...' : item.content}</span>
                            <span className="ml-1 text-[9px] text-muted-foreground/50">
                              {backendContentIds.has(item.content) ? '✓' : '…'}
                            </span>
                            <div className="absolute top-1 right-1 flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                              <button
                                onClick={() => { setEditingId(item.id); setEditText(item.content) }}
                                className="text-muted-foreground hover:text-foreground p-0.5"
                                aria-label="Edit knowledge"
                              >
                                <svg className="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" /></svg>
                              </button>
                              <button
                                onClick={() => removeKnowledge(item.id)}
                                className="text-muted-foreground hover:text-destructive p-0.5"
                                aria-label="Remove knowledge"
                              >
                                <IconX className="h-3 w-3" />
                              </button>
                            </div>
                          </>
                        )}
                      </li>
                    ))}
                  </ul>
                )}

                {knowledge.length > 0 && (
                  <button
                    onClick={() => { saveKnowledge([]) }}
                    className="text-[10px] text-muted-foreground hover:text-destructive transition-colors"
                  >
                    Clear all
                  </button>
                )}
              </div>
            )}

            {activeTab === 'vision' && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium">Vision Model</span>
                  <span className={cn(
                    'inline-block h-1.5 w-1.5 rounded-full',
                    visionTrained ? 'bg-success' : (visionImagesLearned || 0) > 0 ? 'bg-warning' : 'bg-muted-foreground/30',
                  )} />
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div className="p-2 rounded bg-muted/30 border border-border/40">
                    <div className="text-[10px] text-muted-foreground">Images learned</div>
                    <div className="text-sm font-medium">{visionImagesLearned ?? 0}</div>
                  </div>
                  <div className="p-2 rounded bg-muted/30 border border-border/40">
                    <div className="text-[10px] text-muted-foreground">Status</div>
                    <div className="text-sm font-medium capitalize">{visionStatus || 'ready'}</div>
                  </div>
                  {visionVocabSize !== undefined && (
                    <div className="p-2 rounded bg-muted/30 border border-border/40">
                      <div className="text-[10px] text-muted-foreground">Vocabulary</div>
                      <div className="text-sm font-medium">{visionVocabSize} words</div>
                    </div>
                  )}
                </div>

                {visionCaptionHistory && visionCaptionHistory.length > 0 && (
                  <div className="space-y-1">
                    <div className="text-[10px] text-muted-foreground font-medium">Recent captions</div>
                    <ul className="space-y-1 max-h-32 overflow-y-auto">
                      {visionCaptionHistory.slice(-10).map((cap, i) => (
                        <li key={i} className="p-1.5 rounded bg-muted/20 text-[10px] leading-relaxed border border-border/20">
                          {cap}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <p className="text-[10px] text-muted-foreground/60">
                  The vision model learns from images you upload in chat. No external training data needed.
                </p>
              </div>
            )}

            {activeTab === 'learner' && (
              <div className="space-y-2">
                {learnerInfo ? (
                  <>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="p-2 rounded bg-muted/30 border border-border/40">
                        <div className="text-[10px] text-muted-foreground">Tokens</div>
                        <div className="text-sm font-medium">{learnerInfo.total_tokens_ingested}</div>
                      </div>
                      <div className="p-2 rounded bg-muted/30 border border-border/40">
                        <div className="text-[10px] text-muted-foreground">Steps</div>
                        <div className="text-sm font-medium">{learnerInfo.train_steps_completed}</div>
                      </div>
                    </div>
                    {learnerInfo.current_loss != null && (
                      <div className="p-2 rounded bg-muted/30 border border-border/40">
                        <div className="text-[10px] text-muted-foreground">Current loss</div>
                        <div className="text-sm font-medium font-mono">{learnerInfo.current_loss.toFixed(4)}</div>
                      </div>
                    )}
                    {learnerInfo.loss_history && learnerInfo.loss_history.length >= 2 && (
                      <LossCurve data={learnerInfo.loss_history} />
                    )}
                    <Button
                      size="sm"
                      className="w-full text-xs"
                      disabled={learnerTraining}
                      onClick={onTrainStep}
                    >
                      {learnerTraining ? (
                        <><IconRefresh className="h-3 w-3 animate-spin mr-1" /> Training...</>
                      ) : (
                        <><IconBrain className="h-3 w-3 mr-1" /> Train step</>
                      )}
                    </Button>
                  </>
                ) : (
                  <div className="space-y-2 animate-pulse">
                    <div className="grid grid-cols-2 gap-2">
                      <div className="h-10 rounded bg-muted/30 border border-border/40" />
                      <div className="h-10 rounded bg-muted/30 border border-border/40" />
                    </div>
                    <div className="h-10 rounded bg-muted/30 border border-border/40" />
                    <div className="h-7 rounded bg-muted/30 border border-border/40" />
                  </div>
                )}
              </div>
            )}

            {activeTab === 'agents' && (
              <div className="space-y-1">
                {agents.length === 0 ? (
                  <p className="text-xs text-muted-foreground">No agents configured. Create one in the Agents page.</p>
                ) : (
                  <>
                    <button
                      onClick={() => onSelectAgent(null)}
                      className={cn(
                        'w-full text-left px-2 py-1.5 rounded text-xs transition-colors',
                        currentAgent === null ? 'bg-primary/10 text-primary font-medium' : 'hover:bg-muted',
                      )}
                    >
                      <span className="text-[10px] text-muted-foreground block">Default (no agent)</span>
                      <span>Direct chat</span>
                    </button>
                    {agents.map(agent => (
                      <button
                        key={agent.id}
                        onClick={() => onSelectAgent(agent)}
                        className={cn(
                          'w-full text-left px-2 py-1.5 rounded text-xs transition-colors',
                          currentAgent?.id === agent.id ? 'bg-primary/10 text-primary font-medium' : 'hover:bg-muted',
                        )}
                      >
                        <span className="font-medium">{agent.name}</span>
                        {agent.description && (
                          <span className="text-[10px] text-muted-foreground block truncate">{agent.description}</span>
                        )}
                      </button>
                    ))}
                  </>
                )}
              </div>
            )}

            {activeTab === 'checkpoints' && (
              <div className="space-y-2">
                {/* Personality section */}
                {souls.length > 0 && (
                  <div className="rounded-lg border border-border/40 bg-muted/10">
                    <div className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider px-2.5 py-1.5 border-b border-border/30">Personality</div>
                    <div className="max-h-28 overflow-y-auto p-1.5 space-y-0.5">
                      {souls.map(s => (
                        <button
                          key={s.name}
                          onClick={() => onSwitchSoul?.(s.name)}
                          className={cn(
                            'w-full text-left px-2 py-1 rounded text-xs transition-colors flex items-center justify-between',
                            currentSoulName === s.name ? 'bg-primary/10 text-primary font-medium' : 'hover:bg-muted/80',
                          )}
                        >
                          <span>{s.name}</span>
                          {currentSoulName === s.name && <IconCheck className="h-3 w-3 shrink-0" />}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Model section */}
                <div className="rounded-lg border border-border/40 bg-muted/10">
                  <div className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider px-2.5 py-1.5 border-b border-border/30">Backend Model</div>
                  <div className="max-h-28 overflow-y-auto p-1.5 space-y-0.5">
                    {availableModels.length === 0 ? (
                      <div className="px-2 py-3 text-[10px] text-muted-foreground text-center">No models available</div>
                    ) : availableModels.map(m => {
                      const info = modelInfoMap?.[m]
                      const sizeLabel = info?.size_gb ? `${info.size_gb.toFixed(2)} GB` : ''
                      return (
                        <button
                          key={m}
                          onClick={() => onSelectModel(m)}
                          className={cn(
                            'w-full text-left px-2 py-1 rounded text-xs transition-colors flex items-center justify-between font-mono',
                            currentModel === m ? 'bg-primary/10 text-primary font-medium' : 'hover:bg-muted/80',
                          )}
                          title={`${m}${sizeLabel ? ` — ${sizeLabel}` : ''}`}
                        >
                          <span className="truncate">{m.includes('/') ? m.split('/').pop() : m}</span>
                          <span className="text-[10px] text-muted-foreground/60 ml-1 shrink-0">{sizeLabel}</span>
                          {currentModel === m && <IconCheck className="h-3 w-3 shrink-0 ml-1" />}
                        </button>
                      )
                    })}
                  </div>
                </div>

                {/* Checkpoints section */}
                <div className="rounded-lg border border-border/40 bg-muted/10">
                  <div className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider px-2.5 py-1.5 border-b border-border/30 flex items-center justify-between">
                    <span>Checkpoints</span>
                    {checkpoints.length > 0 && <span className="text-[9px] font-normal normal-case">{checkpoints.length} saved</span>}
                  </div>
                  <div className="max-h-36 overflow-y-auto p-1.5 space-y-0.5">
                    {checkpoints.length === 0 ? (
                      <div className="px-2 py-3 text-[10px] text-muted-foreground text-center">No checkpoints yet</div>
                    ) : checkpoints.map(ckpt => (
                      <button
                        key={ckpt.name}
                        onClick={() => onLoadCheckpoint?.(ckpt.name)}
                        className={cn(
                          'w-full text-left px-2 py-1.5 rounded text-xs transition-colors',
                          ckpt.is_loaded ? 'bg-primary/10 text-primary font-medium' : 'hover:bg-muted/80',
                        )}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-1 min-w-0">
                            <span className="truncate">{ckpt.name}</span>
                            {ckpt.is_loaded && <IconCheck className="h-3 w-3 shrink-0" />}
                          </div>
                          {ckpt.eval_verdict && (
                            <span className={cn(
                              'text-[9px] px-1 py-0 rounded shrink-0 ml-1',
                              ckpt.eval_verdict === 'PASS' ? 'bg-success/10 text-success' : 'bg-warning/10 text-warning',
                            )}>
                              {ckpt.eval_verdict}
                            </span>
                          )}
                        </div>
                        {(ckpt.loss != null || (ckpt.traits && ckpt.traits.length > 0)) && (
                          <div className="flex items-center gap-2 text-[10px] text-muted-foreground mt-0.5">
                            {ckpt.loss != null && <span>loss {ckpt.loss.toFixed(4)}</span>}
                            {ckpt.traits && ckpt.traits.length > 0 && (
                              <span className="truncate">{ckpt.traits.join(' · ')}</span>
                            )}
                          </div>
                        )}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Footer actions */}
          <div className="border-t border-border/50 p-2 flex gap-1 shrink-0">
            <Button variant="ghost" size="sm" className="text-[10px] h-7 flex-1" onClick={onOpenConversationViewer}>
              <IconEye className="h-3 w-3 mr-1" />
              Log
            </Button>
            <Button variant="ghost" size="sm" className="text-[10px] h-7 flex-1" onClick={onOpenSettings}>
              <IconSettings className="h-3 w-3 mr-1" />
              Settings
            </Button>
            <Button variant="ghost" size="sm" className="text-[10px] h-7 flex-1" onClick={onOpenShortcuts}>
              <svg className="h-3 w-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
              Keys
            </Button>
          </div>
        </>
      )}
    </div>
  )
}

function LossCurve({ data }: { data: Array<{ step: number; loss: number }> }) {
  const W = 220, H = 72, P = 8
  const losses = data.map(d => d.loss)
  const min = Math.min(...losses)
  const max = Math.max(...losses)
  const range = max - min || 1
  const steps = data.map(d => d.step)
  const stepMin = Math.min(...steps)
  const stepRange = Math.max(...steps) - stepMin || 1

  const toX = (s: number) => P + ((s - stepMin) / stepRange) * (W - 2 * P)
  const toY = (l: number) => P + ((max - l) / range) * (H - 2 * P)

  const points = data.map(d => `${toX(d.step)},${toY(d.loss)}`).join(' ')
  const fillPath = data.length >= 2
    ? `M${data.map(d => `${toX(d.step)},${toY(d.loss)}`).join(' L')} L${toX(data[data.length - 1].step)},${H - P} L${toX(data[0].step)},${H - P} Z`
    : ''

  return (
    <div className="p-2 rounded bg-muted/30 border border-border/40">
      <div className="text-[10px] text-muted-foreground mb-1">Loss curve</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" aria-label="Training loss over steps">
        <defs>
          <linearGradient id="loss-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="hsl(var(--primary))" stopOpacity="0.2" />
            <stop offset="100%" stopColor="hsl(var(--primary))" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {data.length >= 2 && (
          <>
            <path d={fillPath} fill="url(#loss-fill)" />
            <polyline
              points={points}
              fill="none"
              stroke="hsl(var(--primary))"
              strokeWidth="1.5"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {/* Last point dot */}
            <circle
              cx={toX(data[data.length - 1].step)}
              cy={toY(data[data.length - 1].loss)}
              r="2"
              fill="hsl(var(--primary))"
            />
          </>
        )}
      </svg>
      <div className="flex justify-between text-[9px] text-muted-foreground mt-0.5">
        <span>step {stepMin}</span>
        <span>{max.toFixed(2)} &rarr; {min.toFixed(2)}</span>
        <span>step {Math.max(...steps)}</span>
      </div>
    </div>
  )
}
