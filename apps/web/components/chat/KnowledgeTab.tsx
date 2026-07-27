'use client'

import { useState, useRef, useEffect } from 'react'
import { cn, Button } from '@sloughgpt/strui'
import { IconX } from '@sloughgpt/strui'
import { chatDB, type KnowledgeItem } from '@/lib/db'
import { logger } from '@/lib/dev-log'

interface KnowledgeTabProps {
  onOpenConversationViewer: () => void
  onOpenSettings: () => void
  onOpenShortcuts: () => void
}

export function KnowledgeTab({
  onOpenConversationViewer,
  onOpenSettings,
  onOpenShortcuts,
}: KnowledgeTabProps) {
  const [knowledge, setKnowledge] = useState<KnowledgeItem[]>([])
  const [showAddKnowledge, setShowAddKnowledge] = useState(false)
  const [newKnowledge, setNewKnowledge] = useState('')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editText, setEditText] = useState('')
  const [backendContentIds, setBackendContentIds] = useState<Map<string, string>>(new Map())
  const syncTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    chatDB.getKnowledge().then(items => setKnowledge(items)).catch(() => {})
  }, [])

  useEffect(() => {
    return () => {
      if (syncTimerRef.current) clearTimeout(syncTimerRef.current)
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    import('@/lib/knowledge-controller').then(({ knowledgeController }) => {
      knowledgeController.list().then(async (backendItems) => {
        if (cancelled) return
        const idMap = new Map<string, string>()
        for (const item of backendItems) {
          idMap.set(item.content, item.id)
        }
        setBackendContentIds(idMap)
        const local = await chatDB.getKnowledge()
        const existingContent = new Set(local.map(k => k.content))
        const needsSave = backendItems.some(item => !existingContent.has(item.content))
        if (needsSave) {
          const merged = [...local]
          for (const item of backendItems) {
            if (!existingContent.has(item.content)) {
              merged.push({
                id: `know_b_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
                content: item.content,
                timestamp: Date.now(),
              })
            }
          }
          setKnowledge(merged)
          await chatDB.clearKnowledge()
          await chatDB.importKnowledge(merged)
        }
      }).catch(() => {})
    }).catch(() => {})
    return () => { cancelled = true }
  }, [])

  const syncToBackend = (items: KnowledgeItem[]) => {
    if (syncTimerRef.current) clearTimeout(syncTimerRef.current)
    syncTimerRef.current = setTimeout(async () => {
      try {
        const { knowledgeController } = await import('@/lib/knowledge-controller')
        await knowledgeController.batchIngest(
          items.map(k => ({ content: k.content, source: 'injected' }))
        )
        const updated = await knowledgeController.list()
        const idMap = new Map<string, string>()
        for (const item of updated) idMap.set(item.content, item.id)
        setBackendContentIds(idMap)
      } catch (err) {
        logger.warning('Knowledge sync failed, will retry', { error: String(err) })
      }
    }, 2000)
  }

  const saveKnowledge = async (items: KnowledgeItem[]) => {
    setKnowledge(items)
    await chatDB.clearKnowledge()
    await chatDB.importKnowledge(items)
    syncToBackend(items)
  }

  const addKnowledge = async () => {
    if (!newKnowledge.trim()) return
    const item: KnowledgeItem = {
      id: `know_${Date.now()}`,
      content: newKnowledge.trim(),
      timestamp: Date.now(),
    }
    await saveKnowledge([...knowledge, item])
    setNewKnowledge('')
    setShowAddKnowledge(false)
  }

  const removeKnowledge = async (id: string) => {
    const item = knowledge.find(k => k.id === id)
    await saveKnowledge(knowledge.filter(k => k.id !== id))
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
                    <Button size="sm" className="h-5 text-[10px] px-2 flex-1" onClick={async () => {
                      await saveKnowledge(knowledge.map(k => k.id === item.id ? { ...k, content: editText } : k))
                      setEditingId(null)
                    }}>Save</Button>
                    <Button variant="outline" size="sm" className="h-5 text-[10px] px-2" onClick={() => setEditingId(null)}>Cancel</Button>
                  </div>
                </div>
              ) : (
                <>
                  <span>{item.content.length > 200 ? item.content.slice(0, 200) + '...' : item.content}</span>
                  <div className="flex items-center gap-1.5 mt-1">
                    <span className={cn(
                      "text-[9px] px-1.5 py-0.5 rounded font-medium",
                      backendContentIds.has(item.content) ? "bg-success/10 text-success" : "bg-muted text-muted-foreground"
                    )}>
                      {backendContentIds.has(item.content) ? 'Synced' : 'Local'}
                    </span>
                    {item.timestamp && (
                      <span className="text-[9px] text-muted-foreground/50">
                        {new Date(item.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                      </span>
                    )}
                  </div>
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
          onClick={() => saveKnowledge([])}
          className="text-[10px] text-muted-foreground hover:text-destructive transition-colors"
        >
          Clear all
        </button>
      )}

      <a
        href="/datasets"
        className="block text-center text-[10px] text-muted-foreground hover:text-foreground pt-1 border-t border-border/30 transition-colors"
      >
        Browse datasets →
      </a>
    </div>
  )
}
