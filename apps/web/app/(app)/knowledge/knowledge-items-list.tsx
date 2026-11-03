'use client'

/** The knowledge list: skeleton, empty state, search highlight, inline edit and selection. */
import {
  Button,
  Checkbox,
  EmptyCard,
  IconCheck,
  IconEdit,
  IconSearch,
  IconTrash,
  IconX,
  Input,
  Skeleton,
  Slider,
  cn,
} from '@sloughgpt/strui'
import { MS_PER_SECOND } from '@/lib/format-bytes'
import type { KnowledgeWorkbench } from './useKnowledgeWorkbench'

function highlightText(text: string, query: string): React.ReactNode {
  if (!query) return text
  const lowered = text.toLowerCase()
  const q = query.toLowerCase()
  const idx = lowered.indexOf(q)
  if (idx === -1) return text
  const before = text.slice(0, idx)
  const match = text.slice(idx, idx + q.length)
  const after = text.slice(idx + q.length)
  return (
    <>
      {before}
      <mark className="bg-warning/30 text-foreground rounded px-0.5">{match}</mark>
      {after}
    </>
  )
}

export function KnowledgeItemsList({ wb }: { wb: KnowledgeWorkbench }) {
  const {
    displayItems,
    editContent,
    editErrors,
    editTopic,
    editingId,
    editingImportanceId,
    handleSaveEdit,
    handleSaveImportance,
    importanceValue,
    loading,
    router,
    search,
    selectedIds,
    setEditContent,
    setEditErrors,
    setEditTopic,
    setEditingId,
    setEditingImportanceId,
    setImportanceValue,
    setPendingDelete,
    toggleSelect,
    toggleSelectAll,
  } = wb

  return (
    <>
      {loading ? (
        <div className="space-y-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-16 rounded-lg" />
          ))}
        </div>
      ) : displayItems.length === 0 ? (
        <EmptyCard
          message={search ? 'No results found' : 'Nothing here yet'}
          description={
            search
              ? 'Try a different search term'
              : 'Add things you want your AI to remember about you. Click the + button to get started.'
          }
          icon={<IconSearch className="h-5 w-5" />}
          action={null}
        />
      ) : (
        <>
          {displayItems.length > 1 && (
            <label className="flex items-center gap-2 text-sm text-muted-foreground cursor-pointer">
              <Checkbox
                checked={selectedIds.size === displayItems.length && displayItems.length > 0}
                onCheckedChange={toggleSelectAll}
                aria-label="Select all knowledge items"
                className="rounded border-border"
              />
              Select all ({displayItems.length})
            </label>
          )}
          <div className="space-y-2 max-h-[60vh] overflow-y-auto overscroll-contain">
            {displayItems.map((item) => (
              <div
                key={item.id}
                className={cn(
                  'group relative p-4 rounded-lg border text-sm leading-relaxed transition-colors cursor-pointer',
                  selectedIds.has(item.id)
                    ? 'bg-primary/[0.06] border-primary/30'
                    : editingId === item.id
                      ? 'bg-primary/[0.04] border-primary/30'
                      : 'bg-card border-border/40 hover:bg-muted/30',
                )}
                onClick={() => router.push(`/knowledge/${item.id}`)}
              >
                <div className="flex items-start gap-2">
                  <Checkbox
                    checked={selectedIds.has(item.id)}
                    onCheckedChange={() => toggleSelect(item.id)}
                    aria-label={`Select knowledge item`}
                    className="mt-0.5 rounded border-border shrink-0"
                    onClick={(e) => e.stopPropagation()}
                  />
                  <div className="flex-1 min-w-0">
                    {editingId === item.id ? (
                      <div className="space-y-2">
                        <div>
                          <textarea
                            className={cn(
                              'w-full p-2 text-sm border rounded-lg resize-none h-16 bg-background focus:outline-none focus:ring-1 focus:ring-primary/40',
                              editErrors.content
                                ? 'border-destructive ring-destructive/20'
                                : 'border-input',
                            )}
                            value={editContent}
                            onChange={(e) => {
                              setEditContent(e.target.value)
                              if (editErrors.content)
                                setEditErrors((prev) => ({ ...prev, content: undefined }))
                            }}
                            autoFocus
                            onKeyDown={(e) => {
                              if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault()
                                handleSaveEdit(item.id)
                              }
                              if (e.key === 'Escape') setEditingId(null)
                            }}
                            aria-invalid={!!editErrors.content}
                            aria-describedby={editErrors.content ? 'edit-content-error' : undefined}
                            aria-label="Edit knowledge content"
                          />
                          {editErrors.content && (
                            <p
                              id="edit-content-error"
                              className="text-xs text-destructive mt-1"
                              role="alert"
                            >
                              {editErrors.content}
                            </p>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <Input
                            value={editTopic}
                            onChange={(e) => setEditTopic(e.target.value)}
                            className="h-7 text-xs flex-1"
                            placeholder="Topic"
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') handleSaveEdit(item.id)
                              if (e.key === 'Escape') setEditingId(null)
                            }}
                          />
                          <button
                            type="button"
                            onClick={() => handleSaveEdit(item.id)}
                            className="text-success hover:text-success/80 p-1"
                            aria-label="Save edit"
                          >
                            <IconCheck className="h-3.5 w-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => setEditingId(null)}
                            className="text-muted-foreground hover:text-foreground p-1"
                            aria-label="Cancel edit"
                          >
                            <IconX className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <p className="text-sm text-foreground break-words">
                          {search ? highlightText(item.content, search) : item.content}
                        </p>
                        <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
                          {item.topic && (
                            <span className="text-xs px-1.5 py-0.5 rounded font-medium bg-primary/10 text-primary">
                              {item.topic}
                            </span>
                          )}
                          {item.source && (
                            <span className="text-xs px-1.5 py-0.5 rounded font-medium bg-muted text-muted-foreground">
                              {item.source}
                            </span>
                          )}
                          {editingImportanceId === item.id ? (
                            <div className="flex items-center gap-1.5 bg-muted/50 rounded px-1.5 py-0.5">
                              <Slider
                                value={[importanceValue]}
                                min={0}
                                max={10}
                                step={0.1}
                                onValueChange={([v]) => setImportanceValue(v)}
                                size="sm"
                                className="w-16"
                              />
                              <Button
                                size="icon-sm"
                                variant="ghost"
                                onClick={() => handleSaveImportance(item.id)}
                                aria-label="Save importance"
                              >
                                <IconCheck className="h-3 w-3" />
                              </Button>
                              <Button
                                size="icon-sm"
                                variant="ghost"
                                onClick={() => setEditingImportanceId(null)}
                                aria-label="Cancel editing"
                              >
                                <IconX className="h-3 w-3" />
                              </Button>
                            </div>
                          ) : (
                            <button
                              type="button"
                              onClick={() => {
                                setEditingImportanceId(item.id)
                                setImportanceValue(item.importance)
                              }}
                              className="text-xs text-muted-foreground/50 hover:text-muted-foreground transition-colors cursor-pointer"
                            >
                              importance: {item.importance.toFixed(1)}
                            </button>
                          )}
                          <span className="text-xs text-muted-foreground/50">
                            {new Date(item.timestamp * MS_PER_SECOND).toLocaleDateString(
                              undefined,
                              { month: 'short', day: 'numeric' },
                            )}
                          </span>
                        </div>
                      </>
                    )}
                  </div>
                  {editingId !== item.id && (
                    <div className="flex items-center gap-1 shrink-0">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation()
                          setEditingId(item.id)
                          setEditContent(item.content)
                          setEditTopic(item.topic || '')
                        }}
                        className="opacity-0 group-hover:opacity-100 focus-within:opacity-100 text-muted-foreground hover:text-primary p-1 transition-opacity"
                        aria-label="Edit knowledge"
                      >
                        <IconEdit className="h-4 w-4" />
                      </button>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation()
                          setPendingDelete(item)
                        }}
                        className="opacity-0 group-hover:opacity-100 focus-within:opacity-100 text-muted-foreground hover:text-destructive p-1 transition-opacity"
                        aria-label="Delete knowledge"
                      >
                        <IconTrash className="h-4 w-4" />
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </>
  )
}
