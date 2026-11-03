'use client'

/** Add-knowledge dialog with topic suggestion and validation errors. */
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  Button,
  Input,
  cn,
} from '@sloughgpt/strui'
import { suggestTopic } from './knowledge-format'
import type { KnowledgeWorkbench } from './useKnowledgeWorkbench'

export function KnowledgeAddDialog({ wb }: { wb: KnowledgeWorkbench }) {
  const {
    addErrors,
    handleAdd,
    newContent,
    newTopic,
    setAddErrors,
    setNewContent,
    setNewTopic,
    setShowAdd,
    showAdd,
  } = wb

  return (
    <AlertDialog open={showAdd} onOpenChange={() => setShowAdd(false)}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Add knowledge</AlertDialogTitle>
          <AlertDialogDescription>
            Add a fact the AI should remember across conversations.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <div className="space-y-3 py-2">
          <div>
            <textarea
              className={cn(
                'w-full p-2.5 text-sm border rounded-lg resize-none h-24 bg-background focus:outline-none focus:ring-1 focus:ring-primary/40',
                addErrors.content ? 'border-destructive ring-destructive/20' : 'border-input',
              )}
              placeholder="Enter a fact, preference, or piece of context..."
              value={newContent}
              onChange={(e) => {
                const val = e.target.value
                setNewContent(val)
                if (addErrors.content) setAddErrors((prev) => ({ ...prev, content: undefined }))
                if (newTopic === 'general' || newTopic === '') {
                  setNewTopic(suggestTopic(val))
                }
              }}
              autoFocus
              aria-invalid={!!addErrors.content}
              aria-describedby={addErrors.content ? 'add-content-error' : undefined}
              aria-label="New knowledge content"
            />
            {addErrors.content && (
              <p id="add-content-error" className="text-xs text-destructive mt-1" role="alert">
                {addErrors.content}
              </p>
            )}
          </div>
          <div className="flex items-center gap-2">
            <label htmlFor="add-knowledge-topic" className="text-xs text-muted-foreground shrink-0">
              Topic:
            </label>
            <Input
              id="add-knowledge-topic"
              value={newTopic}
              onChange={(e) => setNewTopic(e.target.value)}
              className="h-8 text-xs flex-1"
              placeholder="general"
            />
          </div>
        </div>
        <AlertDialogFooter>
          <AlertDialogCancel
            onClick={() => {
              setNewContent('')
              setNewTopic('general')
            }}
          >
            Cancel
          </AlertDialogCancel>
          <Button size="sm" disabled={!newContent.trim()} onClick={handleAdd}>
            Add
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
