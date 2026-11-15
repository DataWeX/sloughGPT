'use client'

/** Move-to-topic dialog for the current selection. */
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
} from '@sloughgpt/strui'
import type { KnowledgeWorkbench } from './useKnowledgeWorkbench'

export function KnowledgeBulkTopicDialog({ wb }: { wb: KnowledgeWorkbench }) {
  const {
    bulkTopic,
    handleBulkTopicReassign,
    selectedIds,
    setBulkTopic,
    setShowBulkTopic,
    showBulkTopic,
    topics,
  } = wb

  return (
    <AlertDialog open={showBulkTopic} onOpenChange={() => setShowBulkTopic(false)}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Move {selectedIds.size} items to topic</AlertDialogTitle>
          <AlertDialogDescription>
            Assign all selected knowledge items to a new topic.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <div className="py-2">
          <Input
            value={bulkTopic}
            onChange={(e) => setBulkTopic(e.target.value)}
            placeholder="Enter topic name..."
            autoFocus
            onKeyDown={(e) => {
              if (e.key === 'Enter' && bulkTopic.trim()) handleBulkTopicReassign()
            }}
          />
          <div className="flex flex-wrap gap-1 mt-2">
            {topics.slice(0, 6).map((t) => (
              <button
                key={t.name}
                type="button"
                onClick={() => setBulkTopic(t.name)}
                className="text-xs px-2 py-0.5 rounded-full border border-border/40 text-muted-foreground hover:bg-muted/50 transition-colors"
              >
                {t.name}
              </button>
            ))}
          </div>
        </div>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <Button size="sm" disabled={!bulkTopic.trim()} onClick={handleBulkTopicReassign}>
            Move
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
