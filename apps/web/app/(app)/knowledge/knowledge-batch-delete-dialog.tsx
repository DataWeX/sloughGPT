'use client'

/** Delete confirmation for the current selection. */
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@sloughgpt/strui'
import type { KnowledgeWorkbench } from './useKnowledgeWorkbench'

export function KnowledgeBatchDeleteDialog({ wb }: { wb: KnowledgeWorkbench }) {
  const { handleBatchDelete, pendingBatchDelete, selectedIds, setPendingBatchDelete } = wb

  return (
    <AlertDialog open={pendingBatchDelete} onOpenChange={() => setPendingBatchDelete(false)}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete {selectedIds.size} items?</AlertDialogTitle>
          <AlertDialogDescription>
            This will permanently remove these facts from the knowledge base.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={handleBatchDelete}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
          >
            Delete all
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
