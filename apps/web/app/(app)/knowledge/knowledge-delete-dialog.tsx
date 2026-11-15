'use client'

/** Delete confirmation for a single knowledge item. */
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

export function KnowledgeDeleteDialog({ wb }: { wb: KnowledgeWorkbench }) {
  const { handleDelete, pendingDelete, setPendingDelete } = wb

  return (
    <AlertDialog open={pendingDelete !== null} onOpenChange={() => setPendingDelete(null)}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete knowledge?</AlertDialogTitle>
          <AlertDialogDescription>
            This will permanently remove this fact from the knowledge base.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={handleDelete}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
          >
            Delete
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
