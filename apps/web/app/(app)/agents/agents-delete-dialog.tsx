'use client'

/**
 * Confirmation dialog for a single agent deletion.
 */
import type { AgentsWorkbench } from './useAgentsWorkbench'
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

export function DeleteAgentDialog({
  pendingDelete,
  setPendingDelete,
  handleDelete,
}: {
  pendingDelete: AgentsWorkbench['pendingDelete']
  setPendingDelete: AgentsWorkbench['setPendingDelete']
  handleDelete: AgentsWorkbench['handleDelete']
}) {
  return (
    <AlertDialog open={pendingDelete !== null} onOpenChange={() => setPendingDelete(null)}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete agent?</AlertDialogTitle>
          <AlertDialogDescription>
            Are you sure you want to delete &ldquo;{pendingDelete?.name}&rdquo;? This cannot be
            undone.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={() => {
              if (pendingDelete) handleDelete(pendingDelete.id)
            }}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
          >
            Delete
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
