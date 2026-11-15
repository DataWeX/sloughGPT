'use client'

/** Delete confirmation for a checkpoint or weight snapshot. */
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
import type { SoulsWorkbench } from './useSoulsWorkbench'

export function DeleteConfirmDialog({ wb }: { wb: SoulsWorkbench }) {
  const {
    deleteTarget,
    deleteType,
    handleDeleteCheckpoint,
    handleDeleteSnapshot,
    setDeleteTarget,
  } = wb

  return (
    <AlertDialog
      open={deleteTarget !== null}
      onOpenChange={(open) => {
        if (!open) setDeleteTarget(null)
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete {deleteType}?</AlertDialogTitle>
          <AlertDialogDescription>
            This will permanently delete <strong>{deleteTarget}</strong>. This action cannot be
            undone.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            onClick={() => {
              if (!deleteTarget) return
              if (deleteType === 'checkpoint') handleDeleteCheckpoint(deleteTarget)
              else handleDeleteSnapshot(deleteTarget)
            }}
          >
            Delete
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
