'use client'

import { useState, useCallback, memo } from 'react'
import { Button } from '@sloughgpt/strui'
import { ConfirmDialog } from '@/components/ConfirmDialog'

interface StopTrainingButtonProps {
  onStop: () => Promise<void>
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
  variant?: 'destructive' | 'outline'
  size?: 'sm' | 'default'
  className?: string
}

export const StopTrainingButton = memo(function StopTrainingButton({
  onStop,
  addToast,
  variant = 'destructive',
  size = 'sm',
  className,
}: StopTrainingButtonProps) {
  const [showConfirm, setShowConfirm] = useState(false)
  const [stopping, setStopping] = useState(false)

  const handleStop = useCallback(async () => {
    setStopping(true)
    try {
      await onStop()
      addToast('Training stopped', 'success')
    } catch {
      addToast('Could not stop training', 'error')
    } finally {
      setStopping(false)
      setShowConfirm(false)
    }
  }, [onStop, addToast])

  return (
    <>
      <Button
        variant={variant}
        size={size}
        className={className}
        onClick={() => setShowConfirm(true)}
        disabled={stopping}
      >
        {stopping ? 'Stopping...' : 'Stop training'}
      </Button>
      <ConfirmDialog
        open={showConfirm}
        onOpenChange={setShowConfirm}
        title="Stop training?"
        description="This will cancel the current training run. You can resume from the last checkpoint later."
        confirmLabel={stopping ? 'Stopping...' : 'Stop training'}
        onConfirm={handleStop}
        destructive
      />
    </>
  )
})
