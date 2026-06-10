'use client'

import { useState } from 'react'
import type { ReactNode } from 'react'
import Link from 'next/link'

import { Button } from '@/components/ui/button'
import { ModelStatusPill, type ModelStatus } from '@sloughgpt/strui'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'
import { catalogIdMatchesRuntime } from '@/lib/inference-display'
import { IconAlert, IconRefresh } from '@/components/ui'
import { cn } from '@/lib/cn'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'

type Props = {
  health: ApiHealthSnapshot
  selectedCatalogId: string
}

function InlineCode({ children }: { children: ReactNode }) {
  return <code className="sl-chat-inline-code">{children}</code>
}

function runtimeModelLabel(health: ApiHealthSnapshot): string {
  if (health === null || health === 'offline') return 'unknown'
  return String(health.model_type ?? '').trim() || 'unknown'
}

/** Blocked / unreachable — reads clearly at 16px in header toolbars. */
function OfflineIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <circle cx="12" cy="12" r="9" strokeWidth={2} />
      <path strokeLinecap="round" strokeWidth={2} d="M7 7l10 10" />
    </svg>
  )
}

function DotsPulseIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="currentColor" viewBox="0 0 24 24" aria-hidden>
      <circle className="animate-pulse opacity-60" cx="6" cy="12" r="2" />
      <circle className="animate-pulse [animation-delay:150ms]" cx="12" cy="12" r="2" />
      <circle className="animate-pulse [animation-delay:300ms]" cx="18" cy="12" r="2" />
    </svg>
  )
}

function CpuIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
      />
    </svg>
  )
}

/**
 * Offline / no-weights / catalog-vs-runtime notices below the chat toolbar.
 * Styled as compact shell notes (see `.sl-chat-toolbar-note` in globals.css).
 * Runtime controls live in the toolbar (`InferenceRuntimeToolbar`).
 */
export function InferenceStatusBar({ health, selectedCatalogId }: Props) {
  const mismatch =
    health !== null &&
    health !== 'offline' &&
    health.model_loaded &&
    !catalogIdMatchesRuntime(selectedCatalogId, health.model_type)

  const showOffline = health === 'offline'
  const showNoWeights =
    health !== null && health !== 'offline' && !health.model_loaded
  const showMismatch = mismatch

  if (!showOffline && !showNoWeights && !showMismatch) {
    return null
  }

  return (
    <div className="flex flex-col gap-1.5" role="status">
      {showOffline ? (
        <div className="sl-chat-toolbar-note sl-chat-toolbar-note--err">
          <p className="sl-chat-toolbar-note__label">API unreachable</p>
          <p className="text-xs leading-snug text-muted-foreground">
            Start the server from the repo root (
            <InlineCode>python3 apps/api/server/main.py</InlineCode>) and ensure{' '}
            <InlineCode>NEXT_PUBLIC_API_URL</InlineCode> matches.
          </p>
        </div>
      ) : null}

      {showNoWeights ? (
        <div className="sl-chat-toolbar-note sl-chat-toolbar-note--warn">
          <p className="sl-chat-toolbar-note__label">No weights loaded</p>
          <p className="text-xs leading-snug text-muted-foreground">
            Load weights in the API before chatting.{' '}
            <Link href="/models" className="text-primary underline-offset-2 hover:underline">
              Open models
            </Link>{' '}
            or wait for autoload (<InlineCode>MAN_AUTOLOAD_MODEL</InlineCode>).
          </p>
        </div>
      ) : null}

      {showMismatch ? (
        <div className="sl-chat-toolbar-note sl-chat-toolbar-note--hint flex flex-row items-start gap-2.5">
          <IconAlert className="mt-0.5 h-4 w-4 shrink-0 text-primary" aria-hidden="true" />
          <p className="min-w-0 text-xs leading-snug text-muted-foreground">
            <span className="font-medium text-foreground/85">Catalog ≠ runtime.</span>{' '}
            Chat selection <InlineCode>{selectedCatalogId}</InlineCode>
            <span className="text-foreground/35"> · </span>
            API <InlineCode>{runtimeModelLabel(health)}</InlineCode>.
          </p>
        </div>
      ) : null}
    </div>
  )
}

type ToolbarProps = {
  health: ApiHealthSnapshot
  onRefresh: () => void
}

/** Simple status indicator for chat header. */
export function InferenceRuntimeToolbar({ health, onRefresh }: ToolbarProps) {
  if (health === null) return <span className="text-xs text-muted-foreground/50">...</span>
  if (health === 'offline') return <span className="text-xs text-destructive">Offline</span>
  if (health.model_loaded) return <span className="text-xs text-success/70">{health.model_type}</span>
  return <span className="text-xs text-warning/70">No model</span>
}

/** Compact model status bar for navbar/header — clickable to show details */
export function ModelStatusBar({ health }: { health: ApiHealthSnapshot }) {
  const [showDetails, setShowDetails] = useState(false)

  const getStatus = (): ModelStatus => {
    if (health === null) return 'loading'
    if (health === 'offline') return 'offline'
    if (health.model_loaded) return 'loaded'
    return 'no-model'
  }

  const getAriaLabel = () => {
    if (health === null) return 'Checking API status'
    if (health === 'offline') return 'API offline'
    if (health.model_loaded) return `Model loaded: ${health.model_type}`
    return 'No model loaded'
  }

  const status = getStatus()

  const hasDetails = health !== null && health !== 'offline' && health.model_loaded

  return (
    <>
      <div
        role="status"
        aria-label={getAriaLabel()}
        className={cn(hasDetails && 'cursor-pointer')}
        onClick={hasDetails ? () => setShowDetails(true) : undefined}
        onKeyDown={hasDetails ? (e) => { if (e.key === 'Enter' || e.key === ' ') setShowDetails(true) } : undefined}
        tabIndex={hasDetails ? 0 : undefined}
      >
        <ModelStatusPill
          status={status}
          modelName={health !== null && health !== 'offline' ? health.model_type : undefined}
        />
      </div>
      <ModelDetailsDialog open={showDetails} onClose={() => setShowDetails(false)} health={health} />
    </>
  )
}

function ModelDetailsDialog({ open, onClose, health }: { open: boolean; onClose: () => void; health: ApiHealthSnapshot }) {
  const h = health !== null && health !== 'offline' ? health : null

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-sm">
        <DialogHeader className="flex-row items-center justify-between">
          <DialogTitle className="text-base">Model Details</DialogTitle>
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onClose} aria-label="Close">
            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </Button>
        </DialogHeader>
        <div className="space-y-3 pt-2 text-sm">
          <div className="grid grid-cols-2 gap-y-2 gap-x-4">
            <span className="text-muted-foreground">Status</span>
            <span className={h?.model_loaded ? 'text-success font-medium' : 'text-muted-foreground'}>
              {h?.model_loaded ? 'Loaded' : h ? 'No weights' : 'Connecting...'}
            </span>
            <span className="text-muted-foreground">Model</span>
            <span className="font-mono text-xs truncate max-w-[160px]">{h?.model_type ?? '—'}</span>
            {h?.num_parameters != null && (
              <>
                <span className="text-muted-foreground">Parameters</span>
                <span className="font-mono text-xs">
                  {h.num_parameters < 1e6
                    ? `${(h.num_parameters / 1e3).toFixed(0)}K`
                    : `${(h.num_parameters / 1e9).toFixed(1)}B`}
                </span>
              </>
            )}
            {h?.vocab_size != null && (
              <>
                <span className="text-muted-foreground">Vocab size</span>
                <span className="font-mono text-xs">{h.vocab_size.toLocaleString()}</span>
              </>
            )}
            {h?.block_size != null && (
              <>
                <span className="text-muted-foreground">Block size</span>
                <span className="font-mono text-xs">{h.block_size}</span>
              </>
            )}
            {h?.inference_count != null && (
              <>
                <span className="text-muted-foreground">Inference calls</span>
                <span className="font-mono text-xs">{h.inference_count.toLocaleString()}</span>
              </>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
