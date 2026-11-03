'use client'

/** Terminal tab — the TUI shell panel. */
import { TerminalPanel } from '@/components/shell/TerminalPanel'

export function ShellTab() {
  // The Linux VM journey lives on /vm (VM Console) — no duplicate sub-tab here
  // (Page Rule: a page is a journey projection, not a view of another page).
  return (
    <div className="space-y-3">
      <TerminalPanel className="h-[calc(100vh-14rem)]" />
    </div>
  )
}
