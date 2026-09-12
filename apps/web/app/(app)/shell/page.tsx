'use client'

import { useState } from 'react'
import { cn } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { TerminalPanel } from '@/components/shell/TerminalPanel'
import { V86TerminalPanel } from '@/components/shell/V86TerminalPanel'

type ShellMode = 'backend' | 'v86'

export default function ShellPage() {
  const [mode, setMode] = useState<ShellMode>('backend')

  return (
    <PageContainer title="Shell" maxWidth="max-w-5xl">
      <div className="space-y-3">
        <div className="flex items-center gap-1 rounded-xl border border-white/[0.06] bg-[#111111] p-1 w-fit">
          <button
            type="button"
            onClick={() => setMode('backend')}
            className={cn(
              'rounded-lg px-4 py-1.5 text-[11px] font-medium transition-all duration-200',
              mode === 'backend'
                ? 'bg-[#1c1c1e] text-[#c7c7cc] shadow-sm shadow-black/20'
                : 'text-[#636366] hover:text-[#8e8e93]',
            )}
          >
            Backend
          </button>
          <button
            type="button"
            onClick={() => setMode('v86')}
            className={cn(
              'rounded-lg px-4 py-1.5 text-[11px] font-medium transition-all duration-200',
              mode === 'v86'
                ? 'bg-[#1c1c1e] text-[#c7c7cc] shadow-sm shadow-black/20'
                : 'text-[#636366] hover:text-[#8e8e93]',
            )}
          >
            Browser VM
          </button>
        </div>
        {mode === 'backend' ? (
          <TerminalPanel className="h-[calc(100vh-10rem)]" />
        ) : (
          <V86TerminalPanel className="h-[calc(100vh-10rem)]" />
        )}
      </div>
    </PageContainer>
  )
}
