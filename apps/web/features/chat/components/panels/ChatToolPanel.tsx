'use client'

import { useState } from 'react'

import { cn, Button } from '@sloughgpt/strui'
import { IconX, IconEye, IconSettings, IconDocument } from '@sloughgpt/strui'
import { useChatContext } from '@/features/chat/contexts/ChatContext'
import { KnowledgeTab } from './KnowledgeTab'
import { MemoryTab } from './MemoryTab'
import { ContextTab } from './ContextTab'
import { VisionTabContent } from './VisionTabContent'
import { QuickPrompts } from './../input/QuickPrompts'
import { ChatBookmarksPanel } from './ChatBookmarksPanel'
import { ChatSessionStatsCard } from './ChatSessionStatsCard'

interface ChatToolPanelProps {
  open: boolean
  onClose: () => void
  sessionId: string | null
  bookmarks?: import('@/features/chat/hooks/useChatBookmarks').BookmarkedMessage[]
  onRemoveBookmark?: (id: string) => void
  onClearBookmarks?: () => void
}

export function ChatToolPanel({ open, onClose, sessionId, bookmarks = [], onRemoveBookmark, onClearBookmarks }: ChatToolPanelProps) {
  const [showVision, setShowVision] = useState(false)
  const ctx = useChatContext()

  return (
    <div
      id="chat-tool-panel"
      className={cn(
        'border-l border-border/50 bg-background overflow-hidden transition-all duration-200 flex flex-col',
        open ? 'w-[var(--tool-panel-width)] min-w-[var(--tool-panel-width)] lg:relative lg:w-[var(--tool-panel-width)] fixed right-0 top-0 bottom-0 z-50 shadow-xl lg:shadow-none' : 'w-0 min-w-0',
      )}
    >
      {open && (
        <>
          {/* ── Header ── */}
          <div className="flex items-center justify-between px-3 py-2 border-b border-border/50 shrink-0">
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              {showVision ? 'Vision' : 'Tools'}
            </span>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => setShowVision(!showVision)}
                className={cn(
                  'h-6 w-6 flex items-center justify-center rounded hover:bg-muted/60 transition-colors',
                  showVision && 'text-primary',
                )}
                title={showVision ? 'Show tools' : 'Show vision'}
                aria-label={showVision ? 'Show tools panel' : 'Show vision panel'}
              >
                <IconEye className="h-3.5 w-3.5" />
              </button>
              <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onClose} aria-label="Close tools panel">
                <IconX className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>

          {/* ── Content ── */}
          <div className="flex-1 overflow-y-auto p-3 space-y-4 text-xs scrollbar-thin">
            {showVision ? (
              <VisionTabContent
                visionImagesLearned={ctx.visionCaps?.images_learned}
                visionTrained={ctx.visionCaps?.trained}
                visionStatus={ctx.visionCaps?.status}
                visionCaptionHistory={ctx.visionCaptionHistory}
                visionVocabSize={ctx.visionVocabSize}
                sessionId={sessionId}
                onGeneratedImage={(dataUrl, prompt) => {
                  const event = new CustomEvent('generate-image', { detail: { dataUrl, prompt } })
                  window.dispatchEvent(event)
                }}
                onSendText={(text) => {
                  window.dispatchEvent(new CustomEvent('send-text', { detail: { text } }))
                }}
              />
            ) : (
              <>
                <ChatSessionStatsCard sessionId={sessionId} />
                <section>
                  <div className="flex items-center gap-1.5 mb-2">
                    <IconDocument className="h-3 w-3 text-muted-foreground" />
                    <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Knowledge</span>
                  </div>
                  <KnowledgeTab
                    onOpenConversationViewer={ctx.onOpenConversationViewer}
                    onOpenSettings={ctx.onOpenSettings}
                    onOpenShortcuts={ctx.onOpenShortcuts}
                  />
                </section>
                <section>
                  <div className="flex items-center gap-1.5 mb-2">
                    <svg className="h-3 w-3 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z" /></svg>
                    <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Memory</span>
                  </div>
                  <MemoryTab />
                </section>
                <section>
                  <div className="flex items-center gap-1.5 mb-2">
                    <svg className="h-3 w-3 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" /></svg>
                    <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Context</span>
                  </div>
                  <ContextTab />
                </section>
                <section>
                  <div className="flex items-center gap-1.5 mb-2">
                    <svg className="h-3 w-3 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                    <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Quick Prompts</span>
                  </div>
                  <QuickPrompts onUsePrompt={(text) => ctx.setInput(text)} />
                </section>
                <section>
                  <ChatBookmarksPanel
                    bookmarks={bookmarks}
                    onRemove={onRemoveBookmark || (() => {})}
                    onClear={onClearBookmarks || (() => {})}
                    onJumpToMessage={(id) => {
                      const el = document.getElementById(`msg-${id}`)
                      el?.scrollIntoView({ behavior: 'smooth', block: 'center' })
                    }}
                  />
                </section>
              </>
            )}
          </div>

          {/* ── Footer actions ── */}
          <div className="border-t border-border/50 p-2 flex gap-1 shrink-0">
            <Button variant="ghost" size="sm" className="text-[10px] h-7 flex-1" onClick={ctx.onOpenConversationViewer}>
              <IconEye className="h-3 w-3 mr-1" />
              Log
            </Button>
            <Button variant="ghost" size="sm" className="text-[10px] h-7 flex-1" onClick={ctx.onOpenSettings}>
              <IconSettings className="h-3 w-3 mr-1" />
              Settings
            </Button>
            <Button variant="ghost" size="sm" className="text-[10px] h-7 flex-1" onClick={ctx.onOpenShortcuts}>
              <svg className="h-3 w-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
              Keys
            </Button>
          </div>
        </>
      )}
    </div>
  )
}
