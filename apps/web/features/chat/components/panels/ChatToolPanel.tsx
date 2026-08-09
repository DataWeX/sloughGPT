'use client'

import { useState } from 'react'

import { cn, Button } from '@sloughgpt/strui'
import { IconX, IconEye } from '@sloughgpt/strui'
import { useChatContext } from '@/features/chat/contexts/ChatContext'
import { KnowledgeTab } from './KnowledgeTab'
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
                    <svg className="h-3 w-3 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5-1.253"/></svg>
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
              <svg className="h-3 w-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
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
