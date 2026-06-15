'use client'

import { useState, useEffect, useCallback, useRef } from 'react'

import { Button } from '@/components/ui/button'
import { IconX, IconBrain, IconEye, IconHeart, IconSettings, IconDocument, IconModel, IconInfo } from '@/components/ui'
import { cn } from '@/lib/cn'
import { useChatContext } from '@/contexts/ChatContext'
import { ContextInspector } from './ContextInspector'
import { KnowledgeTab } from './KnowledgeTab'
import { VisionTabContent } from './VisionTabContent'
import { LearnerTab } from './LearnerTab'
import { AgentsTab } from './AgentsTab'
import { CheckpointsTab } from './CheckpointsTab'
import { QuickPrompts } from './QuickPrompts'

type TabId = 'knowledge' | 'vision' | 'learner' | 'checkpoints' | 'agents' | 'context' | 'prompts'

interface ChatToolPanelProps {
  open: boolean
  onClose: () => void
  sessionId: string | null
}

const tabs: { id: TabId; label: string; icon: typeof IconBrain }[] = [
  { id: 'context', label: 'Context', icon: IconInfo },
  { id: 'knowledge', label: 'Knowledge', icon: IconDocument },
  { id: 'prompts', label: 'Prompts', icon: IconSettings },
  { id: 'vision', label: 'Vision', icon: IconEye },
  { id: 'learner', label: 'Learner', icon: IconBrain },
  { id: 'agents', label: 'Agents', icon: IconHeart },
  { id: 'checkpoints', label: 'Models', icon: IconModel },
]

export function ChatToolPanel({ open, onClose, sessionId }: ChatToolPanelProps) {
  const [activeTab, setActiveTab] = useState<TabId>('knowledge')
  const ctx = useChatContext()
  const tabListRef = useRef<HTMLDivElement>(null)
  const tabRefs = useRef<Map<TabId, HTMLButtonElement>>(new Map())

  const handleTabKeyDown = useCallback((e: React.KeyboardEvent, tabId: TabId) => {
    const currentIndex = tabs.findIndex(t => t.id === tabId)
    let nextIndex = currentIndex

    if (e.key === 'ArrowRight') {
      nextIndex = (currentIndex + 1) % tabs.length
    } else if (e.key === 'ArrowLeft') {
      nextIndex = (currentIndex - 1 + tabs.length) % tabs.length
    } else if (e.key === 'Home') {
      nextIndex = 0
    } else if (e.key === 'End') {
      nextIndex = tabs.length - 1
    } else {
      return
    }

    e.preventDefault()
    const nextTab = tabs[nextIndex]
    setActiveTab(nextTab.id)
    tabRefs.current.get(nextTab.id)?.focus()
  }, [])

  useEffect(() => {
    if (!open) return
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', handleEscape)
    return () => document.removeEventListener('keydown', handleEscape)
  }, [open, onClose])

  return (
    <div
      id="chat-tool-panel"
      className={cn(
        'border-l border-border/50 bg-background overflow-hidden transition-all duration-200 flex flex-col',
        open ? 'w-72 min-w-[16rem] lg:relative lg:w-72 fixed right-0 top-0 bottom-0 z-50 shadow-xl lg:shadow-none' : 'w-0 min-w-0',
      )}
    >
      {open && (
        <>
          <div className="flex items-center justify-between px-3 py-2 border-b border-border/50 shrink-0">
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              {tabs.find(t => t.id === activeTab)?.label}
            </span>
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onClose} aria-label="Close tools panel">
              <IconX className="h-3.5 w-3.5" />
            </Button>
          </div>

          <div ref={tabListRef} className="flex border-b border-border/50 shrink-0" role="tablist" aria-label="Tools">
            {tabs.map(tab => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  ref={(el) => { if (el) tabRefs.current.set(tab.id, el) }}
                  role="tab"
                  id={`tab-${tab.id}`}
                  aria-selected={activeTab === tab.id}
                  aria-controls={`tabpanel-${tab.id}`}
                  tabIndex={activeTab === tab.id ? 0 : -1}
                  onClick={() => setActiveTab(tab.id)}
                  onKeyDown={(e) => handleTabKeyDown(e, tab.id)}
                  className={cn(
                    'flex-1 flex flex-col items-center gap-0.5 py-1.5 text-[10px] font-medium transition-colors',
                    activeTab === tab.id
                      ? 'text-foreground border-b-2 border-primary'
                      : 'text-muted-foreground hover:text-foreground',
                  )}
                  title={tab.label}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {tab.label}
                </button>
              )
            })}
          </div>

          <div className="flex-1 overflow-y-auto p-3 space-y-3 text-xs scrollbar-thin">
            {activeTab === 'context' && (
              <div role="tabpanel" id="tabpanel-context" aria-labelledby="tab-context" className="space-y-2">
                <div className="text-xs text-muted-foreground">Context Inspector</div>
                <ContextInspector sessionId={sessionId} />
              </div>
            )}
            {activeTab === 'knowledge' && (
              <div role="tabpanel" id="tabpanel-knowledge" aria-labelledby="tab-knowledge">
                <KnowledgeTab
                  onOpenConversationViewer={ctx.onOpenConversationViewer}
                  onOpenSettings={ctx.onOpenSettings}
                  onOpenShortcuts={ctx.onOpenShortcuts}
                />
              </div>
            )}
            {activeTab === 'prompts' && (
              <div role="tabpanel" id="tabpanel-prompts" aria-labelledby="tab-prompts">
                <QuickPrompts onUsePrompt={(text) => ctx.setInput(text)} />
              </div>
            )}
            {activeTab === 'vision' && (
              <div role="tabpanel" id="tabpanel-vision" aria-labelledby="tab-vision">
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
                />
              </div>
            )}
            {activeTab === 'learner' && (
              <div role="tabpanel" id="tabpanel-learner" aria-labelledby="tab-learner">
                <LearnerTab
                  learnerInfo={ctx.learnerInfo}
                  learnerTraining={ctx.learnerTraining}
                  onTrainStep={ctx.onTrainStep}
                />
              </div>
            )}
            {activeTab === 'agents' && (
              <div role="tabpanel" id="tabpanel-agents" aria-labelledby="tab-agents">
                <AgentsTab
                  agents={ctx.agents}
                  currentAgent={ctx.currentAgent}
                  onSelectAgent={ctx.setCurrentAgent}
                />
              </div>
            )}
            {activeTab === 'checkpoints' && (
              <div role="tabpanel" id="tabpanel-checkpoints" aria-labelledby="tab-checkpoints">
                <CheckpointsTab
                  checkpoints={ctx.checkpoints}
                  onLoadCheckpoint={ctx.onLoadCheckpoint}
                  currentCheckpoint={ctx.currentCheckpoint}
                  availableModels={ctx.availableModels}
                  currentModel={ctx.model}
                  onSelectModel={ctx.handleSelectModel}
                  modelInfoMap={ctx.modelInfoMap}
                  souls={ctx.souls}
                  currentSoulName={ctx.currentSoul?.name}
                  onSwitchSoul={(name) => {
                    const s = ctx.souls.find(s => s.name === name)
                    if (s) ctx.handleSelectSoul(s)
                  }}
                />
              </div>
            )}
          </div>

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
