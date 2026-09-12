'use client'

import { useState, memo, type JSX, type ReactNode } from 'react'
import { IconChat, IconEdit, IconBrain, IconVision, IconSearch, IconBolt, IconDocument, IconMic, IconSparkle, cn } from '@sloughgpt/strui'
import { Waves } from 'lucide-react'

export type ChatMode = 'chat' | 'write' | 'rewrite' | 'decide' | 'explain' | 'translate' | 'brainstorm' | 'wellness' | 'create' | 'read' | 'talk'

const MODES: { value: ChatMode; label: string; icon: ReactNode; color: string }[] = [
  { value: 'chat', label: 'Chat', icon: <IconChat className="h-3.5 w-3.5" />, color: 'bg-primary/10 text-primary border-primary/20' },
  { value: 'write', label: 'Write', icon: <IconEdit className="h-3.5 w-3.5" />, color: 'bg-violet-500/10 text-violet-500 border-violet-500/20' },
  { value: 'rewrite', label: 'Rewrite', icon: <IconSparkle className="h-3.5 w-3.5" />, color: 'bg-amber-500/10 text-amber-500 border-amber-500/20' },
  { value: 'translate', label: 'Translate', icon: <IconVision className="h-3.5 w-3.5" />, color: 'bg-blue-500/10 text-blue-500 border-blue-500/20' },
  { value: 'brainstorm', label: 'Brainstorm', icon: <IconBolt className="h-3.5 w-3.5" />, color: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20' },
  { value: 'decide', label: 'Decide', icon: <IconBrain className="h-3.5 w-3.5" />, color: 'bg-rose-500/10 text-rose-500 border-rose-500/20' },
  { value: 'explain', label: 'Explain', icon: <IconSearch className="h-3.5 w-3.5" />, color: 'bg-cyan-500/10 text-cyan-500 border-cyan-500/20' },
  { value: 'wellness', label: 'Wellness', icon: <Waves className="h-3.5 w-3.5" />, color: 'bg-teal-500/10 text-teal-500 border-teal-500/20' },
  { value: 'create', label: 'Create', icon: <IconVision className="h-3.5 w-3.5" />, color: 'bg-pink-500/10 text-pink-500 border-pink-500/20' },
  { value: 'read', label: 'Read', icon: <IconDocument className="h-3.5 w-3.5" />, color: 'bg-orange-500/10 text-orange-500 border-orange-500/20' },
  { value: 'talk', label: 'Talk', icon: <IconMic className="h-3.5 w-3.5" />, color: 'bg-indigo-500/10 text-indigo-500 border-indigo-500/20' },
]

const TONES = ['Friendly', 'Professional', 'Funny', 'Short', 'Detailed'] as const
const TYPES = ['Email', 'Social Post', 'Story', 'Poem', 'Letter', 'Note'] as const
const DECIDE_STRUCTURES = ['Pros & Cons', 'Comparison', 'Simple Verdict', 'Deep Analysis'] as const
const DIFFICULTIES = ['Simple', 'Moderate', 'Expert'] as const
const LANG_PAIRS = ['EN→ES', 'EN→FR', 'EN→DE', 'EN→ZH', 'EN→JA', 'ES→EN', 'FR→EN'] as const
const BRAINSTORM_TOPICS = ['Name Ideas', 'Weekend Plans', 'Gift Ideas', 'Solve a Problem', 'Plan an Event'] as const
const WELLNESS_TYPES = ['Sleep Story', 'Meditation', 'Breathing', 'Affirmation'] as const
const CREATE_STYLES = ['Realistic', 'Cartoon', 'Watercolor', 'Sketch', 'Fantasy'] as const
const REWRITE_OPTIONS = ['Fix Grammar', 'Make Shorter', 'Make Friendlier', 'Make Professional', 'Sound Like Me'] as const

interface ModeBarProps {
  mode: ChatMode
  tone: string
  type: string
  rewriteStyle: string
  decideStructure: string
  difficulty: string
  langPair: string
  brainstormTopic: string
  wellnessType: string
  createStyle: string
  onModeChange: (mode: ChatMode) => void
  onToneChange: (tone: string) => void
  onTypeChange: (type: string) => void
  onRewriteStyleChange: (style: string) => void
  onDecideStructureChange: (structure: string) => void
  onDifficultyChange: (difficulty: string) => void
  onLangPairChange: (pair: string) => void
  onBrainstormTopicChange: (topic: string) => void
  onWellnessTypeChange: (wt: string) => void
  onCreateStyleChange: (style: string) => void
}

function SubOptionPill({ active, label, onClick }: { active: boolean; label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "px-2 py-0.5 rounded-md text-[11px] transition-all duration-150",
        active
          ? "bg-foreground/10 text-foreground font-medium shadow-sm"
          : "text-muted-foreground/40 hover:text-foreground/60 hover:bg-muted/30"
      )}
      aria-pressed={active}
    >
      {label}
    </button>
  )
}

function SubOptions({ mode, tone, type, rewriteStyle, decideStructure, difficulty, langPair, brainstormTopic, wellnessType, createStyle,
  onToneChange, onTypeChange, onRewriteStyleChange, onDecideStructureChange, onDifficultyChange, onLangPairChange, onBrainstormTopicChange, onWellnessTypeChange, onCreateStyleChange,
}: Pick<ModeBarProps, 'mode' | 'tone' | 'type' | 'rewriteStyle' | 'decideStructure' | 'difficulty' | 'langPair' | 'brainstormTopic' | 'wellnessType' | 'createStyle' | 'onToneChange' | 'onTypeChange' | 'onRewriteStyleChange' | 'onDecideStructureChange' | 'onDifficultyChange' | 'onLangPairChange' | 'onBrainstormTopicChange' | 'onWellnessTypeChange' | 'onCreateStyleChange'>) {
  if (mode === 'write') {
    return (
      <div className="flex items-center gap-3 px-3 py-1.5 border-b border-border/15 bg-muted/30">
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-muted-foreground/40 font-medium">Tone</span>
          {TONES.map(t => <SubOptionPill key={t} active={tone === t} label={t} onClick={() => onToneChange(t)} />)}
        </div>
        <div className="w-px h-3 bg-border/15" />
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-muted-foreground/40 font-medium">Type</span>
          {TYPES.map(t => <SubOptionPill key={t} active={type === t} label={t} onClick={() => onTypeChange(t)} />)}
        </div>
      </div>
    )
  }

  if (mode === 'rewrite') {
    return (
      <div className="flex items-center gap-3 px-3 py-1.5 border-b border-border/15 bg-muted/30">
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-muted-foreground/40 font-medium">Action</span>
          {REWRITE_OPTIONS.map(o => <SubOptionPill key={o} active={rewriteStyle === o} label={o} onClick={() => onRewriteStyleChange(o)} />)}
        </div>
      </div>
    )
  }

  const optionMap: Record<string, { label: string; options: readonly string[]; value: string; onChange: (v: string) => void }[]> = {
    decide: [{ label: 'Output', options: DECIDE_STRUCTURES, value: decideStructure, onChange: onDecideStructureChange }],
    explain: [{ label: 'Level', options: DIFFICULTIES, value: difficulty, onChange: onDifficultyChange }],
    translate: [{ label: 'To', options: LANG_PAIRS, value: langPair, onChange: onLangPairChange }],
    brainstorm: [{ label: 'Topic', options: BRAINSTORM_TOPICS, value: brainstormTopic, onChange: onBrainstormTopicChange }],
    wellness: [{ label: 'Type', options: WELLNESS_TYPES, value: wellnessType, onChange: onWellnessTypeChange }],
    create: [{ label: 'Style', options: CREATE_STYLES, value: createStyle, onChange: onCreateStyleChange }],
  }

  const groups = optionMap[mode]
  if (!groups) return null

  return (
    <div className="flex items-center gap-3 px-3 py-1.5 border-b border-border/15 bg-muted/30">
      {groups.map((g, i) => (
        <div key={g.label} className="flex items-center gap-1">
          {i > 0 && <div className="w-px h-3 bg-border/15 mr-2" />}
          <span className="text-[10px] text-muted-foreground/40 font-medium">{g.label}</span>
          {g.options.map(opt => (
            <SubOptionPill key={opt} active={g.value === opt} label={opt} onClick={() => g.onChange(opt)} />
          ))}
        </div>
      ))}
    </div>
  )
}

export const ModeBar = memo(function ModeBar({
  mode, tone, type, rewriteStyle, decideStructure, difficulty, langPair, brainstormTopic, wellnessType, createStyle,
  onModeChange, onToneChange, onTypeChange, onRewriteStyleChange, onDecideStructureChange, onDifficultyChange, onLangPairChange, onBrainstormTopicChange, onWellnessTypeChange, onCreateStyleChange,
}: ModeBarProps): JSX.Element {
  const current = MODES.find(m => m.value === mode) ?? MODES[0]

  return (
    <>
      <div className="flex items-center gap-1 px-3 py-1.5 border-b border-border/15 bg-muted/20 overflow-x-auto scrollbar-none" role="tablist" aria-label="Chat mode">
        {MODES.map((m) => (
          <button
            key={m.value}
            type="button"
            role="tab"
            aria-selected={m.value === mode}
            onClick={() => onModeChange(m.value)}
            className={cn(
              "flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-medium transition-all duration-150 whitespace-nowrap shrink-0",
              m.value === mode
                ? cn("shadow-sm border", m.color)
                : "text-muted-foreground/40 hover:text-foreground/60 hover:bg-muted/30 border border-transparent"
            )}
          >
            <span className="shrink-0">{m.icon}</span>
            <span>{m.label}</span>
          </button>
        ))}
      </div>

      <SubOptions
        mode={mode} tone={tone} type={type} rewriteStyle={rewriteStyle} decideStructure={decideStructure}
        difficulty={difficulty} langPair={langPair} brainstormTopic={brainstormTopic}
        wellnessType={wellnessType} createStyle={createStyle}
        onToneChange={onToneChange} onTypeChange={onTypeChange} onRewriteStyleChange={onRewriteStyleChange}
        onDecideStructureChange={onDecideStructureChange} onDifficultyChange={onDifficultyChange}
        onLangPairChange={onLangPairChange} onBrainstormTopicChange={onBrainstormTopicChange}
        onWellnessTypeChange={onWellnessTypeChange} onCreateStyleChange={onCreateStyleChange}
      />
    </>
  )
})
