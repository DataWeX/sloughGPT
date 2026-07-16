'use client'

import { cn } from '@sloughgpt/strui'
import type { JSX } from 'react'

export type ChatMode = 'chat' | 'write' | 'decide' | 'explain' | 'translate' | 'brainstorm' | 'wellness' | 'create' | 'read' | 'talk'

const TONES = ['Friendly', 'Professional', 'Funny', 'Short', 'Detailed'] as const
const TYPES = ['Email', 'Social Post', 'Story', 'Poem', 'Letter', 'Note'] as const
const DECIDE_STRUCTURES = ['Pros & Cons', 'Comparison', 'Simple Verdict', 'Deep Analysis'] as const
const DIFFICULTIES = ['Simple', 'Moderate', 'Expert'] as const
const LANG_PAIRS = ['EN→ES', 'EN→FR', 'EN→DE', 'EN→ZH', 'EN→JA', 'ES→EN', 'FR→EN'] as const
const BRAINSTORM_TOPICS = ['Name Ideas', 'Weekend Plans', 'Gift Ideas', 'Solve a Problem', 'Plan an Event'] as const
const WELLNESS_TYPES = ['Sleep Story', 'Meditation', 'Breathing', 'Affirmation'] as const
const CREATE_STYLES = ['Realistic', 'Cartoon', 'Watercolor', 'Sketch', 'Fantasy'] as const

interface ModeBarProps {
  mode: ChatMode
  tone: string
  type: string
  decideStructure: string
  difficulty: string
  langPair: string
  brainstormTopic: string
  wellnessType: string
  createStyle: string
  onModeChange: (mode: ChatMode) => void
  onToneChange: (tone: string) => void
  onTypeChange: (type: string) => void
  onDecideStructureChange: (structure: string) => void
  onDifficultyChange: (difficulty: string) => void
  onLangPairChange: (pair: string) => void
  onBrainstormTopicChange: (topic: string) => void
  onWellnessTypeChange: (wt: string) => void
  onCreateStyleChange: (style: string) => void
}

function SubOptionRow({ label, options, value, onChange }: {
  label: string
  options: readonly string[]
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div className="flex items-center gap-2 ml-2 pl-2 border-l border-border/20">
      <span className="text-[11px] text-muted-foreground/60">{label}:</span>
      <div className="flex items-center gap-1" role="group">
        {options.map(opt => (
          <button
            key={opt}
            onClick={() => onChange(opt)}
            className={cn(
              "px-2 py-0.5 rounded text-[11px] transition-all",
              value === opt
                ? "bg-foreground/10 text-foreground font-medium"
                : "text-muted-foreground/50 hover:text-foreground/70 hover:bg-muted/10"
            )}
            aria-pressed={value === opt}
          >
            {opt}
          </button>
        ))}
      </div>
    </div>
  )
}

function ModeBtn({ active, label, icon, onClick }: { active: boolean; label: string; icon: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "px-2.5 py-1 rounded-lg text-xs font-medium transition-all",
        active
          ? "bg-primary/10 text-primary border border-primary/20"
          : "text-muted-foreground/60 hover:text-foreground hover:bg-muted/20 border border-transparent"
      )}
      aria-pressed={active}
    >
      {icon} {label}
    </button>
  )
}

export function ModeBar({
  mode, tone, type, decideStructure, difficulty, langPair, brainstormTopic, wellnessType, createStyle,
  onModeChange, onToneChange, onTypeChange, onDecideStructureChange, onDifficultyChange, onLangPairChange, onBrainstormTopicChange, onWellnessTypeChange, onCreateStyleChange,
}: ModeBarProps): JSX.Element {
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 border-b border-border/20 bg-muted/5" role="group" aria-label="Chat mode">
      <ModeBtn active={mode === 'chat'} label="Chat" icon="💬" onClick={() => onModeChange('chat')} />
      <ModeBtn active={mode === 'write'} label="Write" icon="✍️" onClick={() => onModeChange('write')} />
      <ModeBtn active={mode === 'decide'} label="Decide" icon="⚖️" onClick={() => onModeChange('decide')} />
      <ModeBtn active={mode === 'explain'} label="Explain" icon="🔍" onClick={() => onModeChange('explain')} />
      <ModeBtn active={mode === 'translate'} label="Translate" icon="🌐" onClick={() => onModeChange('translate')} />
      <ModeBtn active={mode === 'brainstorm'} label="Brainstorm" icon="💡" onClick={() => onModeChange('brainstorm')} />
      <ModeBtn active={mode === 'wellness'} label="Wellness" icon="🧘" onClick={() => onModeChange('wellness')} />
      <ModeBtn active={mode === 'create'} label="Create" icon="🎨" onClick={() => onModeChange('create')} />
      <ModeBtn active={mode === 'read'} label="Read" icon="📄" onClick={() => onModeChange('read')} />
      <ModeBtn active={mode === 'talk'} label="Talk" icon="🎙️" onClick={() => onModeChange('talk')} />

      {mode === 'write' && (
        <>
          <SubOptionRow label="Tone" options={TONES} value={tone} onChange={onToneChange} />
          <SubOptionRow label="Type" options={TYPES} value={type} onChange={onTypeChange} />
        </>
      )}
      {mode === 'decide' && <SubOptionRow label="Output" options={DECIDE_STRUCTURES} value={decideStructure} onChange={onDecideStructureChange} />}
      {mode === 'explain' && <SubOptionRow label="Level" options={DIFFICULTIES} value={difficulty} onChange={onDifficultyChange} />}
      {mode === 'translate' && <SubOptionRow label="To" options={LANG_PAIRS} value={langPair} onChange={onLangPairChange} />}
      {mode === 'brainstorm' && <SubOptionRow label="Topic" options={BRAINSTORM_TOPICS} value={brainstormTopic} onChange={onBrainstormTopicChange} />}
      {mode === 'wellness' && <SubOptionRow label="Type" options={WELLNESS_TYPES} value={wellnessType} onChange={onWellnessTypeChange} />}
      {mode === 'create' && <SubOptionRow label="Style" options={CREATE_STYLES} value={createStyle} onChange={onCreateStyleChange} />}
    </div>
  )
}
