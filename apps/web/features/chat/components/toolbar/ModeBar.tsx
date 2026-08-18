'use client'

import { useState, type JSX } from 'react'
import { cn, IconChevronDown } from '@sloughgpt/strui'
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from '@sloughgpt/strui'

export type ChatMode = 'chat' | 'write' | 'decide' | 'explain' | 'translate' | 'brainstorm' | 'wellness' | 'create' | 'read' | 'talk'

const MODES: { value: ChatMode; label: string; icon: string }[] = [
  { value: 'chat', label: 'Chat', icon: '💬' },
  { value: 'write', label: 'Write', icon: '✍️' },
  { value: 'decide', label: 'Decide', icon: '⚖️' },
  { value: 'explain', label: 'Explain', icon: '🔍' },
  { value: 'translate', label: 'Translate', icon: '🌐' },
  { value: 'brainstorm', label: 'Brainstorm', icon: '💡' },
  { value: 'wellness', label: 'Wellness', icon: '🧘' },
  { value: 'create', label: 'Create', icon: '🎨' },
  { value: 'read', label: 'Read', icon: '📄' },
  { value: 'talk', label: 'Talk', icon: '🎙️' },
]

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

function SubOptionPill({ active, label, onClick }: { active: boolean; label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "px-2 py-0.5 rounded text-[11px] transition-all",
        active
          ? "bg-foreground/10 text-foreground font-medium"
          : "text-muted-foreground/50 hover:text-foreground/70 hover:bg-muted/10"
      )}
      aria-pressed={active}
    >
      {label}
    </button>
  )
}

function SubOptions({ mode, tone, type, decideStructure, difficulty, langPair, brainstormTopic, wellnessType, createStyle,
  onToneChange, onTypeChange, onDecideStructureChange, onDifficultyChange, onLangPairChange, onBrainstormTopicChange, onWellnessTypeChange, onCreateStyleChange,
}: Pick<ModeBarProps, 'mode' | 'tone' | 'type' | 'decideStructure' | 'difficulty' | 'langPair' | 'brainstormTopic' | 'wellnessType' | 'createStyle' | 'onToneChange' | 'onTypeChange' | 'onDecideStructureChange' | 'onDifficultyChange' | 'onLangPairChange' | 'onBrainstormTopicChange' | 'onWellnessTypeChange' | 'onCreateStyleChange'>) {
  if (mode === 'write') {
    return (
      <div className="flex items-center gap-3 px-3 py-1 border-b border-border/20 bg-muted/5">
        <div className="flex items-center gap-1">
          <span className="text-[11px] text-muted-foreground/60">Tone</span>
          {TONES.map(t => <SubOptionPill key={t} active={tone === t} label={t} onClick={() => onToneChange(t)} />)}
        </div>
        <div className="w-px h-3 bg-border/20" />
        <div className="flex items-center gap-1">
          <span className="text-[11px] text-muted-foreground/60">Type</span>
          {TYPES.map(t => <SubOptionPill key={t} active={type === t} label={t} onClick={() => onTypeChange(t)} />)}
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
    <div className="flex items-center gap-3 px-3 py-1 border-b border-border/20 bg-muted/5">
      {groups.map((g, i) => (
        <div key={g.label} className="flex items-center gap-1">
          {i > 0 && <div className="w-px h-3 bg-border/20 mr-2" />}
          <span className="text-[11px] text-muted-foreground/60">{g.label}</span>
          {g.options.map(opt => (
            <SubOptionPill key={opt} active={g.value === opt} label={opt} onClick={() => g.onChange(opt)} />
          ))}
        </div>
      ))}
    </div>
  )
}

export function ModeBar({
  mode, tone, type, decideStructure, difficulty, langPair, brainstormTopic, wellnessType, createStyle,
  onModeChange, onToneChange, onTypeChange, onDecideStructureChange, onDifficultyChange, onLangPairChange, onBrainstormTopicChange, onWellnessTypeChange, onCreateStyleChange,
}: ModeBarProps): JSX.Element {
  const [open, setOpen] = useState(false)
  const current = MODES.find(m => m.value === mode) ?? MODES[0]

  return (
    <>
      <div className="flex items-center px-3 py-1.5 border-b border-border/20 bg-muted/5" role="group" aria-label="Chat mode">
        <DropdownMenu open={open} onOpenChange={setOpen}>
          <DropdownMenuTrigger className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium text-muted-foreground/70 hover:text-foreground hover:bg-muted/20 transition-all border border-transparent hover:border-border/20">
            <span>{current.icon}</span>
            <span>{current.label}</span>
            <IconChevronDown className="w-3 h-3 opacity-40" />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="w-44">
            <DropdownMenuLabel className="text-[11px] text-muted-foreground/60">Mode</DropdownMenuLabel>
            <DropdownMenuRadioGroup value={mode} onValueChange={(v) => {
              if (MODES.some(m => m.value === v)) onModeChange(v as ChatMode)
            }}>
              {MODES.map(m => (
                <DropdownMenuRadioItem key={m.value} value={m.value} className="text-xs gap-2">
                  <span>{m.icon}</span>
                  <span>{m.label}</span>
                </DropdownMenuRadioItem>
              ))}
            </DropdownMenuRadioGroup>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <SubOptions
        mode={mode} tone={tone} type={type} decideStructure={decideStructure}
        difficulty={difficulty} langPair={langPair} brainstormTopic={brainstormTopic}
        wellnessType={wellnessType} createStyle={createStyle}
        onToneChange={onToneChange} onTypeChange={onTypeChange}
        onDecideStructureChange={onDecideStructureChange} onDifficultyChange={onDifficultyChange}
        onLangPairChange={onLangPairChange} onBrainstormTopicChange={onBrainstormTopicChange}
        onWellnessTypeChange={onWellnessTypeChange} onCreateStyleChange={onCreateStyleChange}
      />
    </>
  )
}
