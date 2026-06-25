'use client'

import { cn } from '@/lib/cn'
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
        <div className="flex items-center gap-2 ml-2 pl-2 border-l border-border/20">
          <div className="flex items-center gap-1" role="group" aria-label="Writing tone">
            {TONES.map(t => (
              <button
                key={t}
                onClick={() => onToneChange(t)}
                className={cn(
                  "px-2 py-0.5 rounded text-[11px] transition-all",
                  tone === t
                    ? "bg-foreground/10 text-foreground font-medium"
                    : "text-muted-foreground/50 hover:text-foreground/70 hover:bg-muted/10"
                )}
                aria-pressed={tone === t}
              >
                {t}
              </button>
            ))}
          </div>
          <span className="text-muted-foreground/20">|</span>
          <div className="flex items-center gap-1" role="group" aria-label="Content type">
            {TYPES.map(ty => (
              <button
                key={ty}
                onClick={() => onTypeChange(ty)}
                className={cn(
                  "px-2 py-0.5 rounded text-[11px] transition-all",
                  type === ty
                    ? "bg-foreground/10 text-foreground font-medium"
                    : "text-muted-foreground/50 hover:text-foreground/70 hover:bg-muted/10"
                )}
                aria-pressed={type === ty}
              >
                {ty}
              </button>
            ))}
          </div>
        </div>
      )}

      {mode === 'decide' && (
        <div className="flex items-center gap-2 ml-2 pl-2 border-l border-border/20">
          <span className="text-[11px] text-muted-foreground/60">Output:</span>
          <div className="flex items-center gap-1" role="group" aria-label="Decision format">
            {DECIDE_STRUCTURES.map(s => (
              <button
                key={s}
                onClick={() => onDecideStructureChange(s)}
                className={cn(
                  "px-2 py-0.5 rounded text-[11px] transition-all",
                  decideStructure === s
                    ? "bg-foreground/10 text-foreground font-medium"
                    : "text-muted-foreground/50 hover:text-foreground/70 hover:bg-muted/10"
                )}
                aria-pressed={decideStructure === s}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {mode === 'explain' && (
        <div className="flex items-center gap-2 ml-2 pl-2 border-l border-border/20">
          <span className="text-[11px] text-muted-foreground/60">Level:</span>
          <div className="flex items-center gap-1" role="group" aria-label="Difficulty level">
            {DIFFICULTIES.map(d => (
              <button
                key={d}
                onClick={() => onDifficultyChange(d)}
                className={cn(
                  "px-2 py-0.5 rounded text-[11px] transition-all",
                  difficulty === d
                    ? "bg-foreground/10 text-foreground font-medium"
                    : "text-muted-foreground/50 hover:text-foreground/70 hover:bg-muted/10"
                )}
                aria-pressed={difficulty === d}
              >
                {d}
              </button>
            ))}
          </div>
        </div>
      )}

      {mode === 'translate' && (
        <div className="flex items-center gap-2 ml-2 pl-2 border-l border-border/20">
          <span className="text-[11px] text-muted-foreground/60">To:</span>
          <div className="flex items-center gap-1" role="group" aria-label="Language pair">
            {LANG_PAIRS.map(p => (
              <button
                key={p}
                onClick={() => onLangPairChange(p)}
                className={cn(
                  "px-2 py-0.5 rounded text-[11px] transition-all",
                  langPair === p
                    ? "bg-foreground/10 text-foreground font-medium"
                    : "text-muted-foreground/50 hover:text-foreground/70 hover:bg-muted/10"
                )}
                aria-pressed={langPair === p}
              >
                {p}
              </button>
            ))}
          </div>
        </div>
      )}

      {mode === 'brainstorm' && (
        <div className="flex items-center gap-2 ml-2 pl-2 border-l border-border/20">
          <span className="text-[11px] text-muted-foreground/60">Topic:</span>
          <div className="flex items-center gap-1" role="group" aria-label="Brainstorm topic">
            {BRAINSTORM_TOPICS.map(t => (
              <button
                key={t}
                onClick={() => onBrainstormTopicChange(t)}
                className={cn(
                  "px-2 py-0.5 rounded text-[11px] transition-all",
                  brainstormTopic === t
                    ? "bg-foreground/10 text-foreground font-medium"
                    : "text-muted-foreground/50 hover:text-foreground/70 hover:bg-muted/10"
                )}
                aria-pressed={brainstormTopic === t}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
      )}

      {mode === 'wellness' && (
        <div className="flex items-center gap-2 ml-2 pl-2 border-l border-border/20">
          <span className="text-[11px] text-muted-foreground/60">Type:</span>
          <div className="flex items-center gap-1" role="group" aria-label="Wellness type">
            {WELLNESS_TYPES.map(w => (
              <button
                key={w}
                onClick={() => onWellnessTypeChange(w)}
                className={cn(
                  "px-2 py-0.5 rounded text-[11px] transition-all",
                  wellnessType === w
                    ? "bg-foreground/10 text-foreground font-medium"
                    : "text-muted-foreground/50 hover:text-foreground/70 hover:bg-muted/10"
                )}
                aria-pressed={wellnessType === w}
              >
                {w}
              </button>
            ))}
          </div>
        </div>
      )}

      {mode === 'create' && (
        <div className="flex items-center gap-2 ml-2 pl-2 border-l border-border/20">
          <span className="text-[11px] text-muted-foreground/60">Style:</span>
          <div className="flex items-center gap-1" role="group" aria-label="Image style">
            {CREATE_STYLES.map(s => (
              <button
                key={s}
                onClick={() => onCreateStyleChange(s)}
                className={cn(
                  "px-2 py-0.5 rounded text-[11px] transition-all",
                  createStyle === s
                    ? "bg-foreground/10 text-foreground font-medium"
                    : "text-muted-foreground/50 hover:text-foreground/70 hover:bg-muted/10"
                )}
                aria-pressed={createStyle === s}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
