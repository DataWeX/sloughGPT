'use client'

interface PersonalitySummaryProps {
  traitWeights: Record<string, Record<string, number>>
  currentSoulName: string | null
}

const GROUP_LABELS: Record<string, string> = {
  personality: 'Personality',
  cognition: 'Cognition',
  emotion: 'Emotion',
}

const GROUP_COLORS: Record<string, string> = {
  personality: 'rgb(var(--primary))',
  cognition: 'rgb(var(--chart-4))',
  emotion: 'rgb(var(--destructive))',
}

const GROUP_ICONS: Record<string, string> = {
  personality: '❤️',
  cognition: '🧠',
  emotion: '💖',
}

interface ArchetypeDef {
  label: string
  description: string
  /** Trait keys that must all be >= 0.6 in the top traits */
  required: string[]
}

const ARCHETYPES: ArchetypeDef[] = [
  { label: 'The Nurturer', description: 'Warm, empathetic, and deeply caring. Your conversations feel like talking to a close friend.', required: ['warmth', 'empathy', 'empathy_depth'] },
  { label: 'The Analyst', description: 'Logical, precise, and systematic. You break down complex problems with clarity.', required: ['abstract_reasoning', 'factual_precision'] },
  { label: 'The Professional', description: 'Precise, formal, and thorough. Perfect for technical discussions and detailed analysis.', required: ['formality', 'factual_precision'] },
  { label: 'The Artist', description: 'Creative, witty, and imaginative. Every conversation becomes a canvas.', required: ['creativity', 'humor'] },
  { label: 'The Explorer', description: 'Naturally curious and endlessly creative. Always asking "what if?"', required: ['curiosity', 'creativity'] },
  { label: 'The Leader', description: 'Direct, confident, and decisive. No beating around the bush.', required: ['confidence', 'directness'] },
  { label: 'The Companion', description: 'Fun, warm, and easygoing. The kind of presence that puts people at ease.', required: ['humor', 'warmth'] },
  { label: 'The Listener', description: 'Patient and understanding. You create space for others to be heard.', required: ['empathy', 'patience'] },
  { label: 'The Optimist', description: 'Upbeat and encouraging. You see the bright side and lift others up.', required: ['optimism'] },
]

export function deriveArchetype(traitWeights: Record<string, Record<string, number>>): { label: string; description: string } {
  const p = traitWeights.personality ?? {}
  const c = traitWeights.cognition ?? {}
  const e = traitWeights.emotion ?? {}

  // Build flat trait lookup from all 23 traits across all 3 groups
  const allTraits: Record<string, number> = {}
  for (const group of [p, c, e]) {
    for (const [key, val] of Object.entries(group)) {
      allTraits[key] = val ?? 0.5
    }
  }

  const THRESHOLD = 0.6

  // Score each matching archetype by sum of its required trait values.
  // An archetype matches when ALL its required traits are >= THRESHOLD.
  let best: { label: string; description: string; score: number } | null = null
  for (const arch of ARCHETYPES) {
    const values = arch.required.map(t => allTraits[t] ?? 0)
    if (values.every(v => v >= THRESHOLD)) {
      const score = values.reduce((s, v) => s + v, 0)
      if (!best || score > best.score) {
        best = { label: arch.label, description: arch.description, score }
      }
    }
  }

  if (best) return best

  return { label: 'The Balanced', description: 'A versatile, well-rounded personality that adapts to any conversation.' }
}

const staggerDelay = (index: number) => ({ animationDelay: `${(index + 1) * 80}ms` })

export default function PersonalitySummary({ traitWeights, currentSoulName }: PersonalitySummaryProps) {
  const archetype = deriveArchetype(traitWeights)

  const groups = ['personality', 'cognition', 'emotion'] as const

  let totalVal = 0
  let totalCount = 0
  const groupAverages: Record<string, number> = {}
  for (const group of groups) {
    const traits = traitWeights[group]
    if (!traits || typeof traits !== 'object') continue
    const entries = Object.entries(traits) as [string, number][]
    if (entries.length === 0) continue
    const sum = entries.reduce((s, [, v]) => s + v * 100, 0)
    const avg = Math.round(sum / entries.length)
    groupAverages[group] = avg
    totalVal += sum
    totalCount += entries.length
  }

  const overall = totalCount > 0 ? Math.round(totalVal / totalCount) : 0

  const ratingColor = (v: number) =>
    v >= 75 ? 'text-green-400' : v >= 55 ? 'text-amber-400' : 'text-red-400'

  return (
    <>
      <style>{`
        @keyframes scaleBar {
          from { transform: scaleX(0); }
          to { transform: scaleX(1); }
        }
        @keyframes fadeSlideUp {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes glowPulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(139, 92, 246, 0.15); }
          50% { box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.05); }
        }
        .trait-bar-inner {
          transform-origin: left;
          animation: scaleBar 0.5s ease-out forwards;
        }
        .stat-card {
          animation: fadeSlideUp 0.4s ease-out both;
        }
        .badge-glow {
          animation: glowPulse 3s ease-in-out infinite;
        }
      `}</style>

      {/* ── Archetype Badge ── */}
      {currentSoulName && (
        <div className="mb-4 pb-4 border-b border-border/40">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500/20 to-pink-500/20 border border-purple-500/30 flex items-center justify-center text-lg shrink-0">
              {GROUP_ICONS.personality}
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold capitalize">{currentSoulName}</span>
                <span className="text-[10px] font-medium px-1.5 py-0.5 rounded-full bg-primary/10 text-primary badge-glow">
                  {archetype.label}
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground mt-0.5">{archetype.description}</p>
            </div>
            <div className="flex flex-col items-center shrink-0">
              <span className="text-[9px] text-muted-foreground uppercase tracking-wider">Overall</span>
              <span className={`text-2xl font-bold ${ratingColor(overall)}`}>{overall}</span>
              <span className="text-[8px] text-muted-foreground/50">/100</span>
            </div>
          </div>
        </div>
      )}

      {/* ── Group Stats ── */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        {groups.map((group, idx) => {
          const avg = groupAverages[group] ?? 0
          const color = GROUP_COLORS[group]
          return (
            <div key={group} className="flex flex-col items-center p-3 rounded-lg bg-muted/30 border border-border/40 stat-card" style={staggerDelay(idx)}>
              <span className="text-lg mb-1">{GROUP_ICONS[group]}</span>
              <span className="text-[9px] text-muted-foreground uppercase tracking-wider mb-0.5">
                {GROUP_LABELS[group]}
              </span>
              <span className="text-lg font-bold" style={{ color }}>{avg}</span>
              <div className="w-full h-1.5 rounded-full bg-muted-foreground/10 mt-1.5 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500 trait-bar-inner"
                  style={{ width: `${avg}%`, backgroundColor: color, opacity: 0.6 }}
                />
              </div>
            </div>
          )
        })}
      </div>

      {/* ── Individual Traits ── */}
      <div className="space-y-3">
        {groups.map(group => {
          const traits = traitWeights[group]
          if (!traits || typeof traits !== 'object') return null
          const entries = Object.entries(traits) as [string, number][]
          if (entries.length === 0) return null
          const groupColor = GROUP_COLORS[group]

          return (
            <div key={group}>
              <div className="flex items-center gap-1.5 mb-1.5">
                <span className="text-[9px] font-semibold text-muted-foreground uppercase tracking-wider">
                  {GROUP_LABELS[group]}
                </span>
                <span className="text-[9px] text-muted-foreground/40">· {entries.length} traits</span>
              </div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                {entries.map(([name, value]) => {
                  const pct = Math.round(value * 100)
                  return (
                    <div key={name} className="flex items-center gap-2 py-0.5">
                      <span className="text-[10px] text-muted-foreground capitalize min-w-0 flex-1 truncate">
                        {name.replace(/_/g, ' ')}
                      </span>
                      <div className="flex items-center gap-1.5 flex-shrink-0">
                        <div className="w-12 h-1.5 rounded-full bg-muted-foreground/10 overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-300 trait-bar-inner"
                            style={{ width: `${pct}%`, backgroundColor: groupColor, opacity: 0.7 }}
                          />
                        </div>
                        <span className="text-[10px] font-mono font-medium tabular-nums w-6 text-right" style={{ color: groupColor }}>
                          {pct}
                        </span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>
    </>
  )
}
