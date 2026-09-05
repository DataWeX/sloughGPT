'use client'

import { memo, useMemo } from 'react'
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts'

interface TraitRadarChartProps {
  data: Record<string, number>
  label: string
  color: string
}

const LABEL_MAP: Record<string, string> = {
  warmth: 'Warmth',
  creativity: 'Creativity',
  empathy: 'Empathy',
  formality: 'Formality',
  humor: 'Humor',
  patience: 'Patience',
  confidence: 'Confidence',
  curiosity: 'Curiosity',
  directness: 'Directness',
  optimism: 'Optimism',
  pattern_recognition: 'Pattern Rec',
  long_context_handling: 'Long Ctx',
  abstract_reasoning: 'Abstract',
  factual_precision: 'Precision',
  creative_divergence: 'Divergence',
  systematic_planning: 'Planning',
  metacognitive_awareness: 'Meta Cog',
  learning_adaptability: 'Adaptability',
  empathy_depth: 'Empathy Dp',
  mood_responsiveness: 'Mood Resp',
  tone_flexibility: 'Tone Flex',
  sentiment_awareness: 'Sentiment',
  distress_handling: 'Distress',
}

export default memo(function TraitRadarChart({ data, label, color }: TraitRadarChartProps) {
  const entries = Object.entries(data)

  const chartData = useMemo(() => entries.map(([name, value]) => ({
    trait: LABEL_MAP[name] || name.replace(/_/g, ' '),
    value: Math.round(value * 100),
    fullName: name.replace(/_/g, ' '),
  })), [entries])

  if (entries.length === 0) return null

  return (
    <div className="flex flex-col items-center">
      <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">
        {label}
      </span>
      <ResponsiveContainer width="100%" height={180}>
        <RadarChart data={chartData} cx="50%" cy="50%" outerRadius="65%">
          <PolarGrid stroke="hsl(var(--border))" strokeOpacity={0.4} />
          <PolarAngleAxis
            dataKey="trait"
            tick={{ fontSize: 8, fill: 'hsl(var(--muted-foreground))' }}
            tickLine={false}
          />
          <PolarRadiusAxis
            angle={30}
              domain={[0, 100]}
              tick={false}
              axisLine={false}
            />
            <Radar
              name={label}
              dataKey="value"
              stroke={color}
              fill={color}
              fillOpacity={0.15}
              strokeWidth={1.5}
              dot={false}
          />
        </RadarChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap justify-center gap-x-3 gap-y-0.5 mt-1">
        {chartData.map(d => (
          <span key={d.trait} className="text-[9px] text-muted-foreground">
            {d.trait}: <span className="font-mono font-medium">{d.value}</span>
          </span>
        ))}
      </div>
    </div>
  )
})
