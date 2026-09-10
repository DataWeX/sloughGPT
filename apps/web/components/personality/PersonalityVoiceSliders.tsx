import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@sloughgpt/strui'

const VOICE_LABELS: Record<string, string> = {
  formality: 'Formality',
  warmth: 'Warmth',
  confidence: 'Confidence',
  humor: 'Humor',
  verbosity: 'Verbosity',
  empathy: 'Empathy',
}

export interface PersonalityVoiceSlidersProps {
  voice: Record<string, number>
  onVoiceChange: (key: string, value: number) => void
}

export function PersonalityVoiceSliders({ voice, onVoiceChange }: PersonalityVoiceSlidersProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Voice</CardTitle>
        <CardDescription>How the system sounds when communicating</CardDescription>
      </CardHeader>
      <CardContent className="space-y-2 sm:space-y-3">
        {Object.entries(voice).map(([key, value]) => (
          <div key={key} className="flex items-center gap-2 sm:gap-4">
            <label className="text-xs sm:text-sm w-24 sm:w-32 shrink-0">{VOICE_LABELS[key] || key}</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={value}
              onChange={(e) => onVoiceChange(key, parseFloat(e.target.value))}
              className="flex-1 min-w-0"
            />
            <span className="text-[10px] sm:text-xs text-muted-foreground w-8 sm:w-10 text-right shrink-0">{(value * 100).toFixed(0)}%</span>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
