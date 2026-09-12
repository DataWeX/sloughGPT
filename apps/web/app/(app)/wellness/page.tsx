'use client'

import { useState, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, Button, Textarea } from '@sloughgpt/strui'
import { chatController } from '@/lib/chat-controller'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'

const _log = logger.child('wellness')

type WellnessType = 'sleep' | 'meditate' | 'journal' | 'breathe' | 'affirm'

const OPTIONS: { id: WellnessType; label: string; icon: string; description: string }[] = [
  { id: 'sleep', label: 'Sleep Story', icon: '🌙', description: 'A gentle story to help you drift off' },
  { id: 'meditate', label: 'Meditation', icon: '🧘', description: 'Guided meditation for peace' },
  { id: 'journal', label: 'Journal Prompt', icon: '📝', description: 'Reflect on your day' },
  { id: 'breathe', label: 'Breathing Exercise', icon: '💨', description: 'Calm your mind with breath' },
  { id: 'affirm', label: 'Positive Affirmation', icon: '✨', description: 'Uplifting words for your day' },
]

export default function WellnessPage() {
  const [selected, setSelected] = useState<WellnessType | null>(null)
  const [response, setResponse] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [preferences, setPreferences] = useState('')
  const addToast = useToastStore((s) => s.addToast)

  const generate = useCallback(async () => {
    if (!selected) return

    setIsGenerating(true)
    setResponse('')

    try {
      const prompts: Record<WellnessType, string> = {
        sleep: `Tell me a gentle, calming sleep story. ${preferences ? `The user wants: ${preferences}` : 'Make it about a peaceful natural setting.'} The story should be soothing, with a slow pace and calming imagery. End with the words fading into silence.`,
        meditate: `Guide me through a short meditation. ${preferences ? `Focus on: ${preferences}` : 'Focus on breathing and presence.'} Speak slowly, calmly. Include pauses marked with "...". Help me feel grounded.`,
        journal: `Give me a thoughtful journal prompt to reflect on. ${preferences ? `Theme: ${preferences}` : 'Make it about gratitude and growth.'} The prompt should inspire deep reflection.`,
        breathe: `Guide me through a breathing exercise. ${preferences ? `Style: ${preferences}` : 'Use 4-7-8 breathing.'} Give me clear instructions with counts. Make it calming and rhythmic.`,
        affirm: `Give me 3 positive affirmations for today. ${preferences ? `Theme: ${preferences}` : 'Make them empowering and kind.'} Each should be short, present-tense, and meaningful.`,
      }

      let result = ''
      for await (const event of chatController.stream(prompts[selected], { max_tokens: 500 })) {
        if (event.token) {
          result += event.token
          setResponse(result)
        }
      }
    } catch (err) {
      _log.error('Wellness generation failed', { error: err instanceof Error ? err.message : String(err) })
      addToast('Failed to generate — is a model loaded?', 'error')
    } finally {
      setIsGenerating(false)
    }
  }, [selected, preferences, addToast])

  return (
    <PageContainer title="Make Me Well">
      <div className="max-w-2xl mx-auto space-y-6">
        {/* Welcome */}
        <div className="text-center py-4">
          <p className="text-muted-foreground">Take a moment for yourself. What would feel good right now?</p>
        </div>

        {/* Options Grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {OPTIONS.map((opt) => (
            <Card
              key={opt.id}
              className={`cursor-pointer transition-all hover:border-primary ${
                selected === opt.id ? 'border-primary bg-primary/5' : ''
              }`}
              onClick={() => setSelected(opt.id)}
            >
              <CardContent className="p-4 text-center">
                <div className="text-2xl mb-2">{opt.icon}</div>
                <div className="text-sm font-medium">{opt.label}</div>
                <div className="text-[10px] text-muted-foreground mt-1">{opt.description}</div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Preferences */}
        {selected && (
          <Card>
            <CardContent className="p-4 space-y-3">
              <label className="text-xs font-medium text-muted-foreground">
                Any preferences? (optional)
              </label>
              <Textarea
                value={preferences}
                onChange={(e) => setPreferences(e.target.value)}
                placeholder={
                  selected === 'sleep' ? 'e.g. A story about the ocean, a forest, or a starry night...'
                  : selected === 'meditate' ? 'e.g. Focus on anxiety relief, self-love, or letting go...'
                  : selected === 'journal' ? 'e.g. Focus on wins this week, lessons learned...'
                  : selected === 'breathe' ? 'e.g. Box breathing, 4-7-8, or calm breathing...'
                  : 'e.g. Focus on confidence, gratitude, or inner peace...'
                }
                className="text-sm resize-none"
                rows={2}
              />
              <Button
                onClick={generate}
                disabled={isGenerating}
                className="w-full"
              >
                {isGenerating ? 'Generating...' : 'Begin'}
              </Button>
            </CardContent>
          </Card>
        )}

        {/* Response */}
        {response && (
          <Card>
            <CardContent className="p-6">
              <div className="prose prose-sm max-w-none text-sm whitespace-pre-wrap leading-relaxed">
                {response}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </PageContainer>
  )
}
