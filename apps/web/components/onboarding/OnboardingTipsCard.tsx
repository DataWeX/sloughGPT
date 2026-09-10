'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, cn } from '@sloughgpt/strui'

interface Tip {
  id: string
  category: string
  title: string
  body: string
}

const TIPS: Tip[] = [
  { id: 't1', category: 'Chat', title: 'Use /clear to reset', body: 'Type /clear in chat to reset the conversation context when switching topics.' },
  { id: 't2', category: 'Knowledge', title: 'Be specific', body: 'Instead of "I like coffee", try "I prefer oat milk lattes from Blue Bottle". Specific details help the AI give better responses.' },
  { id: 't3', category: 'Companion', title: 'Start with a preset', body: 'Presets are a great starting point. Pick one close to what you want, then fine-tune individual traits.' },
  { id: 't4', category: 'Voice', title: 'Speak naturally', body: 'The voice system works best with natural speech. No need to speak slowly or enunciate excessively.' },
  { id: 't5', category: 'Images', title: 'Describe style explicitly', body: 'Add style keywords like "watercolor", "photorealistic", "anime", or "oil painting" to get the look you want.' },
  { id: 't6', category: 'Files', title: 'PDFs work best', body: 'PDF documents are parsed and indexed automatically. The AI can answer questions about their content.' },
  { id: 't7', category: 'General', title: 'Keyboard shortcuts', body: 'Press Ctrl+K to open the command palette. Use arrow keys to navigate between sections.' },
  { id: 't8', category: 'General', title: 'Dark mode', body: 'Toggle dark mode in Settings or press Ctrl+Shift+D. Your preference is saved automatically.' },
  { id: 't9', category: 'Training', title: 'Rate responses', body: 'Click the thumbs up/down on AI responses. This feedback directly improves future responses.' },
  { id: 't10', category: 'Chat', title: 'Multi-line messages', body: 'Press Shift+Enter to add a new line without sending. Great for longer prompts.' },
]

const CATEGORIES = [...new Set(TIPS.map(t => t.category))]

interface OnboardingTipsCardProps {
  onNavigate?: (link: string) => void
}

export function OnboardingTipsCard({ onNavigate }: OnboardingTipsCardProps) {
  const [category, setCategory] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)

  const filtered = category ? TIPS.filter(t => t.category === category) : TIPS

  return (
    <Card data-testid="onboarding-tips">
      <CardHeader>
        <CardTitle className="text-base">Tips & Tricks</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex gap-1 mb-3 flex-wrap">
          <Button
            size="sm"
            variant={category === null ? 'default' : 'ghost'}
            className="text-[9px]"
            onClick={() => setCategory(null)}
          >
            All
          </Button>
          {CATEGORIES.map(c => (
            <Button
              key={c}
              size="sm"
              variant={category === c ? 'default' : 'ghost'}
              className="text-[9px]"
              onClick={() => setCategory(c)}
            >
              {c}
            </Button>
          ))}
        </div>
        <div className="space-y-1 max-h-72 overflow-y-auto">
          {filtered.map(tip => (
            <button
              key={tip.id}
              className={cn(
                'w-full text-left p-2 rounded transition-colors',
                expanded === tip.id ? 'bg-primary/5' : 'hover:bg-muted/50'
              )}
              onClick={() => setExpanded(expanded === tip.id ? null : tip.id)}
              data-testid={`tip-${tip.id}`}
            >
              <div className="flex items-center justify-between">
                <div className="text-xs font-medium">{tip.title}</div>
                <span className="text-[9px] text-muted-foreground px-1.5 py-0.5 bg-muted rounded">{tip.category}</span>
              </div>
              {expanded === tip.id && (
                <div className="text-[10px] text-muted-foreground mt-1">{tip.body}</div>
              )}
            </button>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
