'use client'

import { useState, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, Button, Textarea } from '@sloughgpt/strui'
import { generateTool } from '@/lib/tools-controller'
import { useToolProfile } from '@/lib/use-tool-profile'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'

const _log = logger.child('explain')

type Difficulty = 'simple' | 'normal' | 'detailed'

const DIFFICULTIES: { id: Difficulty; label: string; description: string }[] = [
  { id: 'simple', label: 'Simple', description: 'Like explaining to a child' },
  { id: 'normal', label: 'Normal', description: 'Clear and straightforward' },
  { id: 'detailed', label: 'Detailed', description: 'In-depth with examples' },
]

export default function ExplainPage() {
  const [topic, setTopic] = useState('')
  const [difficulty, setDifficulty] = useState<Difficulty>('normal')
  const [explanation, setExplanation] = useState('')
  const [isExplaining, setIsExplaining] = useState(false)
  const addToast = useToastStore((s) => s.addToast)
  const profile = useToolProfile('explain')

  const difficulties = (profile?.options?.difficulty?.length
    ? profile.options.difficulty
    : DIFFICULTIES) as { id: Difficulty; label: string; description: string }[]

  const explain = useCallback(async () => {
    if (!topic.trim()) {
      addToast('Type something to explain', 'info')
      return
    }

    setIsExplaining(true)
    setExplanation('')

    try {
      await generateTool(
        'explain',
        { topic, difficulty },
        {
          onToken: (token) => setExplanation(token),
          onError: (msg) => {
            _log.error('Explanation failed', { error: msg })
            addToast('Explanation failed — is a model loaded?', 'error')
          },
        },
        { max_tokens: 800 },
      )
    } catch (err) {
      _log.error('Explanation failed', { error: err instanceof Error ? err.message : String(err) })
      addToast('Explanation failed — is a model loaded?', 'error')
    } finally {
      setIsExplaining(false)
    }
  }, [topic, difficulty, addToast])

  const copyToClipboard = useCallback(() => {
    if (explanation) {
      navigator.clipboard.writeText(explanation)
      addToast('Copied to clipboard', 'success')
    }
  }, [explanation, addToast])

  return (
    <PageContainer title="Explain Things Simply">
      <div className="max-w-3xl mx-auto space-y-4">
        {/* Input */}
        <Card>
          <CardContent className="p-4 space-y-4">
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1 block">
                What do you want explained?
              </label>
              <Textarea
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="e.g. How does the internet work?"
                className="text-sm resize-none"
                rows={2}
              />
            </div>

            <div>
              <label className="text-xs font-medium text-muted-foreground mb-2 block">
                How detailed?
              </label>
              <div className="flex gap-2">
                {difficulties.map((d) => (
                  <Button
                    key={d.id}
                    variant={difficulty === d.id ? 'default' : 'outline'}
                    onClick={() => setDifficulty(d.id)}
                    className="flex-1"
                  >
                    <div className="text-left">
                      <div className="text-xs font-medium">{d.label}</div>
                      <div className="text-[10px] text-muted-foreground">{d.description}</div>
                    </div>
                  </Button>
                ))}
              </div>
            </div>

            <Button
              onClick={explain}
              disabled={isExplaining || !topic.trim()}
              className="w-full"
            >
              {isExplaining ? 'Explaining...' : 'Explain'}
            </Button>
          </CardContent>
        </Card>

        {/* Explanation */}
        {explanation && (
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs font-medium text-muted-foreground">Explanation</label>
                <Button variant="outline" size="sm" onClick={copyToClipboard}>
                  Copy
                </Button>
              </div>
              <div className="prose prose-sm max-w-none text-sm whitespace-pre-wrap">
                {explanation}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </PageContainer>
  )
}
