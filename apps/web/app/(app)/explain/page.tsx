'use client'

import { useState, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, Button, Textarea } from '@sloughgpt/strui'
import { chatController } from '@/lib/chat-controller'
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

  const explain = useCallback(async () => {
    if (!topic.trim()) {
      addToast('Type something to explain', 'info')
      return
    }

    setIsExplaining(true)
    setExplanation('')

    try {
      const difficultyPrompt = {
        simple: 'Explain this like I\'m 5 years old. Use simple words, analogies, and examples a child would understand.',
        normal: 'Explain this clearly and simply. Use everyday language and relatable examples.',
        detailed: 'Explain this in depth. Include technical details, examples, and real-world applications.',
      }

      const prompt = `${difficultyPrompt[difficulty]}\n\nTopic: ${topic}`

      let result = ''
      for await (const event of chatController.stream(prompt, { max_tokens: 800 })) {
        if (event.token) {
          result += event.token
          setExplanation(result)
        }
      }
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
                {DIFFICULTIES.map((d) => (
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
