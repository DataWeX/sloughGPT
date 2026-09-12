'use client'

import { useState, useCallback, useRef } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, Button, Textarea } from '@sloughgpt/strui'
import { generateTool } from '@/lib/tools-controller'
import { useToolProfile } from '@/lib/use-tool-profile'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'

const _log = logger.child('writing-assistant')

type Tone = 'friendly' | 'professional' | 'funny' | 'short' | 'detailed'
type WriteType = 'email' | 'social' | 'story' | 'poem' | 'letter' | 'note'

const TONES: { id: Tone; label: string }[] = [
  { id: 'friendly', label: 'Friendly' },
  { id: 'professional', label: 'Professional' },
  { id: 'funny', label: 'Funny' },
  { id: 'short', label: 'Short' },
  { id: 'detailed', label: 'Detailed' },
]

const TYPES: { id: WriteType; label: string }[] = [
  { id: 'email', label: 'Email' },
  { id: 'social', label: 'Social Post' },
  { id: 'story', label: 'Story' },
  { id: 'poem', label: 'Poem' },
  { id: 'letter', label: 'Letter' },
  { id: 'note', label: 'Note' },
]

export default function WritingAssistantPage() {
  const [input, setInput] = useState('')
  const [output, setOutput] = useState('')
  const [tone, setTone] = useState<Tone>('professional')
  const [type, setType] = useState<WriteType>('email')
  const [isGenerating, setIsGenerating] = useState(false)
  const addToast = useToastStore((s) => s.addToast)
  const outputRef = useRef<HTMLTextAreaElement>(null)
  const profile = useToolProfile('writing')

  const tones: { id: Tone; label: string }[] = profile?.options?.tone?.length
    ? profile.options.tone.map((o) => ({ id: o.id as Tone, label: o.label }))
    : TONES
  const types: { id: WriteType; label: string }[] = profile?.options?.type?.length
    ? profile.options.type.map((o) => ({ id: o.id as WriteType, label: o.label }))
    : TYPES

  const generate = useCallback(async (action: 'write' | 'rewrite' | 'shorter' | 'funnier') => {
    if (!input.trim()) {
      addToast('Write something first', 'info')
      return
    }

    setIsGenerating(true)
    setOutput('')

    const text = action === 'write' ? input : output || input

    try {
      await generateTool(
        'writing',
        { text, action, tone, type },
        {
          onToken: (token) => setOutput(token),
          onError: (msg) => {
            _log.error('Generation failed', { error: msg })
            addToast('Failed to generate — is a model loaded?', 'error')
          },
        },
        { max_tokens: 500 },
      )
    } catch (err) {
      _log.error('Generation failed', { error: err instanceof Error ? err.message : String(err) })
      addToast('Failed to generate — is a model loaded?', 'error')
    } finally {
      setIsGenerating(false)
    }
  }, [input, output, tone, type, addToast])

  const copyToClipboard = useCallback(() => {
    if (output) {
      navigator.clipboard.writeText(output)
      addToast('Copied to clipboard', 'success')
    }
  }, [output, addToast])

  return (
    <PageContainer title="Writing Assistant">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-[calc(100vh-8rem)]">
        {/* Input Panel */}
        <Card className="flex flex-col">
          <CardContent className="flex-1 flex flex-col gap-3 p-4">
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1 block">Tone</label>
              <div className="flex flex-wrap gap-1">
                {tones.map((t) => (
                  <Button
                    key={t.id}
                    variant={tone === t.id ? 'default' : 'outline'}
                    size="sm"
                    className="h-7 text-xs"
                    onClick={() => setTone(t.id)}
                  >
                    {t.label}
                  </Button>
                ))}
              </div>
            </div>

            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1 block">Type</label>
              <div className="flex flex-wrap gap-1">
                {types.map((t) => (
                  <Button
                    key={t.id}
                    variant={type === t.id ? 'default' : 'outline'}
                    size="sm"
                    className="h-7 text-xs"
                    onClick={() => setType(t.id)}
                  >
                    {t.label}
                  </Button>
                ))}
              </div>
            </div>

            <div className="flex-1 flex flex-col gap-2">
              <label className="text-xs font-medium text-muted-foreground">What do you want to write?</label>
              <Textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Tell me what you want to write about..."
                className="flex-1 resize-none text-sm"
              />
            </div>

            <div className="flex gap-2">
              <Button
                onClick={() => generate('write')}
                disabled={isGenerating || !input.trim()}
                className="flex-1"
              >
                {isGenerating ? 'Writing...' : 'Write'}
              </Button>
              <Button
                variant="outline"
                onClick={() => generate('rewrite')}
                disabled={isGenerating || !output}
              >
                Rewrite
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Output Panel */}
        <Card className="flex flex-col">
          <CardContent className="flex-1 flex flex-col gap-3 p-4">
            <label className="text-xs font-medium text-muted-foreground">Result</label>
            <Textarea
              ref={outputRef}
              value={output}
              readOnly
              placeholder="Your writing will appear here..."
              className="flex-1 resize-none text-sm bg-muted/30"
            />

            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={() => generate('shorter')}
                disabled={isGenerating || !output}
              >
                Shorter
              </Button>
              <Button
                variant="outline"
                onClick={() => generate('funnier')}
                disabled={isGenerating || !output}
              >
                Funnier
              </Button>
              <Button
                variant="outline"
                onClick={copyToClipboard}
                disabled={!output}
                className="ml-auto"
              >
                Copy
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
