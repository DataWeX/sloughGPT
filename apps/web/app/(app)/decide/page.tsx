'use client'

import { useState, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, Button, Textarea, Input } from '@sloughgpt/strui'
import { generateTool } from '@/lib/tools-controller'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'

const _log = logger.child('decide')

export default function DecidePage() {
  const [question, setQuestion] = useState('')
  const [optionA, setOptionA] = useState('')
  const [optionB, setOptionB] = useState('')
  const [notesA, setNotesA] = useState('')
  const [notesB, setNotesB] = useState('')
  const [result, setResult] = useState('')
  const [isDeciding, setIsDeciding] = useState(false)
  const addToast = useToastStore((s) => s.addToast)

  const decide = useCallback(async () => {
    if (!question.trim() || !optionA.trim() || !optionB.trim()) {
      addToast('Fill in the question and both options', 'info')
      return
    }

    setIsDeciding(true)
    setResult('')

    try {
      await generateTool(
        'decide',
        { question, option_a: optionA, option_b: optionB, notes_a: notesA, notes_b: notesB },
        {
          onToken: (token) => setResult(token),
          onError: (msg) => {
            _log.error('Decision failed', { error: msg })
            addToast('Failed to analyze — is a model loaded?', 'error')
          },
        },
        { max_tokens: 800 },
      )
    } catch (err) {
      _log.error('Decision failed', { error: err instanceof Error ? err.message : String(err) })
      addToast('Failed to analyze — is a model loaded?', 'error')
    } finally {
      setIsDeciding(false)
    }
  }, [question, optionA, optionB, notesA, notesB, addToast])

  const copyToClipboard = useCallback(() => {
    if (result) {
      navigator.clipboard.writeText(result)
      addToast('Copied to clipboard', 'success')
    }
  }, [result, addToast])

  return (
    <PageContainer title="Help Me Decide">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Input Panel */}
        <Card>
          <CardContent className="p-4 space-y-4">
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1 block">
                What are you deciding between?
              </label>
              <Input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="e.g. Should I take the job in New York or stay?"
                className="text-sm"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-1 block">Option A</label>
                <Input
                  value={optionA}
                  onChange={(e) => setOptionA(e.target.value)}
                  placeholder="First option"
                  className="text-sm"
                />
                <Textarea
                  value={notesA}
                  onChange={(e) => setNotesA(e.target.value)}
                  placeholder="Notes (optional)"
                  className="text-xs mt-1 resize-none"
                  rows={2}
                />
              </div>
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-1 block">Option B</label>
                <Input
                  value={optionB}
                  onChange={(e) => setOptionB(e.target.value)}
                  placeholder="Second option"
                  className="text-sm"
                />
                <Textarea
                  value={notesB}
                  onChange={(e) => setNotesB(e.target.value)}
                  placeholder="Notes (optional)"
                  className="text-xs mt-1 resize-none"
                  rows={2}
                />
              </div>
            </div>

            <Button
              onClick={decide}
              disabled={isDeciding || !question.trim() || !optionA.trim() || !optionB.trim()}
              className="w-full"
            >
              {isDeciding ? 'Analyzing...' : 'Help Me Decide'}
            </Button>
          </CardContent>
        </Card>

        {/* Result Panel */}
        <Card>
          <CardContent className="p-4">
            <label className="text-xs font-medium text-muted-foreground mb-2 block">Analysis</label>
            <div className="prose prose-sm max-w-none text-sm whitespace-pre-wrap min-h-[200px]">
              {result || (
                <span className="text-muted-foreground">
                  Fill in the options and click &quot;Help Me Decide&quot; to get a comparison.
                </span>
              )}
            </div>
            {result && (
              <Button variant="outline" onClick={copyToClipboard} className="mt-3">
                Copy
              </Button>
            )}
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
