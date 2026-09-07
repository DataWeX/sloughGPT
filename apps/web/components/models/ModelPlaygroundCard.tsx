'use client'

import { useState, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Textarea } from '@sloughgpt/strui'
import { Slider } from '@sloughgpt/strui'
import { generateController } from '@/lib/generate-controller'
import { extractErrorMessage } from '@/lib/error-utils'

interface ModelPlaygroundCardProps {
  activeRuntimeId: string | null
}

export default function ModelPlaygroundCard({ activeRuntimeId }: ModelPlaygroundCardProps) {
  const [testPrompt, setTestPrompt] = useState('')
  const [testOutput, setTestOutput] = useState('')
  const [testGenerating, setTestGenerating] = useState(false)
  const [testStreaming, setTestStreaming] = useState(false)
  const [testTemp, setTestTemp] = useState(0.7)
  const [testMaxTokens, setTestMaxTokens] = useState(100)
  const streamingRef = useRef(false)

  const handleTestGenerate = async () => {
    if (!testPrompt.trim() || !activeRuntimeId) return
    setTestGenerating(true)
    setTestStreaming(true)
    setTestOutput('')
    streamingRef.current = true

    try {
      await generateController.generateStream(
        {
          prompt: testPrompt,
          max_new_tokens: testMaxTokens,
          temperature: testTemp,
        },
        (token) => {
          if (!streamingRef.current) return
          setTestOutput(prev => prev + token)
        },
        () => {},
        (error) => {
          if (!streamingRef.current) return
          setTestOutput(`Error: ${error}`)
        },
      )
    } catch (err) {
      setTestOutput(`Error: ${extractErrorMessage(err)}`)
    } finally {
      streamingRef.current = false
      setTestGenerating(false)
      setTestStreaming(false)
    }
  }

  if (!activeRuntimeId) return null

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Model Playground</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <Textarea
          value={testPrompt}
          onChange={e => setTestPrompt(e.target.value)}
          placeholder="Enter a prompt to test the loaded model..."
          rows={3}
          className="text-sm"
        />
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <Slider label="Temperature" value={[testTemp]} onValueChange={([v]) => setTestTemp(v)} min={0} max={2} step={0.1} />
          </div>
          <div className="flex-1">
            <Slider label="Max tokens" value={[testMaxTokens]} onValueChange={([v]) => setTestMaxTokens(v)} min={10} max={500} step={10} />
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" onClick={handleTestGenerate} disabled={testGenerating || !testPrompt.trim()}>
            {testGenerating ? 'Generating...' : 'Generate'}
          </Button>
          {testOutput && (
            <Button size="sm" variant="ghost" onClick={() => { setTestOutput(''); setTestPrompt(''); streamingRef.current = false }}>Clear</Button>
          )}
        </div>
        {testGenerating && <div className="text-xs text-muted-foreground animate-pulse">Generating...</div>}
        {testOutput && (
          <pre className="text-sm bg-muted/30 rounded-lg p-3 whitespace-pre-wrap break-words max-h-48 overflow-y-auto border border-border/40">
            {testOutput}
            {testStreaming && <span className="inline-block w-1.5 h-3 bg-primary/70 animate-pulse ml-0.5" />}
          </pre>
        )}
      </CardContent>
    </Card>
  )
}
