'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Slider } from '@/components/ui/slider'
import { generateController } from '@/lib/generate-controller'

interface ModelPlaygroundCardProps {
  activeRuntimeId: string | null
}

export default function ModelPlaygroundCard({ activeRuntimeId }: ModelPlaygroundCardProps) {
  const [testPrompt, setTestPrompt] = useState('')
  const [testOutput, setTestOutput] = useState('')
  const [testGenerating, setTestGenerating] = useState(false)
  const [testTemp, setTestTemp] = useState(0.7)
  const [testMaxTokens, setTestMaxTokens] = useState(100)

  const handleTestGenerate = async () => {
    if (!testPrompt.trim() || !activeRuntimeId) return
    setTestGenerating(true)
    setTestOutput('')
    try {
      const result = await generateController.generate({
        prompt: testPrompt,
        max_new_tokens: testMaxTokens,
        temperature: testTemp,
      })
      setTestOutput(result.text || 'No output')
    } catch (err) {
      setTestOutput(`Error: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setTestGenerating(false)
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
            <Button size="sm" variant="ghost" onClick={() => { setTestOutput(''); setTestPrompt('') }}>Clear</Button>
          )}
        </div>
        {testGenerating && <div className="text-xs text-muted-foreground animate-pulse">Generating...</div>}
        {testOutput && !testGenerating && (
          <pre className="text-sm bg-muted/30 rounded-lg p-3 whitespace-pre-wrap break-words max-h-48 overflow-y-auto border border-border/40">
            {testOutput}
          </pre>
        )}
      </CardContent>
    </Card>
  )
}
