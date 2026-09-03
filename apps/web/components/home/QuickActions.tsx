'use client'

import { Card, CardContent, Button } from '@sloughgpt/strui'
import { chatController } from '@/lib/chat-controller'
import { extractErrorMessage } from '@/lib/error-utils'
import { knowledgeController } from '@/lib/knowledge-controller'
import { useToastStore } from '@/lib/toast-store'

interface QuickActionsProps {
  loading: boolean
  modelStatus: { loaded: boolean; model: string | null }
  testRunning: boolean
  testResponse: string | null
  setTestRunning: (v: boolean) => void
  setTestResponse: (v: string | null) => void
  knowledgeCount: number
  setKnowledgeCount: React.Dispatch<React.SetStateAction<number>>
}

export function QuickActions({ loading, modelStatus, testRunning, testResponse, setTestRunning, setTestResponse, knowledgeCount, setKnowledgeCount }: QuickActionsProps) {
  const addToast = useToastStore(s => s.addToast)

  if (loading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <Card>
          <CardContent className="py-3">
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0 flex-1 space-y-1">
                <div className="h-4 w-20 animate-pulse rounded bg-muted" />
                <div className="h-3 w-40 animate-pulse rounded bg-muted" />
              </div>
              <div className="h-8 w-20 animate-pulse rounded bg-muted shrink-0" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="py-3">
            <div className="h-4 w-24 animate-pulse rounded bg-muted mb-2" />
            <div className="flex gap-2">
              <div className="h-9 flex-1 animate-pulse rounded bg-muted" />
              <div className="h-8 w-14 animate-pulse rounded bg-muted shrink-0" />
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (!modelStatus.loaded) return null

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      <Card>
        <CardContent className="py-3">
          <div className="flex items-center justify-between gap-3">
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium">Quick test</p>
              <p className="text-xs text-muted-foreground">Send &quot;Hello!&quot; to verify the model works</p>
            </div>
            <Button
              size="sm"
              variant="outline"
              className="h-8 text-xs shrink-0"
              disabled={testRunning}
              onClick={async () => {
                setTestRunning(true)
                setTestResponse(null)
                try {
                  const result = await chatController.send('Hello!', { waitForModel: true })
                  setTestResponse(result.message || 'No response')
                } catch (e: unknown) {
                  setTestResponse(extractErrorMessage(e, 'Could not connect'))
                } finally {
                  setTestRunning(false)
                }
              }}
            >
              {testRunning ? 'Testing...' : 'Test model'}
            </Button>
          </div>
          {testResponse && (
            <div className="mt-2 rounded bg-muted/50 p-2 text-xs text-muted-foreground font-mono leading-relaxed">
              {testResponse}
            </div>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardContent className="py-3">
          <div className="flex items-center gap-2 mb-2">
            <p className="text-sm font-medium">Quick note</p>
            <p className="text-xs text-muted-foreground">Add a fact the AI can remember</p>
          </div>
          <form onSubmit={async (e) => {
            e.preventDefault()
            const input = e.currentTarget.querySelector('input') as HTMLInputElement
            const text = input.value.trim()
            if (!text) return
            try {
              await knowledgeController.add(text, 'general')
              input.value = ''
              setKnowledgeCount(k => k + 1)
              addToast('Fact saved', 'success')
            } catch { addToast('Could not save', 'error') }
          }} className="flex gap-2">
            <input
              type="text"
              placeholder="e.g., I prefer Python over JavaScript"
              aria-label="Quick add knowledge"
              className="flex-1 h-9 rounded-md border border-border/60 bg-background px-2.5 text-sm placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary/30"
            />
            <Button size="sm" type="submit" className="h-8 text-xs shrink-0">Save</Button>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
