'use client'

import { useState, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, Button, Textarea } from '@sloughgpt/strui'
import { generateTool } from '@/lib/tools-controller'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'

const _log = logger.child('rewrite')

type Action = 'grammar' | 'shorter' | 'friendlier' | 'professional' | 'sound-like-me'

const ACTIONS: { id: Action; label: string }[] = [
  { id: 'grammar', label: 'Fix Grammar' },
  { id: 'shorter', label: 'Make Shorter' },
  { id: 'friendlier', label: 'Make Friendlier' },
  { id: 'professional', label: 'Make Professional' },
  { id: 'sound-like-me', label: 'Sound Like Me' },
]

export default function RewritePage() {
  const [original, setOriginal] = useState('')
  const [rewritten, setRewritten] = useState('')
  const [activeAction, setActiveAction] = useState<Action | null>(null)
  const addToast = useToastStore((s) => s.addToast)

  const rewrite = useCallback(async (action: Action) => {
    if (!original.trim()) {
      addToast('Paste something to rewrite', 'info')
      return
    }

    setActiveAction(action)
    setRewritten('')

    try {
      await generateTool(
        'rewrite',
        { text: original, action },
        {
          onToken: (token) => setRewritten(token),
          onError: (msg) => {
            _log.error('Rewrite failed', { error: msg })
            addToast('Rewrite failed — is a model loaded?', 'error')
          },
        },
        { max_tokens: 500 },
      )
    } catch (err) {
      _log.error('Rewrite failed', { error: err instanceof Error ? err.message : String(err) })
      addToast('Rewrite failed — is a model loaded?', 'error')
    } finally {
      setActiveAction(null)
    }
  }, [original, addToast])

  const useThisVersion = useCallback(() => {
    setOriginal(rewritten)
    setRewritten('')
    addToast('Applied to original', 'success')
  }, [rewritten, addToast])

  const copyToClipboard = useCallback(() => {
    if (rewritten) {
      navigator.clipboard.writeText(rewritten)
      addToast('Copied to clipboard', 'success')
    }
  }, [rewritten, addToast])

  return (
    <PageContainer title="Rewrite & Polish">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-[calc(100vh-8rem)]">
        {/* Original */}
        <Card className="flex flex-col">
          <CardContent className="flex-1 flex flex-col gap-3 p-4">
            <label className="text-xs font-medium text-muted-foreground">Original</label>
            <Textarea
              value={original}
              onChange={(e) => setOriginal(e.target.value)}
              placeholder="Paste what you wrote..."
              className="flex-1 resize-none text-sm"
            />
          </CardContent>
        </Card>

        {/* Rewritten */}
        <Card className="flex flex-col">
          <CardContent className="flex-1 flex flex-col gap-3 p-4">
            <label className="text-xs font-medium text-muted-foreground">Rewritten</label>
            <Textarea
              value={rewritten}
              readOnly
              placeholder="Rewritten version will appear here..."
              className="flex-1 resize-none text-sm bg-muted/30"
            />
            {rewritten && (
              <div className="flex gap-2">
                <Button variant="outline" onClick={useThisVersion} className="flex-1">
                  Use This Version
                </Button>
                <Button variant="outline" onClick={copyToClipboard}>
                  Copy
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-wrap justify-center gap-2 mt-4">
        {ACTIONS.map((action) => (
          <Button
            key={action.id}
            variant="outline"
            onClick={() => rewrite(action.id)}
            disabled={activeAction !== null || !original.trim()}
          >
            {activeAction === action.id ? 'Working...' : action.label}
          </Button>
        ))}
      </div>
    </PageContainer>
  )
}
