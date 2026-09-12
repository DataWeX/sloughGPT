'use client'

import { useState, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, Button, Textarea, Select } from '@sloughgpt/strui'
import { generateTool } from '@/lib/tools-controller'
import { useToastStore } from '@/lib/toast-store'
import { logger } from '@/lib/dev-log'

const _log = logger.child('translate')

const LANGUAGES = [
  'Spanish', 'French', 'German', 'Italian', 'Portuguese',
  'Chinese', 'Japanese', 'Korean', 'Arabic', 'Hindi',
  'Russian', 'Dutch', 'Swedish', 'Polish', 'Turkish',
]

export default function TranslatePage() {
  const [source, setSource] = useState('')
  const [target, setTarget] = useState('')
  const [targetLang, setTargetLang] = useState('Spanish')
  const [isTranslating, setIsTranslating] = useState(false)
  const addToast = useToastStore((s) => s.addToast)

  const translate = useCallback(async () => {
    if (!source.trim()) {
      addToast('Type something to translate', 'info')
      return
    }

    setIsTranslating(true)
    setTarget('')

    try {
      await generateTool(
        'translate',
        { text: source, target_lang: targetLang },
        {
          onToken: (token) => setTarget(token),
          onError: (msg) => {
            _log.error('Translation failed', { error: msg })
            addToast('Translation failed — is a model loaded?', 'error')
          },
        },
        { max_tokens: 500 },
      )
    } catch (err) {
      _log.error('Translation failed', { error: err instanceof Error ? err.message : String(err) })
      addToast('Translation failed — is a model loaded?', 'error')
    } finally {
      setIsTranslating(false)
    }
  }, [source, targetLang, addToast])

  const swap = useCallback(() => {
    setSource(target)
    setTarget(source)
  }, [source, target])

  const copyToClipboard = useCallback(() => {
    if (target) {
      navigator.clipboard.writeText(target)
      addToast('Copied to clipboard', 'success')
    }
  }, [target, addToast])

  return (
    <PageContainer title="Translate">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-[calc(100vh-8rem)]">
        {/* Source */}
        <Card className="flex flex-col">
          <CardContent className="flex-1 flex flex-col gap-3 p-4">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-muted-foreground">Source</label>
              <span className="text-[10px] text-muted-foreground">Auto-detected</span>
            </div>
            <Textarea
              value={source}
              onChange={(e) => setSource(e.target.value)}
              placeholder="Type or paste text to translate..."
              className="flex-1 resize-none text-sm"
            />
          </CardContent>
        </Card>

        {/* Target */}
        <Card className="flex flex-col">
          <CardContent className="flex-1 flex flex-col gap-3 p-4">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-muted-foreground">Translation</label>
              <Select
                value={targetLang}
                onChange={(e) => setTargetLang(e.target.value)}
                className="w-40 h-7 text-xs"
              >
                {LANGUAGES.map((lang) => (
                  <option key={lang} value={lang}>{lang}</option>
                ))}
              </Select>
            </div>
            <Textarea
              value={target}
              readOnly
              placeholder="Translation will appear here..."
              className="flex-1 resize-none text-sm bg-muted/30"
            />
          </CardContent>
        </Card>
      </div>

      {/* Actions */}
      <div className="flex justify-center gap-2 mt-4">
        <Button
          variant="outline"
          onClick={swap}
          disabled={isTranslating}
        >
          Swap
        </Button>
        <Button
          onClick={translate}
          disabled={isTranslating || !source.trim()}
        >
          {isTranslating ? 'Translating...' : 'Translate'}
        </Button>
        <Button
          variant="outline"
          onClick={copyToClipboard}
          disabled={!target}
        >
          Copy
        </Button>
      </div>
    </PageContainer>
  )
}
