'use client'

import { Card, CardContent, Button, Input } from '@sloughgpt/strui'

interface MemoryQuickRememberProps {
  content: string
  topic: string
  onContentChange: (c: string) => void
  onTopicChange: (t: string) => void
  onRemember: () => void
  remembering?: boolean
}

export function MemoryQuickRemember({ content, topic, onContentChange, onTopicChange, onRemember, remembering }: MemoryQuickRememberProps) {
  return (
    <Card>
      <CardContent className="p-2.5">
        <div className="flex items-center gap-1.5">
          <Input
            value={content}
            onChange={e => onContentChange(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') onRemember() }}
            placeholder="Quick remember: type a fact and press Enter"
            className="h-7 text-[11px] flex-1"
          />
          <Input
            value={topic}
            onChange={e => onTopicChange(e.target.value)}
            placeholder="Topic (optional)"
            className="h-7 text-[11px] w-24"
          />
          <Button size="sm" className="h-7 text-[11px]" onClick={onRemember} disabled={remembering || !content.trim()}>
            {remembering ? 'Saving...' : 'Remember'}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
