'use client'

import { useState } from 'react'
import { Card, CardContent, Button, Input } from '@sloughgpt/strui'

interface AuditTrailFilterCardProps {
  types?: string[]
  onFilterChange?: (filter: { text: string; type: string; dateFrom: string; dateTo: string }) => void
  onExport?: () => void
}

export function AuditTrailFilterCard({ types = [], onFilterChange, onExport }: AuditTrailFilterCardProps) {
  const [text, setText] = useState('')
  const [type, setType] = useState('all')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')

  const update = (patch: Partial<{ text: string; type: string; dateFrom: string; dateTo: string }>) => {
    const next = { text, type, dateFrom, dateTo, ...patch }
    if (patch.text !== undefined) setText(patch.text)
    if (patch.type !== undefined) setType(patch.type)
    if (patch.dateFrom !== undefined) setDateFrom(patch.dateFrom)
    if (patch.dateTo !== undefined) setDateTo(patch.dateTo)
    onFilterChange?.(next)
  }

  const clearAll = () => {
    setText('')
    setType('all')
    setDateFrom('')
    setDateTo('')
    onFilterChange?.({ text: '', type: 'all', dateFrom: '', dateTo: '' })
  }

  return (
    <Card data-testid="audit-trail-filter">
      <CardContent>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Input
              value={text}
              onChange={e => update({ text: e.target.value })}
              placeholder="Filter by action, type, user..."
              className="flex-1"
              data-testid="audit-search"
            />
            <select
              value={type}
              onChange={e => update({ type: e.target.value })}
              className="text-xs border border-border rounded px-2 py-1.5 bg-background"
              data-testid="audit-type-filter"
            >
              <option value="all">All types</option>
              {types.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-muted-foreground">From:</span>
              <Input type="date" value={dateFrom} onChange={e => update({ dateFrom: e.target.value })} className="w-32" data-testid="audit-date-from" />
            </div>
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-muted-foreground">To:</span>
              <Input type="date" value={dateTo} onChange={e => update({ dateTo: e.target.value })} className="w-32" data-testid="audit-date-to" />
            </div>
            {(dateFrom || dateTo || type !== 'all') && (
              <Button size="sm" variant="ghost" className="text-[10px]" onClick={clearAll}>Clear</Button>
            )}
            <div className="flex-1" />
            {onExport && (
              <Button size="sm" variant="outline" className="text-[10px]" onClick={onExport}>Export CSV</Button>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
