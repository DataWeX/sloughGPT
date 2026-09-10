'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'
import { chatDB } from '@/lib/db'

interface ExportTemplate {
  id: string
  name: string
  format: string
  includeTokenizer: boolean
  outputPath: string
  timestamp: number
}

const STORAGE_KEY = 'sloughgpt-export-templates'

async function loadTemplates(): Promise<ExportTemplate[]> {
  try {
    const entry = await chatDB.getKV<ExportTemplate[]>(STORAGE_KEY)
    return entry ?? []
  } catch { return [] }
}

async function saveTemplates(templates: ExportTemplate[]) {
  try { await chatDB.setKV(STORAGE_KEY, templates) } catch { /* quota exceeded */ }
}

interface ExportTemplateCardProps {
  onSelect?: (template: ExportTemplate) => void
}

export function ExportTemplateCard({ onSelect }: ExportTemplateCardProps) {
  const [templates, setTemplates] = useState<ExportTemplate[]>([])
  const [editing, setEditing] = useState(false)
  const [editName, setEditName] = useState('')
  const [editFormat, setEditFormat] = useState('sou')
  const [editTokenizer, setEditTokenizer] = useState(true)
  const [editPath, setEditPath] = useState('models/exported')

  useEffect(() => { loadTemplates().then(setTemplates) }, [])

  const handleSave = useCallback(() => {
    if (!editName.trim()) return
    const template: ExportTemplate = {
      id: `tpl-${Date.now()}`,
      name: editName.trim(),
      format: editFormat,
      includeTokenizer: editTokenizer,
      outputPath: editPath,
      timestamp: Date.now(),
    }
    const updated = [...templates, template]
    setTemplates(updated)
    saveTemplates(updated).catch(() => {})
    setEditing(false)
    setEditName('')
  }, [editName, editFormat, editTokenizer, editPath, templates])

  const handleDelete = useCallback((id: string) => {
    const updated = templates.filter(t => t.id !== id)
    setTemplates(updated)
    saveTemplates(updated).catch(() => {})
  }, [templates])

  return (
    <Card data-testid="export-template">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Export Templates</CardTitle>
          <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => setEditing(!editing)}>
            {editing ? 'Cancel' : '+ New'}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {editing && (
          <div className="space-y-2 rounded-md border border-border/40 p-2.5">
            <Input
              value={editName}
              onChange={e => setEditName(e.target.value)}
              placeholder="Template name"
              className="h-7 text-[11px]"
            />
            <div className="flex gap-2">
              <select
                className="text-[10px] border border-border/40 rounded px-1.5 py-1 bg-background flex-1"
                value={editFormat}
                onChange={e => setEditFormat(e.target.value)}
              >
                {['sou', 'onnx', 'gguf', 'pytorch'].map(f => (
                  <option key={f} value={f}>{f.toUpperCase()}</option>
                ))}
              </select>
              <label className="flex items-center gap-1 text-[10px] text-muted-foreground">
                <input
                  type="checkbox"
                  checked={editTokenizer}
                  onChange={e => setEditTokenizer(e.target.checked)}
                  className="rounded"
                />
                Tokenizer
              </label>
            </div>
            <Input
              value={editPath}
              onChange={e => setEditPath(e.target.value)}
              placeholder="Output path"
              className="h-7 text-[10px]"
            />
            <Button size="sm" className="h-6 text-[10px]" onClick={handleSave} disabled={!editName.trim()}>
              Save Template
            </Button>
          </div>
        )}

        {templates.length === 0 && !editing ? (
          <p className="text-xs text-muted-foreground text-center py-2">No templates saved yet.</p>
        ) : (
          <div className="space-y-1.5">
            {templates.map(t => (
              <div key={t.id} className="flex items-center justify-between rounded-md border border-border/40 px-2.5 py-2 text-[11px] hover:bg-muted/20 transition-colors group">
                <div className="min-w-0 flex-1">
                  <div className="font-medium truncate">{t.name}</div>
                  <div className="text-[9px] text-muted-foreground">
                    {t.format.toUpperCase()} · {t.includeTokenizer ? 'with' : 'no'} tokenizer · {t.outputPath}
                  </div>
                </div>
                <div className="flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                  {onSelect && (
                    <Button size="sm" variant="ghost" className="h-5 text-[9px]" onClick={() => onSelect(t)}>
                      Use
                    </Button>
                  )}
                  <Button size="sm" variant="ghost" className="h-5 text-[9px] text-destructive" onClick={() => handleDelete(t.id)}>
                    Del
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
