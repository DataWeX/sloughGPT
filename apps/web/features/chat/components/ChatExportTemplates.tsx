'use client'

import { memo } from 'react'
import { Button, IconX, IconCheck, IconDownload } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import type { ChatMessage } from '@/lib/chat-utils'
import { useExportTemplates, DEFAULT_TEMPLATES, type ExportTemplate } from './useExportTemplates'

interface ChatExportTemplatesProps {
  messages: ChatMessage[]
  sessionTitle?: string
  className?: string
}

export const ChatExportTemplates = memo(function ChatExportTemplates({
  messages,
  sessionTitle,
  className,
}: ChatExportTemplatesProps) {
  const {
    templates, selectedId, selectedTemplate,
    showCustom, customName, customFormat,
    setSelectedId, setShowCustom, setCustomName, setCustomFormat,
    handleSaveCustom, handleDeleteTemplate, handleExport, handleCopy,
  } = useExportTemplates(messages, sessionTitle)

  return (
    <div className={cn('border rounded-lg bg-card overflow-hidden', className)}>
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/30">
        <div className="flex items-center gap-2">
          <IconDownload className="h-3 w-3 text-muted-foreground" />
          <span className="text-xs font-medium">Export Templates</span>
        </div>
        <Button
          variant="ghost"
          size="icon-sm"
          className="h-5 w-5"
          onClick={() => setShowCustom(!showCustom)}
          aria-label="Create template"
        >
          <IconDownload className="h-3 w-3" />
        </Button>
      </div>

      {showCustom && (
        <div className="p-2 border-b space-y-2">
          <input
            type="text"
            value={customName}
            onChange={(e) => setCustomName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSaveCustom()}
            placeholder="Template name..."
            className="w-full text-xs bg-transparent border rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary/50"
          />
          <select
            value={customFormat}
            onChange={(e) => setCustomFormat(e.target.value as ExportTemplate['format'])}
            className="w-full text-xs bg-transparent border rounded px-2 py-1"
          >
            <option value="markdown">Markdown</option>
            <option value="json">JSON</option>
            <option value="csv">CSV</option>
            <option value="html">HTML</option>
          </select>
          <div className="flex gap-1">
            <Button
              variant="ghost"
              size="sm"
              className="text-[10px] h-6"
              onClick={handleSaveCustom}
              disabled={!customName.trim()}
            >
              <IconCheck className="h-3 w-3 mr-1" />
              Save
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="text-[10px] h-6"
              onClick={() => setShowCustom(false)}
            >
              Cancel
            </Button>
          </div>
        </div>
      )}

      <div className="max-h-[300px] overflow-y-auto">
        {templates.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">No templates</p>
        ) : (
          <div className="divide-y">
            {templates.map(template => (
              <div
                key={template.id}
                className={cn(
                  'flex items-center gap-2 px-3 py-2 hover:bg-muted/30 group',
                  selectedId === template.id && 'bg-primary/5',
                )}
              >
                <button
                  type="button"
                  className="flex-1 text-left min-w-0"
                  onClick={() => setSelectedId(template.id)}
                >
                  <div className="text-xs font-medium">{template.name}</div>
                  <div className="text-[10px] text-muted-foreground uppercase">
                    {template.format} · {template.messageFilter}
                  </div>
                </button>
                {!DEFAULT_TEMPLATES.find(t => t.id === template.id) && (
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    className="h-5 w-5 opacity-0 group-hover:opacity-100"
                    onClick={() => handleDeleteTemplate(template.id)}
                    title="Delete template"
                  >
                    <IconX className="h-3 w-3" />
                  </Button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="p-2 border-t flex gap-1">
        <Button
          variant="ghost"
          size="sm"
          className="text-[10px] h-6 flex-1"
          onClick={handleExport}
          disabled={!selectedTemplate || messages.length === 0}
        >
          <IconDownload className="h-3 w-3 mr-1" />
          Download
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="text-[10px] h-6 flex-1"
          onClick={handleCopy}
          disabled={!selectedTemplate || messages.length === 0}
        >
          Copy
        </Button>
      </div>
    </div>
  )
})
