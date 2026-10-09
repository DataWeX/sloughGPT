'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, Input, cn } from '@sloughgpt/strui'

interface ToolUsed {
  tool: string
  result: unknown
}

interface AgentExecutionFormProps {
  agentName: string
  prompt: string
  result: string | null
  toolsUsed: ToolUsed[]
  running: boolean
  errors: { prompt?: string }
  onPromptChange: (value: string) => void
  onExecute: () => void
  onClose: () => void
}

export function AgentExecutionForm({
  agentName,
  prompt,
  result,
  toolsUsed,
  running,
  errors,
  onPromptChange,
  onExecute,
  onClose,
}: AgentExecutionFormProps) {
  return (
    <div className="space-y-2" data-testid="agent-execution-form">
      <div>
        <Input
          id="exec-prompt"
          placeholder={`What should ${agentName} do?`}
          value={prompt}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => onPromptChange(e.target.value)}
          onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              onExecute()
            }
          }}
          className={errors.prompt ? 'border-destructive ring-destructive/20' : ''}
          aria-invalid={!!errors.prompt}
          aria-describedby={errors.prompt ? 'exec-prompt-error' : undefined}
        />
        {errors.prompt && (
          <p id="exec-prompt-error" className="text-[10px] text-destructive mt-1" role="alert">
            {errors.prompt}
          </p>
        )}
      </div>
      <div className="flex gap-2">
        <Button size="sm" onClick={onExecute} disabled={running || !prompt.trim()}>
          {running ? 'Running...' : 'Execute'}
        </Button>
        <Button size="sm" variant="ghost" onClick={onClose} className="h-6 text-[10px]">
          Close
        </Button>
      </div>
      {result && (
        <div className="rounded-lg bg-muted p-3">
          {toolsUsed.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-1">
              {toolsUsed.map((t, i) => (
                <span key={i} className="text-[9px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">
                  {t.tool.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          )}
          <p className="text-[10px] font-medium text-muted-foreground mb-1">Response</p>
          <p className="text-[11px] whitespace-pre-wrap">{result}</p>
        </div>
      )}
    </div>
  )
}
