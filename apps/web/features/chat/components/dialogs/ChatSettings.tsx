'use client'

import { cn, Button } from '@sloughgpt/strui'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@sloughgpt/strui'
import { IconChevronDown, IconCheck } from '@sloughgpt/strui'
import { shortModelName } from '@/lib/chat-utils'

interface ChatSettingsProps {
  isOpen: boolean
  model: string
  temperature: number
  maxTokens: number
  autoApproveTools: boolean
  onModelChange: (value: string) => void
  onTemperatureChange: (value: number) => void
  onMaxTokensChange: (value: number) => void
  onAutoApproveToolsChange: (value: boolean) => void
  onClear: () => void
  hasMessages: boolean
  availableModels?: string[]
}

const TEMP_OPTIONS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
const MAX_TOKEN_OPTIONS = [50, 100, 150, 200, 300, 400, 500, 750, 1000]
const DEFAULT_MODELS = ['gpt2', 'gpt2-medium', 'gpt2-large', 'distilgpt2']

export function ChatSettings({
  isOpen,
  model,
  temperature,
  maxTokens,
  autoApproveTools,
  onModelChange,
  onTemperatureChange,
  onMaxTokensChange,
  onAutoApproveToolsChange,
  onClear,
  hasMessages,
  availableModels = DEFAULT_MODELS,
}: ChatSettingsProps) {
  return (
    <section
      aria-label="Chat settings"
      className={cn(
        "shrink-0 border-b border-border/40 bg-muted/20 px-3 py-2 transition-all duration-200 sm:px-4",
        isOpen ? "max-h-20 opacity-100" : "max-h-0 border-transparent p-0 opacity-0 overflow-hidden"
      )}
    >
      <div className="mx-auto flex max-w-2xl flex-wrap items-center gap-x-3 gap-y-2 text-sm">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="select" size="sm" className="text-xs font-mono" title={model} aria-label={`Model: ${shortModelName(model)}`}>
              <span className="truncate">{shortModelName(model)}</span>
              <IconChevronDown className="h-3 w-3 opacity-60 shrink-0" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="min-w-[120px]">
            {availableModels.map((m: string) => (
              <DropdownMenuItem
                key={m}
                onSelect={() => onModelChange(m)}
                className="justify-between font-mono text-xs"
                title={m}
              >
                <span className="truncate">{shortModelName(m)}</span>
                {m === model && <IconCheck className="h-3 w-3 text-primary shrink-0" />}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="select" size="sm" className="text-xs font-mono w-16" aria-label={`Temperature: ${temperature}`}>
              <span className="truncate">{temperature}</span>
              <IconChevronDown className="h-3 w-3 opacity-60 shrink-0" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="min-w-[80px]">
            {TEMP_OPTIONS.map((t) => (
              <DropdownMenuItem
                key={t}
                onSelect={() => onTemperatureChange(t)}
                className="justify-between font-mono text-xs"
              >
                <span className="truncate">{t}</span>
                {t === temperature && <IconCheck className="h-3 w-3 text-primary shrink-0" />}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="select" size="sm" className="text-xs font-mono w-16" aria-label={`Max tokens: ${maxTokens}`}>
              <span className="truncate">{maxTokens}</span>
              <IconChevronDown className="h-3 w-3 opacity-60 shrink-0" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="min-w-[80px]">
            {MAX_TOKEN_OPTIONS.map((t) => (
              <DropdownMenuItem
                key={t}
                onSelect={() => onMaxTokensChange(t)}
                className="justify-between font-mono text-xs"
              >
                <span className="truncate">{t}</span>
                {t === maxTokens && <IconCheck className="h-3 w-3 text-primary shrink-0" />}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>

        <Button
          variant={autoApproveTools ? "default" : "outline"}
          size="sm"
          onClick={() => onAutoApproveToolsChange(!autoApproveTools)}
          className="text-xs hover:opacity-80 active:opacity-70"
          aria-label={`Auto-approve tools: ${autoApproveTools ? 'on' : 'off'}`}
        >
          {autoApproveTools ? 'Auto-approve: ON' : 'Auto-approve: OFF'}
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={onClear}
          className="text-xs hover:opacity-80 active:opacity-70 disabled:opacity-50"
          disabled={!hasMessages}
        >
          Clear
        </Button>
      </div>
    </section>
  )
}
