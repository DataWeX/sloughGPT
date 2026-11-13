'use client'

import { Card, CardContent, Button, Spinner } from '@sloughgpt/strui'

interface VMControlBarProps {
  programs: Record<string, string>
  selectedSource: string
  onProgramSelect: (source: string) => void
  maxSteps: number
  onMaxStepsChange: (steps: number) => void
  keyboardInput: string
  onKeyboardInputChange: (input: string) => void
  role: string
  onRoleChange: (role: string) => void
  debug: boolean
  onDebugChange: (debug: boolean) => void
  running: boolean
  onRun: () => void
  onStep: () => void
  onClear: () => void
  hasResult: boolean
  showRef: boolean
  onToggleRef: () => void
}

const MAX_STEPS_LIMIT = 1_000_000

export function VMControlBar({
  programs,
  selectedSource,
  onProgramSelect,
  maxSteps,
  onMaxStepsChange,
  keyboardInput,
  onKeyboardInputChange,
  role,
  onRoleChange,
  debug,
  onDebugChange,
  running,
  onRun,
  onStep,
  onClear,
  hasResult,
  showRef,
  onToggleRef,
}: VMControlBarProps) {
  return (
    <Card>
      <CardContent className="p-3">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex gap-1">
            {Object.entries(programs).map(([name, programSource]) => (
              <Button
                key={name}
                size="sm"
                variant={selectedSource === programSource ? 'default' : 'ghost'}
                onClick={() => onProgramSelect(programSource)}
              >
                {name}
              </Button>
            ))}
          </div>
          <div className="flex items-center gap-2 ml-auto">
            <label className="text-xs text-muted-foreground" htmlFor="vm-steps">Steps:</label>
            <input
              id="vm-steps"
              type="number"
              value={maxSteps}
              onChange={(e) => onMaxStepsChange(Number(e.target.value))}
              className="w-20 px-2 py-1 text-xs border rounded bg-background"
              min={1}
              max={MAX_STEPS_LIMIT}
            />
            <input
              type="text"
              value={keyboardInput}
              onChange={(e) => onKeyboardInputChange(e.target.value)}
              placeholder="Keyboard input..."
              aria-label="Keyboard input"
              className="w-32 px-2 py-1 text-xs border rounded bg-background"
            />
            <select
              value={role}
              onChange={(e) => onRoleChange(e.target.value)}
              aria-label="VM role"
              className="px-2 py-1 text-xs border rounded bg-background"
            >
              <option value="user">user</option>
              <option value="admin">admin</option>
              <option value="kernel">kernel</option>
            </select>
            <label className="flex items-center gap-1 text-xs text-muted-foreground cursor-pointer">
              <input
                type="checkbox"
                checked={debug}
                onChange={(e) => onDebugChange(e.target.checked)}
                className="rounded"
              />
              Debug
            </label>
            <Button
              size="sm"
              onClick={onRun}
              disabled={running}
              className="min-w-[80px]"
            >
              {running ? (
                <span className="inline-flex items-center gap-1">
                  <Spinner size="xs" />
                  Running
                </span>
              ) : (
                'Run'
              )}
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={onStep}
              disabled={running}
            >
              Step
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={onClear}
              disabled={!hasResult}
            >
              Clear
            </Button>
            <Button
              size="sm"
              variant={showRef ? 'default' : 'ghost'}
              onClick={onToggleRef}
            >
              Ref
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
