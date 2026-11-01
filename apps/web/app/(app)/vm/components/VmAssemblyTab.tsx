'use client'

// Assembly Sandbox tab: program-selector top bar, status banners, and the
// editor/results grid plus trace and reference panels. Extracted from
// page.tsx by the page-decomposition refactor; state arrives via the useVmPage
// bundle.

import { Button, Card, CardContent, Spinner } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { DEFAULT_PROGRAMS, MAX_STEPS_LIMIT, clampSteps } from '../vm-data'
import type { VmPageState } from '../useVmPage'
import { VmEditor } from './VmEditor'
import { VmResultColumn } from './VmResultColumn'
import { VmTrace } from './VmTrace'
import { VmReference } from './VmReference'

export function VmAssemblyTab({ vm }: { vm: VmPageState }) {
  const { result } = vm
  return (
    <>
      {/* Top bar: program selector + run */}
      <Card>
        <CardContent className="p-3">
          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex gap-1">
              {Object.entries(DEFAULT_PROGRAMS).map(([name, programSource]) => (
                <Button
                  key={name}
                  size="sm"
                  variant={vm.source === programSource ? 'default' : 'ghost'}
                  onClick={() => vm.setSource(programSource)}
                >
                  {name}
                </Button>
              ))}
            </div>
            <div className="flex items-center gap-2 ml-auto">
              <label className="text-xs text-muted-foreground" htmlFor="vm-steps">
                Steps:
              </label>
              <input
                id="vm-steps"
                type="number"
                value={vm.maxSteps}
                onChange={(e) => vm.setMaxSteps(Number(e.target.value))}
                className="w-20 px-2 py-1 text-xs border rounded bg-background"
                min={1}
                max={MAX_STEPS_LIMIT}
              />
              <input
                type="text"
                value={vm.keyboardInput}
                onChange={(e) => vm.setKeyboardInput(e.target.value)}
                placeholder="Keyboard input..."
                aria-label="Keyboard input"
                className="w-32 px-2 py-1 text-xs border rounded bg-background"
              />
              <select
                value={vm.role}
                onChange={(e) => vm.setRole(e.target.value)}
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
                  checked={vm.debug}
                  onChange={(e) => vm.setDebug(e.target.checked)}
                  className="rounded"
                />
                Debug
              </label>
              <Button
                size="sm"
                onClick={() => vm.handleRun()}
                disabled={vm.running}
                className="min-w-[80px]"
              >
                {vm.running ? (
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
                onClick={() => vm.handleRun(true)}
                disabled={vm.running}
              >
                Step
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => vm.setResult(null)}
                disabled={!result}
              >
                Clear
              </Button>
              <Button
                size="sm"
                variant={vm.showRef ? 'default' : 'ghost'}
                onClick={() => vm.setShowRef(!vm.showRef)}
              >
                Ref
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Status banner */}
      {result && !result.success && result.error && (
        <StatusBanner variant="error" message={result.error} dismissible={false} />
      )}
      {result && result.success && result.steps_executed >= clampSteps(vm.maxSteps) && (
        <div className="bg-warning/10 border border-warning/30 text-warning text-xs p-2 rounded">
          Step limit reached ({result.steps_executed} steps). Increase steps or use HLT to stop
          earlier.
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Editor */}
        <div className="lg:col-span-2">
          <VmEditor
            source={vm.source}
            setSource={vm.setSource}
            textareaRef={vm.textareaRef}
            handleKeyDown={vm.handleKeyDown}
            copied={vm.copied}
            setCopied={vm.setCopied}
            copiedTimerRef={vm.copiedTimerRef}
          />
        </div>

        {/* Results */}
        <VmResultColumn
          result={result}
          permissionDenied={vm.permissionDenied}
          trainingJob={vm.trainingJob}
          onStopTraining={vm.handleStopTraining}
          trainConfig={vm.trainConfig}
          setTrainConfig={vm.setTrainConfig}
          customDataset={vm.customDataset}
          setCustomDataset={vm.setCustomDataset}
          datasetNames={vm.datasetNames}
          role={vm.role}
          setRole={vm.setRole}
          launchedJob={vm.launchedJob}
          setLaunchedJob={vm.setLaunchedJob}
          setSource={vm.setSource}
          onLaunchTraining={vm.handleLaunchTraining}
        />
      </div>

      {/* Trace */}
      <VmTrace trace={result?.trace} />

      {/* Assembly reference */}
      <VmReference open={vm.showRef} />
    </>
  )
}
