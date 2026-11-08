'use client'

// Right-hand results column of the assembly sandbox: run status, training
// job/result, training launch form, registers, output, VGA screen, memory
// dump. Extracted from page.tsx by the page-decomposition refactor — pure
// presentation over hook state.

import { type Dispatch, type SetStateAction } from 'react'
import { Button, Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import type { VMRegister, VMRunResult, VMTrainingJob } from '@/lib/vm-controller'
import type { TrainConfig } from '../vm-data'
import { StatusRow } from './VmStatusRow'
import { TrainingCard } from './VmTrainingCard'
import { VmTrainingLaunch } from './VmTrainingLaunch'
import { VGADisplay } from './VmVgaDisplay'

interface VmResultColumnProps {
  result: VMRunResult | null
  permissionDenied: boolean
  trainingJob: VMTrainingJob | null
  onStopTraining: () => void
  trainConfig: TrainConfig
  setTrainConfig: Dispatch<SetStateAction<TrainConfig>>
  customDataset: boolean
  setCustomDataset: Dispatch<SetStateAction<boolean>>
  datasetNames: string[]
  role: string
  setRole: Dispatch<SetStateAction<string>>
  launchedJob: number | null
  setLaunchedJob: Dispatch<SetStateAction<number | null>>
  setSource: Dispatch<SetStateAction<string>>
  onLaunchTraining: () => void
}

export function VmResultColumn({
  result,
  permissionDenied,
  trainingJob,
  onStopTraining,
  trainConfig,
  setTrainConfig,
  customDataset,
  setCustomDataset,
  datasetNames,
  role,
  setRole,
  launchedJob,
  setLaunchedJob,
  setSource,
  onLaunchTraining,
}: VmResultColumnProps) {
  return (
    <div className="space-y-4">
      {/* Status */}
      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Result</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <StatusRow
              label="Status"
              value={
                result.success ? (
                  <span className="text-success">{result.status}</span>
                ) : (
                  <span className="text-destructive">{result.status}</span>
                )
              }
            />
            <StatusRow
              label="Exit code"
              value={`0x${result.exit_code.toString(16).toUpperCase()}`}
            />
            <StatusRow label="Steps" value={result.steps_executed.toLocaleString()} />
            <StatusRow label="Time" value={`${result.elapsed_ms.toFixed(1)}ms`} />
            {result.error && (
              <StatusBanner variant="error" message={result.error} dismissible={false} />
            )}
            {permissionDenied && (
              <div className="text-xs text-warning bg-warning/10 p-2 rounded">
                A syscall was denied for the current role (EAX = -2). Training and other privileged
                operations require the <span className="font-medium">admin</span> role — switch the
                role selector above and run again.
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Training */}
      <TrainingCard job={trainingJob} onStop={onStopTraining} />

      {/* Training result */}
      {result?.training_result && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Training result</CardTitle>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => navigator.clipboard.writeText(result.training_result as string)}
              >
                Copy
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <pre className="text-xs font-mono bg-muted/30 p-2 rounded overflow-x-auto whitespace-pre-wrap max-h-48 overflow-y-auto text-success">
              {result.training_result}
            </pre>
          </CardContent>
        </Card>
      )}

      {/* Training launch */}
      <VmTrainingLaunch
        trainConfig={trainConfig}
        setTrainConfig={setTrainConfig}
        customDataset={customDataset}
        setCustomDataset={setCustomDataset}
        datasetNames={datasetNames}
        role={role}
        setRole={setRole}
        launchedJob={launchedJob}
        setLaunchedJob={setLaunchedJob}
        setSource={setSource}
        onLaunch={onLaunchTraining}
      />

      {/* Registers */}
      {result && result.registers.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Registers</CardTitle>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  const text = result.registers.map((r) => `${r.name} = ${r.hex}`).join('\n')
                  navigator.clipboard.writeText(text)
                }}
              >
                Copy
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-1">
              {result.registers.map((reg: VMRegister) => (
                <button
                  key={reg.name}
                  className="flex justify-between text-xs font-mono px-2 py-1 bg-muted/30 rounded hover:bg-muted/60 text-left transition-colors"
                  onClick={() => navigator.clipboard.writeText(reg.hex)}
                  title="Click to copy"
                >
                  <span className="text-muted-foreground">{reg.name}</span>
                  <span>{reg.hex}</span>
                </button>
              ))}
            </div>
            <div className="flex justify-between text-xs font-mono px-2 py-1 bg-muted/30 rounded mt-1">
              <span className="text-muted-foreground">EIP</span>
              <span>{result.eip_hex}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Output */}
      {result && result.output && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Output</CardTitle>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => navigator.clipboard.writeText(result.output)}
              >
                Copy
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <pre className="text-xs font-mono bg-muted/30 p-2 rounded overflow-x-auto whitespace-pre-wrap max-h-48 overflow-y-auto">
              {result.output}
            </pre>
          </CardContent>
        </Card>
      )}

      {/* VGA memory */}
      {result && result.success && <VGADisplay text={result.vga_text} cells={result.vga_cells} />}

      {/* Memory dump */}
      {result?.memory_dump && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Memory (stack area)</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="text-xs font-mono bg-muted/30 p-2 rounded overflow-x-auto whitespace-pre max-h-48 overflow-y-auto">
              {result.memory_dump}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
