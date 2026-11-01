'use client'

/**
 * Assembly sandbox tab: program bar, editor, results column, trace and the
 * x86 reference. Pure render — every value and handler arrives through the
 * `vm` workbench object, which `useVmWorkbench` owns.
 */
import { Button, Card, CardContent, CardHeader, CardTitle, Spinner } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import type { VMRegister } from '@/lib/vm-controller'
import { COPY_FEEDBACK_DURATION_MS } from '@/lib/constants'
import { DEFAULT_PROGRAMS } from './vm-programs'
import {
  DEFAULT_TRAIN_CONFIG,
  MAX_STEPS_LIMIT,
  buildTrainSource,
  clampSteps,
  clampTrainConfig,
  num,
  trainConfigHints,
} from './vm-train-config'
import { StatusRow, TrainingCard, VGADisplay } from './vm-panels'
import type { VmWorkbench } from './useVmWorkbench'

export function AssemblyTab({ vm }: { vm: VmWorkbench }) {
  const {
    source,
    setSource,
    result,
    setResult,
    running,
    maxSteps,
    setMaxSteps,
    debug,
    setDebug,
    keyboardInput,
    setKeyboardInput,
    role,
    setRole,
    showRef,
    setShowRef,
    copied,
    setCopied,
    trainingJob,
    trainConfig,
    setTrainConfig,
    datasetNames,
    customDataset,
    setCustomDataset,
    launchedJob,
    setLaunchedJob,
    textareaRef,
    copiedTimerRef,
    handleRun,
    handleLaunchTraining,
    handleKeyDown,
    handleStopTraining,
    permissionDenied,
  } = vm

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
                  variant={source === programSource ? 'default' : 'ghost'}
                  onClick={() => setSource(programSource)}
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
                value={maxSteps}
                onChange={(e) => setMaxSteps(Number(e.target.value))}
                className="w-20 px-2 py-1 text-xs border rounded bg-background"
                min={1}
                max={MAX_STEPS_LIMIT}
              />
              <input
                type="text"
                value={keyboardInput}
                onChange={(e) => setKeyboardInput(e.target.value)}
                placeholder="Keyboard input..."
                aria-label="Keyboard input"
                className="w-32 px-2 py-1 text-xs border rounded bg-background"
              />
              <select
                value={role}
                onChange={(e) => setRole(e.target.value)}
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
                  onChange={(e) => setDebug(e.target.checked)}
                  className="rounded"
                />
                Debug
              </label>
              <Button
                size="sm"
                onClick={() => handleRun()}
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
                onClick={() => handleRun(true)}
                disabled={running}
              >
                Step
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setResult(null)} disabled={!result}>
                Clear
              </Button>
              <Button
                size="sm"
                variant={showRef ? 'default' : 'ghost'}
                onClick={() => setShowRef(!showRef)}
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
      {result && result.success && result.steps_executed >= clampSteps(maxSteps) && (
        <div className="bg-warning/10 border border-warning/30 text-warning text-xs p-2 rounded">
          Step limit reached ({result.steps_executed} steps). Increase steps or use HLT to stop
          earlier.
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Editor */}
        <div className="lg:col-span-2">
          <Card className="h-full">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Assembly Source</CardTitle>
                <div className="flex gap-1">
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      const blob = new Blob([source], { type: 'text/plain' })
                      const url = URL.createObjectURL(blob)
                      const a = document.createElement('a')
                      a.href = url
                      a.download = 'program.asm'
                      a.click()
                      URL.revokeObjectURL(url)
                    }}
                  >
                    Save
                  </Button>
                  <label className="cursor-pointer">
                    <input
                      type="file"
                      accept=".asm,.txt"
                      className="hidden"
                      onChange={(e) => {
                        const file = e.target.files?.[0]
                        if (!file) return
                        const reader = new FileReader()
                        reader.onload = () => setSource(reader.result as string)
                        reader.readAsText(file)
                      }}
                    />
                    <span className="inline-flex items-center justify-center h-7 px-3 text-xs font-medium rounded-md border border-border hover:bg-muted/50 transition-colors">
                      Load
                    </span>
                  </label>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      const encoded = btoa(source)
                      const url = `${window.location.origin}${window.location.pathname}#code=${encoded}`
                      navigator.clipboard.writeText(url)
                      setCopied(true)
                      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current)
                      copiedTimerRef.current = setTimeout(
                        () => setCopied(false),
                        COPY_FEEDBACK_DURATION_MS,
                      )
                    }}
                  >
                    {copied ? 'Copied!' : 'Share'}
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="relative border rounded-md overflow-hidden">
                <div className="flex h-80">
                  {/* Line numbers */}
                  <div className="select-none text-right text-xs text-muted-foreground font-mono bg-muted/20 border-r border-border/40 py-3 px-2 overflow-hidden">
                    {source.split('\n').map((_, i) => (
                      <div key={i} className="leading-5">
                        {i + 1}
                      </div>
                    ))}
                  </div>
                  {/* Editor */}
                  <textarea
                    ref={textareaRef}
                    value={source}
                    onChange={(e) => setSource(e.target.value)}
                    onKeyDown={handleKeyDown}
                    aria-label="Assembly source code"
                    className="flex-1 h-full p-3 font-mono text-sm bg-background resize-none focus:outline-none leading-5"
                    spellCheck={false}
                    placeholder="[BITS 32]&#10;[ORG 0x1000]&#10;&#10;MOV EAX, 42&#10;HLT"
                  />
                </div>
              </div>
              <p className="text-xs text-muted-foreground mt-1">Ctrl+Enter to run</p>
            </CardContent>
          </Card>
        </div>

        {/* Results */}
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
                    A syscall was denied for the current role (EAX = -2). Training and other
                    privileged operations require the <span className="font-medium">admin</span>{' '}
                    role — switch the role selector above and run again.
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Training */}
          <TrainingCard job={trainingJob} onStop={handleStopTraining} />

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
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Training launch</CardTitle>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setTrainConfig(DEFAULT_TRAIN_CONFIG)
                    setCustomDataset(false)
                  }}
                >
                  Reset config
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-2 gap-x-3 gap-y-2">
                <label className="col-span-2 text-xs font-medium text-muted-foreground">
                  Dataset
                </label>
                {datasetNames.length > 0 ? (
                  <>
                    <select
                      className="col-span-2 px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
                      aria-label="Training dataset"
                      value={customDataset ? '__custom__' : trainConfig.dataset}
                      onChange={(e) => {
                        if (e.target.value === '__custom__') {
                          setCustomDataset(true)
                        } else {
                          setCustomDataset(false)
                          setTrainConfig((c) => ({
                            ...c,
                            dataset: e.target.value,
                          }))
                        }
                      }}
                    >
                      <option value="__custom__">Custom…</option>
                      {[
                        ...datasetNames,
                        ...(datasetNames.includes(trainConfig.dataset)
                          ? []
                          : [trainConfig.dataset]),
                      ].map((name) => (
                        <option key={name} value={name}>
                          {name}
                        </option>
                      ))}
                    </select>
                    {customDataset && (
                      <>
                        <input
                          className="col-span-2 px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
                          value={trainConfig.dataset}
                          aria-label="Training dataset"
                          placeholder="Custom dataset name"
                          onChange={(e) =>
                            setTrainConfig((c) => ({
                              ...c,
                              dataset: e.target.value,
                            }))
                          }
                        />
                        {trainConfig.dataset.trim() !== '' &&
                          !datasetNames.includes(trainConfig.dataset.trim()) && (
                            <p className="col-span-2 text-xs text-destructive">
                              Unknown dataset &quot;{trainConfig.dataset.trim()}&quot; — Training
                              will fail to start. Available: {datasetNames.slice(0, 5).join(', ')}
                              {datasetNames.length > 5 ? ` +${datasetNames.length - 5} more` : ''}.
                            </p>
                          )}
                      </>
                    )}
                  </>
                ) : (
                  <input
                    className="col-span-2 px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
                    value={trainConfig.dataset}
                    aria-label="Training dataset"
                    onChange={(e) =>
                      setTrainConfig((c) => ({
                        ...c,
                        dataset: e.target.value,
                      }))
                    }
                  />
                )}
                <label className="text-xs font-medium text-muted-foreground">Epochs</label>
                <input
                  type="number"
                  min={1}
                  className="px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
                  value={trainConfig.epochs}
                  aria-label="Training epochs"
                  onChange={(e) =>
                    setTrainConfig((c) => ({
                      ...c,
                      epochs: num(e.target.value),
                    }))
                  }
                />
                <label className="text-xs font-medium text-muted-foreground">Learning rate</label>
                <input
                  className="px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
                  value={trainConfig.lr}
                  aria-label="Training learning rate"
                  onChange={(e) =>
                    setTrainConfig((c) => ({
                      ...c,
                      lr: num(e.target.value),
                    }))
                  }
                />
                <label className="text-xs font-medium text-muted-foreground">Batch size</label>
                <input
                  type="number"
                  min={1}
                  className="px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
                  value={trainConfig.batch_size}
                  aria-label="Training batch size"
                  onChange={(e) =>
                    setTrainConfig((c) => ({
                      ...c,
                      batch_size: num(e.target.value),
                    }))
                  }
                />
                <label className="text-xs font-medium text-muted-foreground">Layers</label>
                <input
                  type="number"
                  min={1}
                  className="px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
                  value={trainConfig.n_layer}
                  aria-label="Training layers"
                  onChange={(e) =>
                    setTrainConfig((c) => ({
                      ...c,
                      n_layer: num(e.target.value),
                    }))
                  }
                />
                <label className="text-xs font-medium text-muted-foreground">Heads</label>
                <input
                  type="number"
                  min={1}
                  className="px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
                  value={trainConfig.n_head}
                  aria-label="Training heads"
                  onChange={(e) =>
                    setTrainConfig((c) => ({
                      ...c,
                      n_head: num(e.target.value),
                    }))
                  }
                />
                <label className="text-xs font-medium text-muted-foreground">Embed size</label>
                <input
                  type="number"
                  min={1}
                  className="px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring"
                  value={trainConfig.embed_dim}
                  aria-label="Training embed size"
                  onChange={(e) =>
                    setTrainConfig((c) => ({
                      ...c,
                      embed_dim: num(e.target.value),
                    }))
                  }
                />
              </div>
              {trainConfigHints(trainConfig).length > 0 && (
                <ul className="space-y-0.5 text-xs text-warning">
                  {trainConfigHints(trainConfig).map((hint) => (
                    <li key={hint.label}>
                      <span className="font-medium">{hint.label}</span>: {hint.message}
                    </li>
                  ))}
                </ul>
              )}
              <p className="text-xs text-muted-foreground">
                Generates the <span className="font-mono">train</span> sample with this config and
                runs it. Requires the <span className="font-medium">admin</span> role — the Training
                card polls the job and shows the final result.
              </p>
              {role === 'user' && (
                <div className="flex items-center justify-between gap-2 rounded border border-warning/40 bg-warning/10 px-2 py-1.5">
                  <p className="text-xs text-warning">
                    Training is denied for the user role (EAX = -2).
                  </p>
                  <Button size="sm" variant="outline" onClick={() => setRole('admin')}>
                    Switch to admin
                  </Button>
                </div>
              )}
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setSource(buildTrainSource(clampTrainConfig(trainConfig)))}
                >
                  Load sample
                </Button>
                <Button size="sm" onClick={handleLaunchTraining}>
                  Launch training
                </Button>
              </div>
              {launchedJob != null && (
                <div className="flex items-center justify-between gap-2 rounded border border-success/40 bg-success/10 px-2 py-1.5">
                  <p className="text-xs text-success">
                    Launched training job #{launchedJob} — the Training card below polls it to
                    completion.
                  </p>
                  <button
                    type="button"
                    className="text-xs underline text-success hover:text-success/80"
                    onClick={() => setLaunchedJob(null)}
                  >
                    Dismiss
                  </button>
                </div>
              )}
            </CardContent>
          </Card>

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
          {result && result.success && (
            <VGADisplay text={result.vga_text} cells={result.vga_cells} />
          )}

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
      </div>

      {/* Trace */}
      {result?.trace && result.trace.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              Execution Trace (first {result.trace.length} steps)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="max-h-64 overflow-x-auto overflow-y-auto">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="text-muted-foreground">
                    <th className="text-left py-1 px-2">#</th>
                    <th className="text-left py-1 px-2">EIP</th>
                    <th className="text-left py-1 px-2">Opcode</th>
                    <th className="text-left py-1 px-2">Operands</th>
                  </tr>
                </thead>
                <tbody>
                  {result.trace.map((t, i) => (
                    <tr key={i} className="border-t border-border/30">
                      <td className="py-1 px-2">{t.step}</td>
                      <td className="py-1 px-2">{t.eip}</td>
                      <td className="py-1 px-2">{t.opcode}</td>
                      <td className="py-1 px-2">{t.operands}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Assembly reference */}
      {showRef && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">x86 Reference</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
              <div>
                <p className="font-medium mb-1">Data Movement</p>
                <pre className="text-muted-foreground">
                  {'MOV dst, src\nPUSH val\nPOP dst\nXCHG a, b\nLEA dst, [addr]'}
                </pre>
              </div>
              <div>
                <p className="font-medium mb-1">Arithmetic</p>
                <pre className="text-muted-foreground">
                  {'ADD dst, src\nSUB dst, src\nINC reg\nDEC reg\nMUL src\nDIV src\nNEG dst'}
                </pre>
              </div>
              <div>
                <p className="font-medium mb-1">Logic / Shift</p>
                <pre className="text-muted-foreground">
                  {
                    'AND dst, src\nOR  dst, src\nXOR dst, src\nNOT dst\nSHL dst, n\nSHR dst, n\nCMP a, b\nTEST a, b'
                  }
                </pre>
              </div>
              <div>
                <p className="font-medium mb-1">Control Flow</p>
                <pre className="text-muted-foreground">
                  {'JMP label\nJE / JNE label\nJG / JL label\nJGE / JLE label\nCALL func\nRET\nHLT'}
                </pre>
              </div>
              <div>
                <p className="font-medium mb-1">String</p>
                <pre className="text-muted-foreground">
                  {'LODSB/W/D\nSTOSB/W/D\nMOVSB/W/D\nCMPSB/W/D\nSCASB/W/D\nREP prefix'}
                </pre>
              </div>
              <div>
                <p className="font-medium mb-1">Stack</p>
                <pre className="text-muted-foreground">
                  {'PUSHAD\nPOPAD\nPUSHFD\nPOPFD\nENTER\nLEAVE'}
                </pre>
              </div>
              <div>
                <p className="font-medium mb-1">Interrupts</p>
                <pre className="text-muted-foreground">
                  {
                    'INT 0x80\n  EAX=4 write\n  EAX=1 exit\n  EAX=3 read\n  EAX=28 train start\n  EAX=29 train status\n  EAX=30 train result\nINT 0x10 video\nINT 0x16 keyboard'
                  }
                </pre>
              </div>
              <div>
                <p className="font-medium mb-1">Registers</p>
                <pre className="text-muted-foreground">
                  {
                    'EAX  accumulator\nECX  counter\nEDX  data\nEBX  base\nESP  stack ptr\nEBP  base ptr\nESI  src index\nEDI  dst index'
                  }
                </pre>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </>
  )
}
