'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'
import { vmController, type VMRunResult, type VMRegister } from '@/lib/vm-controller'
import { extractErrorMessage } from '@/lib/error-utils'

const DEFAULT_MAX_STEPS = 5000
const MAX_STEPS_LIMIT = 1_000_000

const DEFAULT_PROGRAMS: Record<string, string> = {
  hello: `[BITS 32]

; Write "Hello, VM!" to VGA text buffer
MOV ESI, msg
MOV EDI, 0xB8000
MOV AH, 0x07

.loop:
LODSB
OR AL, AL
JZ .done
STOSW
JMP .loop

.done:
HLT

msg: db 'Hello, VM!', 0`,

  count: `[BITS 32]

; Count 0-9, write digits to VGA
MOV ECX, 0
MOV EDI, 0xB8000
MOV AH, 0x0B

.loop:
CMP ECX, 10
JGE .done
MOV AL, CL
ADD AL, 48
STOSW
INC ECX
JMP .loop

.done:
HLT`,

  fib: `[BITS 32]

; Fibonacci: compute and display first 10 numbers on VGA
MOV EAX, 0
MOV EBX, 1
MOV ECX, 0
MOV EDI, 0xB8000

.loop:
CMP ECX, 10
JGE .done

; Convert EAX (fib value) to ASCII
PUSH EAX
PUSH EBX
PUSH ECX
MOV EBX, 10
XOR ECX, 0

.digit_loop:
XOR EDX, EDX
DIV EBX
ADD DL, 48
PUSH EDX
INC ECX
TEST EAX, EAX
JNZ .digit_loop

; Write digits to VGA
MOV AH, 0x0A
.write_loop:
POP EDX
MOV AL, DL
STOSW
LOOP .write_loop

; Write space
MOV WORD [ES:EDI], 0x0A20
ADD EDI, 2

POP ECX
POP EBX
POP EAX

; next = fib(n-2) + fib(n-1)
MOV EDX, EAX
ADD EDX, EBX
MOV EAX, EBX
MOV EBX, EDX

INC ECX
JMP .loop

.done:
HLT`,

  sort: `[BITS 32]

; Bubble sort 8 bytes, display sorted array on VGA
MOV ECX, 7

.outer:
MOV EDX, 0
MOV BYTE [swapped], 0

.inner:
MOV AL, [arr + EDX]
MOV BL, [arr + EDX + 1]
CMP AL, BL
JLE .no_swap
MOV [arr + EDX], BL
MOV [arr + EDX + 1], AL
MOV BYTE [swapped], 1

.no_swap:
INC EDX
CMP EDX, ECX
JL .inner

CMP BYTE [swapped], 0
JZ .done
DEC ECX
JMP .outer

.done:
; Display sorted array on VGA
MOV ESI, arr
MOV EDI, 0xB8000
MOV AH, 0x0E

MOV ECX, 8
.disp_loop:
LODSB
ADD AL, 48
STOSW
LOOP .disp_loop

HLT

arr: db 5, 3, 8, 1, 9, 2, 7, 4
swapped: db 0`,

  vga_color: `[BITS 32]

; Rainbow stripe pattern on VGA
MOV EDI, 0xB8000
MOV ECX, 25
MOV BL, 1

.row_loop:
PUSH ECX
MOV ECX, 80

.col_loop:
MOV AL, '*'
MOV AH, BL
STOSW
LOOP .col_loop

POP ECX
INC BL
CMP BL, 16
JL .no_wrap
MOV BL, 1

.no_wrap:
LOOP .row_loop
HLT`,

  rainbow: `[BITS 32]

; Rainbow text "HELLO VM!" on VGA
MOV ESI, msg
MOV EDI, 0xB8000
ADD EDI, 160

MOV ECX, 9
MOV EBX, colors

.loop:
LODSB
MOV AH, [EBX]
INC EBX
STOSW
LOOP .loop

HLT

msg: db 'HELLO VM!'
colors: db 4, 14, 2, 1, 5, 6, 3, 11, 4`,

  primes: `[BITS 32]

; Sieve of Eratosthenes — find primes up to 50, display on VGA
MOV EDI, sieve
MOV ECX, 51
MOV AL, 1
REP STOSB

MOV ESI, 2

.sieve_loop:
CMP ESI, 51
JGE .sieve_done
MOV AL, [sieve + ESI]
TEST AL, AL
JZ .next
MOV EDI, ESI
ADD EDI, ESI

.mark_loop:
CMP EDI, 51
JGE .next
MOV BYTE [sieve + EDI], 0
ADD EDI, ESI
JMP .mark_loop

.next:
INC ESI
JMP .sieve_loop

.sieve_done:
; Display primes on VGA
MOV ESI, 2
MOV EDI, 0xB8000
MOV AH, 0x0B

.print_loop:
CMP ESI, 51
JGE .done
MOV AL, [sieve + ESI]
TEST AL, AL
JZ .skip
MOV AL, SIL
ADD AL, 48
STOSW
; space
MOV WORD [ES:EDI], 0x0B20
ADD EDI, 2

.skip:
INC ESI
JMP .print_loop

.done:
HLT

sieve: times 51 db 0`,

  calculator: `[BITS 32]

; Compute 7 * 8 + 5 = 61, display on VGA
MOV EAX, 7
MOV EBX, 8
MUL EBX
ADD EAX, 5

; Convert EAX to decimal (2 digits)
MOV EBX, 10
XOR EDX, EDX
DIV EBX
ADD AL, 48
ADD DL, 48
MOV [result], AL
MOV [result+1], DL
MOV BYTE [result+2], 0

; Write to VGA
MOV ESI, result
MOV EDI, 0xB8000
MOV AH, 0x0A

.loop:
LODSB
OR AL, AL
JZ .done
STOSW
JMP .loop

.done:
HLT

result: times 4 db 0`,

  factorial: `[BITS 32]

; Compute 6! = 720, display on VGA
MOV EAX, 1
MOV ECX, 6

.loop:
MUL ECX
DEC ECX
JNZ .loop

; EAX = 720, convert to decimal
MOV EBX, 10
MOV ECX, 0

.digit_loop:
XOR EDX, EDX
DIV EBX
ADD DL, 48
PUSH EDX
INC ECX
TEST EAX, EAX
JNZ .digit_loop

; Pop digits to VGA
MOV EDI, 0xB8000
MOV AH, 0x0E

.write_loop:
POP EDX
MOV AL, DL
STOSW
LOOP .write_loop

HLT`,

  guess: `[BITS 32]

; Number guessing game (pre-set answer = 5)
MOV ESI, prompt
MOV EDI, 0xB8000
MOV AH, 0x0A

.prompt_loop:
LODSB
OR AL, AL
JZ .done
STOSW
JMP .prompt_loop

.done:
HLT

prompt: db 'Answer is 5!', 0`,
}

export default function VMPage() {
  const [source, setSource] = useState(DEFAULT_PROGRAMS.hello)
  const [result, setResult] = useState<VMRunResult | null>(null)
  const [running, setRunning] = useState(false)
  const [maxSteps, setMaxSteps] = useState(DEFAULT_MAX_STEPS)
  const [debug, setDebug] = useState(false)
  const [keyboardInput, setKeyboardInput] = useState('')
  const [showRef, setShowRef] = useState(false)
  const [copied, setCopied] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const copiedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    return () => {
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current)
    }
  }, [])

  // Load from URL hash (#code=base64)
  useEffect(() => {
    try {
      const hash = window.location.hash
      if (hash.startsWith('#code=')) {
        const decoded = atob(hash.slice(6))
        setSource(decoded)
        return
      }
    } catch { /* malformed hash — fall through to localStorage */ }
    const saved = localStorage.getItem('vm-source')
    if (saved) setSource(saved)
  }, [])

  // Save source to localStorage on change
  useEffect(() => {
    try {
      localStorage.setItem('vm-source', source)
    } catch { /* quota exceeded — source not persisted */ }
  }, [source])

  const handleRun = useCallback(async (step?: boolean) => {
    setRunning(true)
    setResult(null)
    try {
      const res = await vmController.run(source, {
        maxSteps: step ? 1 : maxSteps,
        debug,
        keyboardInput: keyboardInput || undefined,
      })
      setResult(res)
    } catch (err: unknown) {
      setResult({
        success: false,
        exit_code: -1,
        steps_executed: 0,
        elapsed_ms: 0,
        output: '',
        registers: [],
        eip: 0,
        eip_hex: '0x0',
        status: 'error',
        error: extractErrorMessage(err),
      })
    } finally {
      setRunning(false)
    }
  }, [source, maxSteps, debug])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault()
        handleRun()
      }
    },
    [handleRun],
  )

  return (
    <div className="sl-page mx-auto max-w-6xl">
      <AppRouteHeader
        left={
          <AppRouteHeaderLead
            title="VM Console"
            subtitle="x86-32 assembly sandbox — write, run, inspect"
          />
        }
      />

      <div className="space-y-4">
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
                <label className="text-xs text-muted-foreground" htmlFor="vm-steps">Steps:</label>
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
                      <span className="h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
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
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setResult(null)}
                  disabled={!result}
                >
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
          <div className="bg-destructive/10 border border-destructive/30 text-destructive text-xs p-2 rounded">
            {result.error}
          </div>
        )}
        {result && result.success && result.steps_executed >= maxSteps && (
          <div className="bg-warning/10 border border-warning/30 text-warning text-xs p-2 rounded">
            Step limit reached ({result.steps_executed} steps). Increase steps or use HLT to stop earlier.
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
                        copiedTimerRef.current = setTimeout(() => setCopied(false), 2000)
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
                    <div className="select-none text-right text-xs text-muted-foreground font-mono bg-muted/20 border-r border-border/50 py-3 px-2 overflow-hidden">
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
                <p className="text-xs text-muted-foreground mt-1">
                  Ctrl+Enter to run
                </p>
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
                  <StatusRow label="Exit code" value={`0x${result.exit_code.toString(16).toUpperCase()}`} />
                  <StatusRow label="Steps" value={result.steps_executed.toLocaleString()} />
                  <StatusRow label="Time" value={`${result.elapsed_ms.toFixed(1)}ms`} />
                  {result.error && (
                    <div className="text-xs text-destructive bg-destructive/10 p-2 rounded">
                      {result.error}
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

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
                        const text = result.registers
                          .map((r) => `${r.name} = ${r.hex}`)
                          .join('\n')
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
              <CardTitle className="text-base">Execution Trace (first {result.trace.length} steps)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="max-h-64 overflow-y-auto">
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
                  <pre className="text-muted-foreground">{"MOV dst, src\nPUSH val\nPOP dst\nXCHG a, b\nLEA dst, [addr]"}</pre>
                </div>
                <div>
                  <p className="font-medium mb-1">Arithmetic</p>
                  <pre className="text-muted-foreground">{"ADD dst, src\nSUB dst, src\nINC reg\nDEC reg\nMUL src\nDIV src\nNEG dst"}</pre>
                </div>
                <div>
                  <p className="font-medium mb-1">Logic / Shift</p>
                  <pre className="text-muted-foreground">{"AND dst, src\nOR  dst, src\nXOR dst, src\nNOT dst\nSHL dst, n\nSHR dst, n\nCMP a, b\nTEST a, b"}</pre>
                </div>
                <div>
                  <p className="font-medium mb-1">Control Flow</p>
                  <pre className="text-muted-foreground">{"JMP label\nJE / JNE label\nJG / JL label\nJGE / JLE label\nCALL func\nRET\nHLT"}</pre>
                </div>
                <div>
                  <p className="font-medium mb-1">String</p>
                  <pre className="text-muted-foreground">{"LODSB/W/D\nSTOSB/W/D\nMOVSB/W/D\nCMPSB/W/D\nSCASB/W/D\nREP prefix"}</pre>
                </div>
                <div>
                  <p className="font-medium mb-1">Stack</p>
                  <pre className="text-muted-foreground">{"PUSHAD\nPOPAD\nPUSHFD\nPOPFD\nENTER\nLEAVE"}</pre>
                </div>
                <div>
                  <p className="font-medium mb-1">Interrupts</p>
                  <pre className="text-muted-foreground">{"INT 0x80\n  EAX=4 write\n  EAX=1 exit\n  EAX=3 read\nINT 0x10 video\nINT 0x16 keyboard"}</pre>
                </div>
                <div>
                  <p className="font-medium mb-1">Registers</p>
                  <pre className="text-muted-foreground">{"EAX  accumulator\nECX  counter\nEDX  data\nEBX  base\nESP  stack ptr\nEBP  base ptr\nESI  src index\nEDI  dst index"}</pre>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

function StatusRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono">{value}</span>
    </div>
  )
}

function VGADisplay({ text, cells }: { text?: string; cells?: { ch: string; fg: string; bg: string }[] }) {
  const [fullScreen, setFullScreen] = useState(false)

  // Render colored cells as rows of spans
  const renderCells = () => {
    if (!cells) return null
    const rows: React.ReactNode[] = []
    for (let row = 0; row < 25; row++) {
      const rowCells = cells.slice(row * 80, (row + 1) * 80)
      // Skip empty rows
      const hasContent = rowCells.some((c) => c.ch !== ' ')
      if (!hasContent) continue
      rows.push(
        <div key={row}>
          {rowCells.map((c, col) => (
            <span key={col} style={{ color: c.fg, backgroundColor: c.bg }}>
              {c.ch}
            </span>
          ))}
        </div>,
      )
    }
    return rows
  }

  return (
    <Card className={fullScreen ? 'fixed inset-4 z-50' : ''}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Screen (VGA 0xB8000)</CardTitle>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setFullScreen(!fullScreen)}
          >
            {fullScreen ? 'Exit' : 'Fullscreen'}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div
          className={`bg-black font-mono text-xs p-3 rounded overflow-y-auto whitespace-pre ${
            fullScreen ? 'h-[calc(100dvh-8rem)]' : 'h-40'
          }`}
        >
          {cells ? (
            renderCells()
          ) : text ? (
            <span className="text-green-400">{text}</span>
          ) : (
            <span className="text-green-700">
              Programs that write to 0xB8000 will appear here.
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
