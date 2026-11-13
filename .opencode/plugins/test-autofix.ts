import type { Plugin } from "@opencode-ai/plugin"
import { homedir } from "os"
import { join } from "path"
import { readFileSync, writeFileSync, existsSync } from "fs"

const RESULTS_PATH = join(homedir(), ".opencode-test-results.json")
const FAILURES_PATH = join(homedir(), ".opencode-test-failures.json")

interface TestResult {
  timestamp: string
  command: string
  passed: number
  failed: number
  skipped: number
  duration: string
  errors: string[]
}

interface TestFailure {
  id: string
  timestamp: string
  command: string
  file: string
  testName: string
  error: string
  snippet: string
  framework: "pytest" | "vitest"
  resolved: boolean
}

interface TestHistory {
  runs: TestResult[]
  lastRun: TestResult | null
  totalPassed: number
  totalFailed: number
}

function loadHistory(): TestHistory {
  if (existsSync(RESULTS_PATH)) {
    try { return JSON.parse(readFileSync(RESULTS_PATH, "utf-8")) } catch {}
  }
  return { runs: [], lastRun: null, totalPassed: 0, totalFailed: 0 }
}

function saveHistory(history: TestHistory) {
  try { writeFileSync(RESULTS_PATH, JSON.stringify(history, null, 2)) } catch {}
}

function loadFailures(): TestFailure[] {
  if (existsSync(FAILURES_PATH)) {
    try { return JSON.parse(readFileSync(FAILURES_PATH, "utf-8")) } catch {}
  }
  return []
}

function saveFailures(failures: TestFailure[]) {
  try { writeFileSync(FAILURES_PATH, JSON.stringify(failures.slice(-50), null, 2)) } catch {}
}

function genId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 6)
}

// ── Pytest failure parser ──

function parsePytestFailures(output: string): TestFailure[] {
  const failures: TestFailure[] = []
  const ts = new Date().toISOString()

  // Pattern: FAILED tests/foo.py::test_bar - ValueError: something
  const failLines = output.matchAll(/^FAILED\s+(\S+?)(?:::(\S+))?\s+-\s+(.+)/gm)
  for (const m of failLines) {
    failures.push({
      id: genId(), timestamp: ts, command: "",
      file: m[1], testName: m[2] || "", error: m[3].trim(),
      snippet: "", framework: "pytest", resolved: false,
    })
  }

  // Parse the FAILURES block for tracebacks
  const block = output.match(/={20,} FAILURES ={20,}([\s\S]*?)={20,}/)
  if (block) {
    const sections = block[1].split(/_{5,}\s+/)
    for (const section of sections) {
      // File "...", line N
      const fileMatch = section.match(/File "([^"]+)", line (\d+)/)
      // E   ErrorType: message
      const errMatch = section.match(/E\s+(\w+):\s*(.+)/)
      if (fileMatch && errMatch) {
        const existing = failures.find(f => f.file === fileMatch[1])
        if (existing) {
          existing.snippet = section.slice(0, 400)
          existing.error = `${errMatch[1]}: ${errMatch[2]}`
        } else {
          failures.push({
            id: genId(), timestamp: ts, command: "",
            file: fileMatch[1], testName: "", error: `${errMatch[1]}: ${errMatch[2]}`,
            snippet: section.slice(0, 400), framework: "pytest", resolved: false,
          })
        }
      }
    }
  }

  return failures
}

// ── Vitest failure parser ──

function parseVitestFailures(output: string): TestFailure[] {
  const failures: TestFailure[] = []
  const ts = new Date().toISOString()

  // FAIL src/foo.test.ts (123 ms)
  const failBlocks = output.matchAll(/FAIL\s+(\S+)\s+\((\d+)\s*ms\)/g)
  for (const m of failBlocks) {
    failures.push({
      id: genId(), timestamp: ts, command: "",
      file: m[1], testName: "", error: `test file failed`,
      snippet: "", framework: "vitest", resolved: false,
    })
  }

  // ❯ src/foo.ts:42:9 - error message
  const errLines = output.matchAll(/[❯✓×]\s+(\S+):(\d+):\d+\s+-\s+(.+)/g)
  for (const m of errLines) {
    const existing = failures.find(f => f.file === m[1])
    if (existing) {
      existing.error = m[3].trim()
      existing.testName = `line ${m[2]}`
    } else {
      failures.push({
        id: genId(), timestamp: ts, command: "",
        file: m[1], testName: `line ${m[2]}`, error: m[3].trim(),
        snippet: "", framework: "vitest", resolved: false,
      })
    }
  }

  // Assertion errors: expect(received).toBe(expected)
  const asserts = output.matchAll(/(AssertionError|TypeError|ReferenceError|RangeError):\s*(.+)/g)
  for (const m of asserts) {
    if (failures.length === 0) {
      failures.push({
        id: genId(), timestamp: ts, command: "",
        file: "", testName: "", error: `${m[1]}: ${m[2]}`,
        snippet: "", framework: "vitest", resolved: false,
      })
    }
  }

  return failures
}

function isTestCommand(command: string): boolean {
  return /\b(pytest|vitest|npx vitest|python3 -m pytest|npm run test)\b/.test(command)
}

function appendTestHints(output: any, failures: TestFailure[]) {
  if (failures.length === 0) return

  const shown = failures.slice(0, 8)
  const remaining = failures.length - shown.length
  const lines = shown.map((f, i) => {
    const loc = f.file ? ` → ${f.file}${f.testName ? `::${f.testName}` : ""}` : ""
    return `│  \x1b[90m${String(i + 1).padStart(2)}\x1b[0m  \x1b[31m${f.framework.padEnd(7)}\x1b[0m ${f.error.slice(0, 38).padEnd(38)}${loc.slice(0, 14)}`
  })
  if (remaining > 0) {
    lines.push(`│  \x1b[90m   ... and ${remaining} more\x1b[0m`)
  }

  const hint = [
    "",
    "┌─────────────────────────────────────────────────────────┐",
    `│ \x1b[31m✖\x1b[0m \x1b[1m${failures.length} test failure(s)\x1b[0m${" ".repeat(Math.max(0, 40 - String(failures.length).length))}│`,
    "├─────────────────────────────────────────────────────────┤",
    ...lines,
    "├─────────────────────────────────────────────────────────┤",
    `│ \x1b[33m→ opencode fix-tests\x1b[0m to auto-diagnose and fix${" ".repeat(8)}│`,
    "└─────────────────────────────────────────────────────────┘",
    "",
  ].join("\n")

  if (typeof output?.stdout === "string") output.stdout += hint
  else if (typeof output?.output === "string") output.output += hint
}

// ── Plugin ──

export default (async () => {
  return {
    "tool.execute.after": async (input: any, output: any) => {
      if (input.tool !== "bash" || !output.output) return

      const command = input.args?.command || ""
      if (!isTestCommand(command)) return

      const text = output.output
      let parsed: Partial<TestResult>
      let failures: TestFailure[] = []

      if (/pytest|python3 -m pytest/.test(command)) {
        parsed = parsePytestOutput(text)
        failures = parsePytestFailures(text)
      } else if (/vitest/.test(command)) {
        parsed = parseVitestOutput(text)
        failures = parseVitestFailures(text)
      } else {
        return
      }

      if (!parsed.passed && !parsed.failed) return

      const result: TestResult = {
        timestamp: new Date().toISOString(),
        command: command.slice(0, 200),
        passed: parsed.passed || 0,
        failed: parsed.failed || 0,
        skipped: parsed.skipped || 0,
        duration: parsed.duration || "",
        errors: parsed.errors || [],
      }

      const history = loadHistory()
      history.runs.push(result)
      if (history.runs.length > 50) history.runs = history.runs.slice(-50)
      history.lastRun = result
      history.totalPassed += result.passed
      history.totalFailed += result.failed
      saveHistory(history)

      if (result.failed > 0 && failures.length > 0) {
        // Tag command on each failure
        for (const f of failures) f.command = command.slice(0, 200)

        const existing = loadFailures()
        existing.push(...failures)
        saveFailures(existing)

        appendTestHints(output, failures)
      }
    },
  }
}) satisfies Plugin

// ── Pytest summary parser (used above) ──

function parsePytestOutput(output: string): Partial<TestResult> {
  const passed = (output.match(/(\d+) passed/g) || [])
    .reduce((sum, m) => sum + parseInt(m), 0)
  const failed = (output.match(/(\d+) failed/g) || [])
    .reduce((sum, m) => sum + parseInt(m), 0)
  const skipped = (output.match(/(\d+) skipped/g) || [])
    .reduce((sum, m) => sum + parseInt(m), 0)
  const duration = output.match(/in (\d+\.\d+)s/)?.[1] || ""

  const errors: string[] = []
  const errorBlock = output.match(/={20,} FAILURES ={20,}([\s\S]*?)={20,}/)
  if (errorBlock) {
    const lines = errorBlock[1].split("\n").filter(l => l.trim().startsWith("E "))
    errors.push(...lines.slice(0, 5).map(l => l.trim()))
  }

  return { passed, failed, skipped, duration, errors }
}

function parseVitestOutput(output: string): Partial<TestResult> {
  const passed = parseInt(output.match(/(\d+) passed/)?.[1] || "0")
  const failed = parseInt(output.match(/(\d+) failed/)?.[1] || "0")
  const skipped = parseInt(output.match(/(\d+) skipped/)?.[1] || "0")
  const duration = output.match(/Duration (\d+\.\d+s)/)?.[1] || ""

  const errors: string[] = []
  const failBlocks = output.matchAll(/FAIL\s+(\S+)/g)
  for (const m of failBlocks) {
    errors.push(m[1])
  }

  return { passed, failed, skipped, duration, errors }
}
