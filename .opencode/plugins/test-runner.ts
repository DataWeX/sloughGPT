import type { Plugin } from "@opencode-ai/plugin"
import { homedir } from "os"
import { join } from "path"
import { readFileSync, writeFileSync, existsSync } from "fs"

const RESULTS_PATH = join(homedir(), ".opencode-test-results.json")

interface TestResult {
  timestamp: string
  command: string
  passed: number
  failed: number
  skipped: number
  duration: string
  errors: string[]
}

interface TestHistory {
  runs: TestResult[]
  lastRun: TestResult | null
  totalPassed: number
  totalFailed: number
}

function loadHistory(): TestHistory {
  if (existsSync(RESULTS_PATH)) {
    try {
      return JSON.parse(readFileSync(RESULTS_PATH, "utf-8"))
    } catch {}
  }
  return { runs: [], lastRun: null, totalPassed: 0, totalFailed: 0 }
}

function saveHistory(history: TestHistory) {
  writeFileSync(RESULTS_PATH, JSON.stringify(history, null, 2))
}

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

function isTestCommand(command: string): boolean {
  return /\b(pytest|vitest|npx vitest|python3 -m pytest|npm run test)\b/.test(command)
}

export default (async ({ client }) => {
  return {
    "tool.execute.after": async (input, output) => {
      if (input.tool !== "bash" || !output.output) return

      const command = input.args?.command || ""
      if (!isTestCommand(command)) return

      const text = output.output
      let parsed: Partial<TestResult>

      if (/pytest|python3 -m pytest/.test(command)) {
        parsed = parsePytestOutput(text)
      } else if (/vitest/.test(command)) {
        parsed = parseVitestOutput(text)
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

      if (result.failed > 0) {
        output.output += `\n\n⚠ TEST FAILURES: ${result.failed} failed\n`
        for (const err of result.errors.slice(0, 3)) {
          output.output += `  ${err}\n`
        }
      }
    },
  }
}) satisfies Plugin
