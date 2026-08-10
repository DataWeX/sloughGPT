import type { Plugin } from "@opencode-ai/plugin"
import { homedir } from "os"
import { join } from "path"
import { writeFileSync, readFileSync, existsSync } from "fs"

// ── Error patterns with categories and fix hints ──

interface ErrorPattern {
  pattern: RegExp
  category: string
  fixHint: string
  filePattern?: RegExp
}

const ERROR_PATTERNS: ErrorPattern[] = [
  // Python errors
  { pattern: /ModuleNotFoundError: No module named '(\S+)'/, category: "python-import", fixHint: "pip install $1 or add to requirements.txt", filePattern: /requirements/i },
  { pattern: /ImportError: cannot import name '(\S+)' from '(\S+)'/, category: "python-import", fixHint: "Check import path — $2.$1 may not exist or be renamed" },
  { pattern: /SyntaxError: (.+)/, category: "python-syntax", fixHint: "Fix syntax: $1" },
  { pattern: /TypeError: (.+)/, category: "python-type", fixHint: "Type mismatch: $1" },
  { pattern: /AttributeError: (.+)/, category: "python-attr", fixHint: "Missing attribute: $1" },
  { pattern: /KeyError: (.+)/, category: "python-key", fixHint: "Missing key: $1 — use .get() with default" },
  { pattern: /ValueError: (.+)/, category: "python-value", fixHint: "Invalid value: $1" },
  { pattern: /IndexError: (.+)/, category: "python-index", fixHint: "Index out of range: $1" },
  { pattern: /FileNotFoundError: (.+)/, category: "python-file", fixHint: "File not found: $1 — check path" },
  { pattern: /ConnectionRefusedError/, category: "python-network", fixHint: "Server not running — start it first" },
  { pattern: /TimeoutError/, category: "python-network", fixHint: "Request timed out — increase timeout or check server" },
  { pattern: /RuntimeError: (.+)/, category: "python-runtime", fixHint: "Runtime error: $1" },

  // TypeScript / build errors
  { pattern: /TS(\d{4}): (.+)/, category: "typescript", fixHint: "TS error $1: $2" },
  { pattern: /Type '(.+)' is not assignable to type '(.+)'/, category: "typescript", fixHint: "Type mismatch: $1 → $2" },
  { pattern: /Property '(\S+)' does not exist on type '(\S+)'/, category: "typescript", fixHint: "Missing property $1 on $2" },
  { pattern: /Cannot find module '(\S+)'/, category: "typescript", fixHint: "Missing module: $1 — install or check path" },
  { pattern: /Module not found: (.+)/, category: "build", fixHint: "Module not found: $1" },
  { pattern: /Failed to compile/, category: "build", fixHint: "Build failed — check errors above" },

  // npm / node errors
  { pattern: /npm ERR! (.+)/, category: "npm", fixHint: "npm error: $1" },
  { pattern: /ERR_PACKAGE_PATH_NOT_EXPORTED/, category: "npm", fixHint: "Package export missing — check package.json exports field" },
  { pattern: /ERR_MODULE_NOT_FOUND/, category: "npm", fixHint: "Module not found — run npm install" },

  // System errors
  { pattern: /ENOENT: no such file or directory, (.+)/, category: "filesystem", fixHint: "File/dir missing: $1" },
  { pattern: /EACCES: permission denied, (.+)/, category: "filesystem", fixHint: "Permission denied: $1 — check chmod/chown" },
  { pattern: /ECONNREFUSED (.+)/, category: "network", fixHint: "Connection refused: $1 — is the service running?" },
  { pattern: /Address already in use/, category: "network", fixHint: "Port in use — kill the process or use a different port" },
  { pattern: /command not found: (.+)/, category: "system", fixHint: "Command not found: $1 — install it" },
  { pattern: /Killed$/, category: "system", fixHint: "Process killed (OOM) — reduce memory usage" },
  { pattern: /Segmentation fault/, category: "system", fixHint: "Segfault — likely a C extension or memory corruption" },

  // Test failures
  { pattern: /FAILED (.+)/, category: "test", fixHint: "Test failed: $1" },
  { pattern: /AssertionError(?:\s*:\s*(.+))?/, category: "test", fixHint: "Assertion failed: $1" },
  { pattern: /\d+ failed/, category: "test", fixHint: "Some tests failed — check output above" },

  // Generic exit
  { pattern: /exit code ([1-9]\d*)/, category: "exit", fixHint: "Non-zero exit code $1" },
]

// ── State ──

interface ErrorRecord {
  id: string
  timestamp: string
  command: string
  category: string
  pattern: string
  fixHint: string
  snippet: string
  cwd?: string
  resolved: boolean
  resolvedAt?: string
}

const LOG_PATH = join(homedir(), ".opencode-autofix-log.json")
const MAX_ENTRIES = 50

function loadLog(): ErrorRecord[] {
  if (!existsSync(LOG_PATH)) return []
  try {
    return JSON.parse(readFileSync(LOG_PATH, "utf-8"))
  } catch {
    return []
  }
}

function saveLog(log: ErrorRecord[]) {
  try {
    writeFileSync(LOG_PATH, JSON.stringify(log.slice(-MAX_ENTRIES), null, 2))
  } catch { /* best-effort */ }
}

function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 6)
}

function extractFileLocation(snippet: string): { file?: string; line?: number } {
  // Python: File "/path/to/file.py", line 42
  const pyMatch = snippet.match(/File "([^"]+)", line (\d+)/)
  if (pyMatch) return { file: pyMatch[1], line: parseInt(pyMatch[2]) }

  // TypeScript/JS: at /path/to/file.ts:42:10
  const tsMatch = snippet.match(/(?:at |in |-> )(\S+):(\d+):\d+/)
  if (tsMatch) return { file: tsMatch[1], line: parseInt(tsMatch[2]) }

  // /path/to/file.py:42: error
  const genericMatch = snippet.match(/(\S+\.\w+):(\d+):\s/)
  if (genericMatch) return { file: genericMatch[1], line: parseInt(genericMatch[2]) }

  return {}
}

function formatAge(ts: string): string {
  const diff = Date.now() - new Date(ts).getTime()
  if (diff < 60_000) return "just now"
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`
  return `${Math.floor(diff / 86_400_000)}d ago`
}

function printStartupSummary(log: ErrorRecord[]) {
  if (log.length === 0) return

  const unresolved = log.filter(e => !e.resolved)
  if (unresolved.length === 0) return

  const byCategory = new Map<string, number>()
  for (const e of unresolved) {
    byCategory.set(e.category, (byCategory.get(e.category) || 0) + 1)
  }

  console.log("")
  console.log(`  \x1b[31m⚠ ${unresolved.length} unresolved error(s)\x1b[0m`)
  console.log("  ─────────────────────────────────────")

  const cats = [...byCategory.entries()].sort((a, b) => b[1] - a[1])
  for (const [cat, count] of cats.slice(0, 6)) {
    console.log(`  \x1b[90m${cat.padEnd(16)}\x1b[0m ${count}`)
  }

  const latest = unresolved[unresolved.length - 1]
  console.log("")
  console.log(`  \x1b[31mlatest\x1b[0m  ${latest.fixHint}`)
  console.log(`  \x1b[90m${formatAge(latest.timestamp)} · ${latest.command.slice(0, 60)}\x1b[0m`)
  console.log("")
}

function appendAutofixHint(output: any, record: ErrorRecord) {
  const loc = extractFileLocation(record.snippet)
  const locStr = loc.file ? ` → ${loc.file}:${loc.line || "?"}` : ""

  const hint = [
    "",
    "┌─────────────────────────────────────────────────────────┐",
    `│ \x1b[31m✖\x1b[0m \x1b[1m${record.category}\x1b[0m${locStr}${" ".repeat(Math.max(0, 44 - record.category.length - locStr.length))}│`,
    "│                                                         │",
    `│ \x1b[90m${record.fixHint.slice(0, 55)}\x1b[0m${" ".repeat(Math.max(0, 56 - Math.min(record.fixHint.length, 55)))}│`,
    "│                                                         │",
    `│ \x1b[33m→ opencode fix\x1b[0m${" ".repeat(40)}│`,
    "└─────────────────────────────────────────────────────────┘",
    "",
  ].join("\n")

  if (typeof output?.stdout === "string") {
    output.stdout += hint
  } else if (typeof output?.output === "string") {
    output.output += hint
  }
}

// ── Plugin ──

export default (async () => {
  const log = loadLog()
  printStartupSummary(log)

  return {
    "tool.execute.after": async (input: any, output: any) => {
      if (input?.tool !== "bash") return

      const command = input?.input?.command ?? ""
      if (!command) return

      const combined = [output?.stdout ?? "", output?.stderr ?? ""].join("\n")
      const exitCode = output?.exit_code ?? output?.exitCode
      const cwd = input?.input?.workdir ?? input?.cwd

      if (!combined.trim() && (exitCode == null || exitCode === 0)) return

      const log = loadLog()
      const seen = new Set<string>() // dedup by category+pattern per command
      const matched: ErrorRecord[] = []

      // Find ALL matching errors in the output
      for (const { pattern, category, fixHint } of ERROR_PATTERNS) {
        const matches = combined.matchAll(new RegExp(pattern.source, pattern.flags))
        for (const match of matches) {
          const key = `${category}:${match[0].slice(0, 80)}`
          if (seen.has(key)) continue
          seen.add(key)

          const snippet = combined.slice(
            Math.max(0, (match.index ?? 0) - 80),
            (match.index ?? 0) + 300,
          )

          const record: ErrorRecord = {
            id: generateId(),
            timestamp: new Date().toISOString(),
            command: command.slice(0, 200),
            category,
            pattern: match[0].slice(0, 200),
            fixHint,
            snippet: snippet.slice(0, 500),
            cwd,
            resolved: false,
          }

          log.push(record)
          matched.push(record)
        }
      }

      // Non-zero exit without any matched pattern — log as generic exit error
      if (matched.length === 0 && exitCode != null && exitCode !== 0) {
        const snippet = (output?.stderr ?? output?.stdout ?? "").slice(0, 500)
        const record: ErrorRecord = {
          id: generateId(),
          timestamp: new Date().toISOString(),
          command: command.slice(0, 200),
          category: "exit",
          pattern: `exit code ${exitCode}`,
          fixHint: `Non-zero exit code ${exitCode} — check output above`,
          snippet,
          cwd,
          resolved: false,
        }

        log.push(record)
        matched.push(record)
      }

      if (matched.length === 0) return

      saveLog(log)

      // Append summary of all errors to output
      if (matched.length === 1) {
        appendAutofixHint(output, matched[0])
      } else {
        const shown = matched.slice(0, 10)
        const remaining = matched.length - shown.length
        const hints = shown.map((r, i) => {
          const loc = extractFileLocation(r.snippet)
          const locStr = loc.file ? ` → ${loc.file}:${loc.line || "?"}` : ""
          return `│  \x1b[90m${String(i + 1).padStart(2)}\x1b[0m  \x1b[31m${r.category.padEnd(16)}\x1b[0m${locStr}`
        })
        if (remaining > 0) {
          hints.push(`│  \x1b[90m   ... and ${remaining} more\x1b[0m`)
        }
        const summary = [
          "",
          "┌─────────────────────────────────────────────────────────┐",
          `│ \x1b[31m✖\x1b[0m \x1b[1m${matched.length} errors captured\x1b[0m${" ".repeat(Math.max(0, 40 - String(matched.length).length))}│`,
          "├─────────────────────────────────────────────────────────┤",
          ...hints,
          "├─────────────────────────────────────────────────────────┤",
          `│ \x1b[33m→ opencode fix\x1b[0m to auto-diagnose and fix${" ".repeat(14)}│`,
          "└─────────────────────────────────────────────────────────┘",
          "",
        ].join("\n")

        if (typeof output?.stdout === "string") {
          output.stdout += summary
        } else if (typeof output?.output === "string") {
          output.output += summary
        }
      }
    },
  }
}) satisfies Plugin
