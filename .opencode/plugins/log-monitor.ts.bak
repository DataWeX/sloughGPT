import type { Plugin } from "@opencode-ai/plugin"
import { homedir } from "os"
import { join } from "path"
import { writeFileSync } from "fs"

const ERROR_PATTERNS = [
  /Traceback \(most recent call last\)/i,
  /\bError\b/i,
  /^error:/i,
  /exit code \d+/i,
  /failed to/i,
  /cannot (find|import|resolve|read)/i,
  /ModuleNotFoundError/i,
  /ImportError/i,
  /KeyError/i,
  /AttributeError/i,
  /SyntaxError/i,
  /TypeError/i,
  /ValueError/i,
  /IndexError/i,
  /RuntimeError/i,
  /ConnectionError/i,
  /TimeoutError/i,
  /Killed/i,
  /Segmentation fault/i,
  /AssertionError/i,
  /json\.decode/i,
  /ENOENT/i,
  /EACCES/i,
  /ECONNREFUSED/i,
  /Address already in use/i,
  /port.*already/i,
  /No such file or directory/i,
  /command not found/i,
  /npm ERR/i,
  /ERR_PACKAGE_PATH_NOT_EXPORTED/i,
]

const ERROR_LOG_PATH = join(homedir(), ".opencode-error-log.json")

interface ErrorRecord {
  timestamp: string
  command: string
  pattern: string
  snippet: string
  cwd?: string
}

let errorLog: ErrorRecord[] = []

function persist() {
  try {
    writeFileSync(ERROR_LOG_PATH, JSON.stringify(errorLog.slice(-20), null, 2))
  } catch { /* best-effort */ }
}

export default (async () => {
  return {
    "tool.execute.after": async (input: any, output: any) => {
      const toolName = input?.tool ?? ""
      if (toolName !== "bash") return

      const command = input?.input?.command ?? ""
      if (!command) return

      const combined = [output?.stdout ?? "", output?.stderr ?? ""].join("\n")
      const exitCode = output?.exit_code ?? output?.exitCode
      const cwd = input?.input?.workdir ?? input?.cwd

      for (const pattern of ERROR_PATTERNS) {
        const match = combined.match(pattern)
        if (match) {
          errorLog.push({
            timestamp: new Date().toISOString(),
            command: command.slice(0, 120),
            pattern: match[0],
            snippet: combined.slice(Math.max(0, (match.index ?? 0) - 40), (match.index ?? 0) + 200),
            cwd,
          })
          if (errorLog.length > 100) errorLog.shift()
          persist()
          return
        }
      }

      // Non-zero exit without explicit error text
      if (exitCode != null && exitCode !== 0) {
        errorLog.push({
          timestamp: new Date().toISOString(),
          command: command.slice(0, 120),
          pattern: `exit code ${exitCode}`,
          snippet: (output?.stderr ?? output?.stdout ?? "").slice(0, 300),
          cwd,
        })
        if (errorLog.length > 100) errorLog.shift()
        persist()
      }
    },
  }
}) satisfies Plugin
