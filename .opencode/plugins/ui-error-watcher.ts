import type { Plugin } from "@opencode-ai/plugin"
import { homedir } from "os"
import { join } from "path"
import { writeFileSync, readFileSync } from "fs"

const UI_ERROR_PATTERNS = [
  // React / Next.js runtime
  { pattern: /Hydration (mismatch|failed)/i, category: "hydration" },
  { pattern: /Text content does not match/i, category: "hydration" },
  { pattern: /server-rendered text.*does not match/i, category: "hydration" },
  { pattern: /Cannot read propert(y|ies) of (null|undefined)/i, category: "null-access" },
  { pattern: /is not a function/i, category: "null-access" },
  { pattern: /TypeError: .* is undefined/i, category: "null-access" },
  { pattern: /React.*state update.*unmounted/i, category: "memory-leak" },
  { pattern: /Can't perform a React state update.*unmounted/i, category: "memory-leak" },
  { pattern: /Warning:.*useEffect.*dependency/i, category: "hook-warning" },
  { pattern: /Maximum update depth exceeded/i, category: "infinite-loop" },
  { pattern: /Too many re-renders/i, category: "infinite-loop" },

  // Next.js specific
  { pattern: /NextRouter/i, category: "router" },
  { pattern: /next\/navigation/i, category: "router" },
  { pattern: /ChunkLoadError/i, category: "chunk-load" },
  { pattern: /Loading chunk.*failed/i, category: "chunk-load" },
  { pattern: /Dynamic server usage/i, category: "ssr" },

  // TypeScript / build
  { pattern: /Type .* is not assignable to type/i, category: "type-error" },
  { pattern: /Property .* does not exist on type/i, category: "type-error" },
  { pattern: /Argument of type .* is not assignable/i, category: "type-error" },
  { pattern: /TS\d{4,}/i, category: "type-error" },

  // Network / API
  { pattern: /fetch failed/i, category: "network" },
  { pattern: /ECONNREFUSED/i, category: "network" },
  { pattern: /NetworkError/i, category: "network" },
  { pattern: /Failed to fetch/i, category: "network" },
  { pattern: /AbortError/i, category: "network" },
  { pattern: /Load failed/i, category: "network" },

  // HTTP errors
  { pattern: /\b401\b.*Unauthorized/i, category: "auth" },
  { pattern: /\b403\b.*Forbidden/i, category: "auth" },
  { pattern: /\b404\b.*Not Found/i, category: "not-found" },
  { pattern: /\b500\b.*Internal/i, category: "server-error" },
  { pattern: /\b503\b.*Unavailable/i, category: "server-error" },

  // Build errors
  { pattern: /npm ERR!/i, category: "build" },
  { pattern: /Build error occurred/i, category: "build" },
  { pattern: /Failed to compile/i, category: "build" },
  { pattern: /Module not found/i, category: "build" },
  { pattern: /Import.*could not be resolved/i, category: "build" },
  { pattern: /Cannot find module/i, category: "build" },

  // CORS
  { pattern: /CORS.*blocked/i, category: "cors" },
  { pattern: /Access-Control-Allow-Origin/i, category: "cors" },
]

const ERROR_LOG_PATH = join(homedir(), ".opencode-ui-error-log.json")
const MAX_ENTRIES = 50

interface UIErrorRecord {
  timestamp: string
  command: string
  category: string
  pattern: string
  snippet: string
  cwd?: string
}

function loadLog(): UIErrorRecord[] {
  try {
    return JSON.parse(readFileSync(ERROR_LOG_PATH, "utf-8"))
  } catch {
    return []
  }
}

function persist(log: UIErrorRecord[]) {
  try {
    writeFileSync(ERROR_LOG_PATH, JSON.stringify(log.slice(-MAX_ENTRIES), null, 2))
  } catch { /* best-effort */ }
}

function isUIRelatedCommand(command: string): boolean {
  const uiCommands = [
    "npm run dev", "npm run build", "next dev", "next build", "next start",
    "npx next", "yarn dev", "yarn build", "pnpm dev", "pnpm build",
    "vitest", "jest", "cypress", "tsc", "eslint",
    "npm run test", "npm run e2e", "npm run typecheck",
  ]
  return uiCommands.some(cmd => command.includes(cmd))
}

function formatAge(ts: string): string {
  const diff = Date.now() - new Date(ts).getTime()
  if (diff < 60_000) return "just now"
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`
  return `${Math.floor(diff / 86_400_000)}d ago`
}

function printStartupSummary(log: UIErrorRecord[]) {
  if (log.length === 0) return

  const recent = log.slice(-10) // last 10 errors
  const byCategory = new Map<string, number>()
  for (const e of log) {
    byCategory.set(e.category, (byCategory.get(e.category) || 0) + 1)
  }

  console.log("")
  console.log("  \x1b[33m⚠ UI Errors\x1b[0m")
  console.log("  ─────────────────────────────────────")

  // Category summary
  const cats = [...byCategory.entries()].sort((a, b) => b[1] - a[1])
  for (const [cat, count] of cats.slice(0, 5)) {
    console.log(`  \x1b[90m${cat.padEnd(14)}\x1b[0m ${count}`)
  }

  // Latest error
  const latest = recent[recent.length - 1]
  console.log("")
  console.log(`  \x1b[31mlatest\x1b[0m  ${latest.pattern}`)
  console.log(`  \x1b[90m${formatAge(latest.timestamp)} · ${latest.command.slice(0, 60)}\x1b[0m`)
  console.log("")
}

export default (async () => {
  // On startup: display recent errors
  const log = loadLog()
  if (log.length > 0) {
    printStartupSummary(log)
  }

  return {
    "tool.execute.after": async (input: any, output: any) => {
      const toolName = input?.tool ?? ""
      if (toolName !== "bash") return

      const command = input?.input?.command ?? ""
      if (!command) return

      const combined = [output?.stdout ?? "", output?.stderr ?? ""].join("\n")
      const exitCode = output?.exit_code ?? output?.exitCode
      const cwd = input?.input?.workdir ?? input?.cwd

      // Only scan UI-related commands, or any command that produced an error
      const hasExitError = exitCode != null && exitCode !== 0
      if (!isUIRelatedCommand(command) && !hasExitError) return

      const log = loadLog()

      for (const { pattern, category } of UI_ERROR_PATTERNS) {
        const match = combined.match(pattern)
        if (match) {
          const snippet = combined.slice(
            Math.max(0, (match.index ?? 0) - 60),
            (match.index ?? 0) + 250,
          )
          log.push({
            timestamp: new Date().toISOString(),
            command: command.slice(0, 150),
            category,
            pattern: match[0],
            snippet,
            cwd,
          })
          persist(log)
          // Toast notification
          console.log(`\n  \x1b[31m⚠ UI error captured:\x1b[0m ${category} — ${match[0]}`)
          console.log(`  \x1b[90mRun "opencode ui-fix" to diagnose\x1b[0m\n`)
          return // one error per command to avoid spam
        }
      }

      // Non-zero exit on UI command without a matched pattern — still log it
      if (hasExitError && isUIRelatedCommand(command)) {
        log.push({
          timestamp: new Date().toISOString(),
          command: command.slice(0, 150),
          category: "unknown",
          pattern: `exit code ${exitCode}`,
          snippet: combined.slice(0, 400),
          cwd,
        })
        persist(log)
        console.log(`\n  \x1b[31m⚠ UI error captured:\x1b[0m exit code ${exitCode}`)
        console.log(`  \x1b[90mRun "opencode ui-fix" to diagnose\x1b[0m\n`)
      }
    },
  }
}) satisfies Plugin
