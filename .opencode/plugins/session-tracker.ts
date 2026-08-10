import type { Plugin } from "@opencode-ai/plugin"
import { homedir } from "os"
import { join } from "path"
import { writeFileSync, readFileSync, existsSync } from "fs"

interface FileChange {
  file: string
  tool: "edit" | "write"
  timestamp: string
}

const SESSION_PATH = join(homedir(), ".opencode-session-changes.json")

function loadSession(): FileChange[] {
  if (!existsSync(SESSION_PATH)) return []
  try { return JSON.parse(readFileSync(SESSION_PATH, "utf-8")) } catch { return [] }
}

function saveSession(changes: FileChange[]) {
  try { writeFileSync(SESSION_PATH, JSON.stringify(changes, null, 2)) } catch {}
}

function formatAge(ts: string): string {
  const diff = Date.now() - new Date(ts).getTime()
  if (diff < 60_000) return "just now"
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`
  return `${Math.floor(diff / 3_600_000)}h ago`
}

export default (async () => {
  // On startup: show summary of uncommitted changes
  const changes = loadSession()
  if (changes.length > 0) {
    const files = [...new Set(changes.map(c => c.file))]
    const edits = changes.filter(c => c.tool === "edit").length
    const writes = changes.filter(c => c.tool === "write").length
    const age = formatAge(changes[0].timestamp)
    console.log(`\n  \x1b[36m📝 Session: ${files.length} files changed (${edits} edits, ${writes} writes) since ${age}\x1b[0m\n`)
  }

  return {
    "tool.execute.after": async (input: any, _output: any) => {
      const tool = input?.tool ?? ""
      if (tool !== "edit" && tool !== "write") return

      const filePath = input?.input?.filePath ?? input?.input?.path ?? ""
      if (!filePath) return

      const changes = loadSession()
      changes.push({
        file: filePath,
        tool: tool as "edit" | "write",
        timestamp: new Date().toISOString(),
      })

      // Keep last 200 changes
      if (changes.length > 200) changes.splice(0, changes.length - 200)
      saveSession(changes)
    },
  }
}) satisfies Plugin
