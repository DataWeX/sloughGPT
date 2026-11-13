import type { Plugin } from "@opencode-ai/plugin"
import { execSync } from "child_process"

const PORT = 8000

function findPortProcesses(port: number): { pid: string; cmd: string }[] {
  try {
    const out = execSync(`lsof -ti :${port} 2>/dev/null`, { encoding: "utf-8", timeout: 3000 })
    const pids = out.trim().split("\n").filter(Boolean)
    return pids.map(pid => {
      let cmd = ""
      try {
        cmd = execSync(`ps -p ${pid} -o comm= 2>/dev/null`, { encoding: "utf-8", timeout: 2000 }).trim()
      } catch {}
      return { pid, cmd }
    })
  } catch {
    return []
  }
}

function killProcess(pid: string): boolean {
  try {
    execSync(`kill ${pid}`, { timeout: 3000 })
    return true
  } catch {
    try {
      execSync(`kill -9 ${pid}`, { timeout: 3000 })
      return true
    } catch {
      return false
    }
  }
}

export default (async () => {
  return {
    "tool.execute.after": async (input: any, output: any) => {
      if (input?.tool !== "bash") return

      const command = input?.input?.command ?? ""
      if (!command) return

      const combined = [output?.stdout ?? "", output?.stderr ?? ""].join("\n")

      // Detect port conflict
      const isPortError =
        /address already in use/i.test(combined) ||
        /port.*already/i.test(combined) ||
        /EADDRINUSE/i.test(combined)

      if (!isPortError) return

      const portMatch = combined.match(/(?:port|:)(\d{4,5})/)
      const port = portMatch ? parseInt(portMatch[1]) : PORT

      const procs = findPortProcesses(port)
      if (procs.length === 0) return

      const procList = procs.map(p => `│  \x1b[90mPID\x1b[0m ${p.pid}  \x1b[90m${p.cmd || "unknown"}\x1b[0m`).join("\n")
      const killCmd = `kill ${procs.map(p => p.pid).join(" ")}`
      const hint = [
        "",
        "┌─────────────────────────────────────────────────────────┐",
        `│ \x1b[33m⚠\x1b[0m \x1b[1mPort ${port} in use\x1b[0m by ${procs.length} process(es)${" ".repeat(Math.max(0, 26 - String(procs.length).length))}│`,
        "├─────────────────────────────────────────────────────────┤",
        procList,
        "├─────────────────────────────────────────────────────────┤",
        `│ \x1b[33m→ ${killCmd}\x1b[0m${" ".repeat(Math.max(0, 54 - killCmd.length))}│`,
        "└─────────────────────────────────────────────────────────┘",
        "",
      ].join("\n")

      if (typeof output?.stdout === "string") output.stdout += hint
      else if (typeof output?.output === "string") output.output += hint
    },
  }
}) satisfies Plugin
