import type { Plugin } from "@opencode-ai/plugin"
import { homedir } from "os"
import { join, relative } from "path"
import { writeFileSync, existsSync, readFileSync } from "fs"

const REPO_ROOT = process.cwd()

interface AreaDoc {
  area: string
  docs: string[]
  paths: string[]
}

const AREAS: AreaDoc[] = [
  { area: "frontend", docs: ["docs/UI_INTEGRATION_README.md", "docs/API.md"], paths: ["apps/web/"] },
  { area: "backend", docs: ["docs/routers.md", "docs/API.md", "docs/DATA_STRUCTURE.md"], paths: ["apps/api/"] },
  { area: "core", docs: ["docs/DEVELOPER_GUIDE.md", "docs/AI_SOFTWARE_ENGINEERING.md", "docs/RAG_ARCHITECTURE.md"], paths: ["packages/core-py/domains/inference/", "packages/core-py/domains/feedback/", "packages/core-py/domains/context/", "packages/core-py/domains/core/"] },
  { area: "training", docs: ["docs/DEVELOPER_GUIDE.md"], paths: ["packages/core-py/domains/training/"] },
  { area: "sdk", docs: ["docs/API.md"], paths: ["packages/sdk-py/", "packages/sdk-ts/"] },
  { area: "infra", docs: ["docs/DEPLOYMENT.md", "docs/DEPLOYMENT_CHECKLIST.md"], paths: ["infra/", "docker-compose", "Dockerfile"] },
  { area: "config", docs: ["docs/ENVIRONMENT.md"], paths: ["config/"] },
  { area: "cli", docs: ["docs/integration/CLI_README.md"], paths: ["apps/cli/"] },
  { area: "testing", docs: ["docs/DEVELOPER_GUIDE.md"], paths: ["tests/", "apps/web/cypress/"] },
  { area: "docs", docs: [], paths: ["docs/"] },
]

interface GuardRecord {
  timestamp: string
  file: string
  area: string
  missingDocs: string[]
  action: "warn" | "blocked"
}

const GUARD_LOG = join(homedir(), ".opencode-doc-guard-log.json")

let docsRead: Set<string> = new Set()

function persist(records: GuardRecord[]) {
  try {
    writeFileSync(GUARD_LOG, JSON.stringify(records.slice(-50), null, 2))
  } catch { /* best-effort */ }
}

function areaForFile(filePath: string): AreaDoc | null {
  const rel = relative(REPO_ROOT, filePath)
  for (const area of AREAS) {
    if (area.paths.some(p => rel.startsWith(p) || filePath.includes(p))) {
      return area
    }
  }
  return null
}

let guardHistory: GuardRecord[] = []

export default (async () => {
  if (existsSync(GUARD_LOG)) {
    try {
      const existing = JSON.parse(readFileSync(GUARD_LOG, "utf-8"))
      if (Array.isArray(existing)) guardHistory = existing
    } catch { /* ignore corrupt log */ }
  }

  return {
    "tool.execute.after": async (input: any, output: any) => {
      const tool = input?.tool ?? ""
      const filePath = input?.input?.filePath ?? input?.input?.path ?? ""

      if (tool === "read" && filePath) {
        docsRead.add(filePath)
        const area = areaForFile(filePath)
        if (area) {
          for (const doc of area.docs) {
            const docPath = join(REPO_ROOT, doc)
            if (filePath === docPath) {
              docsRead.add(doc)
            }
          }
        }
      }

      if ((tool === "edit" || tool === "write") && filePath) {
        const area = areaForFile(filePath)
        if (!area || area.docs.length === 0) return

        const missing = area.docs.filter(d => !docsRead.has(d))
        if (missing.length > 0) {
          const record: GuardRecord = {
            timestamp: new Date().toISOString(),
            file: relative(REPO_ROOT, filePath),
            area: area.area,
            missingDocs: missing,
            action: "warn",
          }
          guardHistory.push(record)
          persist(guardHistory)
        }
      }
    },
  }
}) satisfies Plugin
