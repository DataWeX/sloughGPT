import { describe, it, expect } from "vitest"

const UI_ERROR_PATTERNS = [
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
  { pattern: /NextRouter/i, category: "router" },
  { pattern: /next\/navigation/i, category: "router" },
  { pattern: /ChunkLoadError/i, category: "chunk-load" },
  { pattern: /Loading chunk.*failed/i, category: "chunk-load" },
  { pattern: /Dynamic server usage/i, category: "ssr" },
  { pattern: /Type .* is not assignable to type/i, category: "type-error" },
  { pattern: /Property .* does not exist on type/i, category: "type-error" },
  { pattern: /Argument of type .* is not assignable/i, category: "type-error" },
  { pattern: /TS\d{4,}/i, category: "type-error" },
  { pattern: /fetch failed/i, category: "network" },
  { pattern: /ECONNREFUSED/i, category: "network" },
  { pattern: /NetworkError/i, category: "network" },
  { pattern: /Failed to fetch/i, category: "network" },
  { pattern: /AbortError/i, category: "network" },
  { pattern: /Load failed/i, category: "network" },
  { pattern: /\b401\b.*Unauthorized/i, category: "auth" },
  { pattern: /\b403\b.*Forbidden/i, category: "auth" },
  { pattern: /\b404\b.*Not Found/i, category: "not-found" },
  { pattern: /\b500\b.*Internal/i, category: "server-error" },
  { pattern: /\b503\b.*Unavailable/i, category: "server-error" },
  { pattern: /npm ERR!/i, category: "build" },
  { pattern: /Build error occurred/i, category: "build" },
  { pattern: /Failed to compile/i, category: "build" },
  { pattern: /Module not found/i, category: "build" },
  { pattern: /Import.*could not be resolved/i, category: "build" },
  { pattern: /Cannot find module/i, category: "build" },
  { pattern: /CORS.*blocked/i, category: "cors" },
  { pattern: /Access-Control-Allow-Origin/i, category: "cors" },
]

function matchError(output: string): { category: string; pattern: string } | null {
  for (const { pattern, category } of UI_ERROR_PATTERNS) {
    const match = output.match(pattern)
    if (match) return { category, pattern: match[0] }
  }
  return null
}

describe("ui-error-watcher patterns", () => {
  it("matches hydration errors", () => {
    const r = matchError("Hydration failed: Text content did not match")
    expect(r?.category).toBe("hydration")
  })

  it("matches null-access errors", () => {
    const r = matchError("TypeError: Cannot read properties of undefined (reading 'map')")
    expect(r?.category).toBe("null-access")
  })

  it("matches type errors", () => {
    const r = matchError("Type 'string' is not assignable to type 'number'")
    expect(r?.category).toBe("type-error")
  })

  it("matches chunk load errors", () => {
    const r = matchError("ChunkLoadError: Loading chunk 3 failed")
    expect(r?.category).toBe("chunk-load")
  })

  it("matches network errors", () => {
    const r = matchError("TypeError: Failed to fetch")
    expect(r?.category).toBe("network")
  })

  it("matches build errors", () => {
    const r = matchError("npm ERR! code ELSPROCLEMS")
    expect(r?.category).toBe("build")
  })

  it("matches infinite loop errors", () => {
    const r = matchError("Error: Maximum update depth exceeded")
    expect(r?.category).toBe("infinite-loop")
  })

  it("matches memory leak errors", () => {
    const r = matchError("Warning: Can't perform a React state update on an unmounted component")
    expect(r?.category).toBe("memory-leak")
  })

  it("matches auth errors", () => {
    const r = matchError("401 Unauthorized")
    expect(r?.category).toBe("auth")
  })

  it("matches not-found errors", () => {
    const r = matchError("404 Not Found")
    expect(r?.category).toBe("not-found")
  })

  it("matches server errors", () => {
    const r = matchError("500 Internal Server Error")
    expect(r?.category).toBe("server-error")
  })

  it("matches CORS errors", () => {
    const r = matchError("CORS blocked: Access-Control-Allow-Origin")
    expect(r?.category).toBe("cors")
  })

  it("matches router errors", () => {
    const r = matchError("Error: NextRouter was not mounted")
    expect(r?.category).toBe("router")
  })

  it("matches module not found", () => {
    const r = matchError("Module not found: Can't resolve './missing'")
    expect(r?.category).toBe("build")
  })

  it("returns null for clean output", () => {
    expect(matchError("Compiled successfully")).toBeNull()
    expect(matchError("Ready on http://localhost:3000")).toBeNull()
    expect(matchError("✓ Compiled successfully")).toBeNull()
  })
})

describe("isUIRelatedCommand", () => {
  const uiCommands = [
    "npm run dev",
    "npm run build",
    "next dev",
    "next build",
    "npx next dev",
    "yarn dev",
    "pnpm build",
    "npx vitest run",
    "npx tsc --noEmit",
    "npm run test",
    "npm run e2e",
  ]

  const nonUICommands = [
    "git status",
    "ls -la",
    "python main.py",
    "docker compose up",
    "curl localhost:8000/health",
  ]

  for (const cmd of uiCommands) {
    it(`recognizes "${cmd}" as UI-related`, () => {
      const isUI = [
        "npm run dev", "npm run build", "next dev", "next build", "npx next",
        "yarn dev", "yarn build", "pnpm dev", "pnpm build",
        "vitest", "jest", "cypress", "tsc", "eslint",
        "npm run test", "npm run e2e", "npm run typecheck",
      ].some(c => cmd.includes(c))
      expect(isUI).toBe(true)
    })
  }

  for (const cmd of nonUICommands) {
    it(`recognizes "${cmd}" as non-UI`, () => {
      const isUI = [
        "npm run dev", "npm run build", "next dev", "next build", "npx next",
        "yarn dev", "yarn build", "pnpm dev", "pnpm build",
        "vitest", "jest", "cypress", "tsc", "eslint",
        "npm run test", "npm run e2e", "npm run typecheck",
      ].some(c => cmd.includes(c))
      expect(isUI).toBe(false)
    })
  }
})
