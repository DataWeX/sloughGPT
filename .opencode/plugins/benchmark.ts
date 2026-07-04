/**
 * Benchmark suite for the ui-error-watcher plugin.
 *
 * Measures:
 *   1. Speed — pattern matching latency (ns/op)
 *   2. Efficiency — memory allocation per match
 *   3. Portability — OS/path resolution
 *   4. Performance — throughput under load (matches/sec)
 */

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

function matchError(output) {
  for (const { pattern, category } of UI_ERROR_PATTERNS) {
    const match = output.match(pattern)
    if (match) return { category, pattern: match[0] }
  }
  return null
}

// ── Test Data ──────────────────────────────────────────────────────
const CLEAN_OUTPUTS = [
  "Compiled successfully",
  "Ready on http://localhost:3000",
  "✓ Compiled successfully",
  "PASS components/chat/ChatInput.test.tsx",
  "Tests: 12 passed, 12 total",
  "Build completed in 4.2s",
]

const ERROR_OUTPUTS = [
  "Hydration failed: Text content did not match",
  "TypeError: Cannot read properties of undefined (reading 'map')",
  "Type 'string' is not assignable to type 'number'",
  "ChunkLoadError: Loading chunk 3 failed",
  "TypeError: Failed to fetch",
  "npm ERR! code ELSPROCLEMS",
  "Error: Maximum update depth exceeded",
  "Warning: Can't perform a React state update on an unmounted component",
  "401 Unauthorized",
  "404 Not Found",
  "500 Internal Server Error",
  "CORS blocked: Access-Control-Allow-Origin",
  "Module not found: Can't resolve './missing'",
  "TS2322: Type 'string' is not assignable to type 'number'",
  "ECONNREFUSED 127.0.0.1:8000",
]

const LARGE_OUTPUT = `
npm run build

> sloughgpt@0.1.0 build
> next build

  Creating an optimized production build...
  Compiled successfully

  Linting and checking validity of types...
  Type 'string' is not assignable to type 'number'.
  Property 'data' does not exist on type 'Response'.

  Collecting static pages (0/11)...
  Error occurred prerendering page "/chat".
  Hydration failed: Text content did not match.
  ReactServerComponentsError: Cannot read properties of undefined (reading 'render').
  ChunkLoadError: Loading chunk 713 failed.
  Module not found: Can't resolve './components/Missing'.
  npm ERR! Build failed with errors.
  ECONNREFUSED 127.0.0.1:8000
  401 Unauthorized
  404 Not Found
  CORS blocked: Access-Control-Allow-Origin
  Warning: Can't perform a React state update on an unmounted component.
  Error: Maximum update depth exceeded.
  Failed to compile
  Build error occurred
  NetworkError: fetch failed
  AbortError: The operation was aborted
  Load failed
  TS2345: Argument of type 'string' is not assignable to parameter of type 'number'.
  500 Internal Server Error
  503 Service Unavailable
  Too many re-renders
  Dynamic server usage detected
  next/navigation router error
  Failed to compile
`

// ── Benchmark Helpers ──────────────────────────────────────────────
function benchmarkSingle(name, fn, iterations = 100_000) {
  // Warmup
  for (let i = 0; i < 1000; i++) fn()

  const start = performance.now()
  for (let i = 0; i < iterations; i++) fn()
  const elapsed = performance.now() - start
  const nsPerOp = ((elapsed * 1_000_000) / iterations).toFixed(1)
  const opsPerSec = Math.round((iterations / elapsed) * 1000)

  return { name, iterations, elapsedMs: +elapsed.toFixed(2), nsPerOp: +nsPerOp, opsPerSec }
}

function measureMemory(fn, iterations = 10_000) {
  globalThis.gc?.()
  const before = process.memoryUsage().heapUsed
  for (let i = 0; i < iterations; i++) fn()
  globalThis.gc?.()
  const after = process.memoryUsage().heapUsed
  return { bytesPerCall: Math.round((after - before) / iterations), totalBytes: after - before }
}

// ── Run Benchmarks ─────────────────────────────────────────────────
console.log("=" .repeat(70))
console.log("  UI Error Watcher — Performance Benchmark")
console.log("=".repeat(70))

// 1. Speed
console.log("\n▸ SPEED — Pattern matching latency")
console.log("-".repeat(70))

const cleanResult = benchmarkSingle("Clean output (no match)", () => {
  for (const out of CLEAN_OUTPUTS) matchError(out)
})
console.log(`  ${cleanResult.name}`)
console.log(`    ${cleanResult.nsPerOp} ns/op | ${cleanResult.opsPerSec.toLocaleString()} batch/sec | ${cleanResult.elapsedMs}ms total`)

const errorResult = benchmarkSingle("Error output (match)", () => {
  for (const out of ERROR_OUTPUTS) matchError(out)
})
console.log(`  ${errorResult.name}`)
console.log(`    ${errorResult.nsPerOp} ns/op | ${errorResult.opsPerSec.toLocaleString()} batch/sec | ${errorResult.elapsedMs}ms total`)

const largeResult = benchmarkSingle("Large output (mixed, 350+ lines)", () => {
  matchError(LARGE_OUTPUT)
})
console.log(`  ${largeResult.name}`)
console.log(`    ${largeResult.nsPerOp} ns/op | ${largeResult.opsPerSec.toLocaleString()} ops/sec | ${largeResult.elapsedMs}ms total`)

// 2. Efficiency
console.log("\n▸ EFFICIENCY — Memory allocation")
console.log("-".repeat(70))

const memClean = measureMemory(() => {
  for (const out of CLEAN_OUTPUTS) matchError(out)
})
console.log(`  Clean output: ${memClean.bytesPerCall} bytes/call`)

const memError = measureMemory(() => {
  for (const out of ERROR_OUTPUTS) matchError(out)
})
console.log(`  Error output: ${memError.bytesPerCall} bytes/call`)

const memLarge = measureMemory(() => {
  matchError(LARGE_OUTPUT)
})
console.log(`  Large output: ${memLarge.bytesPerCall} bytes/call`)

// 3. Portability
console.log("\n▸ PORTABILITY — Cross-platform paths")
console.log("-".repeat(70))

import { homedir } from "os"
import { join } from "path"

const logPath = join(homedir(), ".opencode-ui-error-log.json")
const isAbsolute = /^[A-Za-z]:\\|^\//.test(logPath)
const separator = logPath.includes("\\") ? "\\" : "/"
const platform = process.platform

console.log(`  Platform:        ${platform}`)
console.log(`  Home dir:        ${homedir()}`)
console.log(`  Log path:        ${logPath}`)
console.log(`  Absolute path:   ${isAbsolute ? "✅" : "❌"}`)
console.log(`  Path separator:  ${separator}`)
console.log(`  Portable:        ${["darwin", "linux", "win32"].includes(platform) ? "✅" : "⚠️  unknown OS"}`)

// 4. Performance under load
console.log("\n▸ PERFORMANCE — Throughput under load")
console.log("-".repeat(70))

const BATCH_SIZES = [1, 10, 50, 100]
for (const batchSize of BATCH_SIZES) {
  const outputs = Array.from({ length: batchSize }, (_, i) => ERROR_OUTPUTS[i % ERROR_OUTPUTS.length])
  const result = benchmarkSingle(`${batchSize} outputs/batch`, () => {
    for (const out of outputs) matchError(out)
  }, 50_000)
  console.log(`  Batch ${batchSize.toString().padStart(3)}: ${result.opsPerSec.toLocaleString().padStart(8)} batch/sec | ${result.nsPerOp} ns/op`)
}

// Summary
console.log("\n" + "=".repeat(70))
console.log("  SUMMARY")
console.log("=".repeat(70))
const score = {
  speed: cleanResult.nsPerOp < 50_000 ? "✅ PASS" : "⚠️  SLOW",
  memory: memClean.bytesPerCall < 5000 ? "✅ PASS" : "⚠️  HIGH",
  portability: ["darwin", "linux", "win32"].includes(platform) ? "✅ PASS" : "⚠️  CHECK",
}
console.log(`  Speed:       ${score.speed} (${cleanResult.nsPerOp} ns/op on clean output)`)
console.log(`  Memory:      ${score.memory} (${memClean.bytesPerCall} bytes/call on clean output)`)
console.log(`  Portability: ${score.portability} (${platform}, ${separator} separator)`)
console.log("=".repeat(70))
