/**
 * codemod-notify-stabilize: hoist the local `addToast` adapter to module scope.
 *
 * The notify migration replaced `const addToast = useToastStore(s => s.addToast)`
 * (a stable zustand selector result) with a per-render arrow that calls
 * `notify.toast`. 71 of those files list `addToast` in a useCallback/useEffect
 * dependency array, so the new identity re-fires the effect on every render —
 * `setLoading(true)` loops forever and the test times out.
 *
 * The adapter only closes over the module-level `notify` import, so moving it to
 * module scope is behaviour-identical and gives it a stable identity. It stays a
 * local declaration (rather than a library export) so partial `vi.mock('@/lib/notify')`
 * factories in tests keep working untouched.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { execSync } from 'node:child_process'

const ADAPTER =
  "  const addToast = (message: string, type?: string) => notify.toast(message, { type: type as 'success' | 'error' | 'info' })"

const DECL = `// Module scope: \`addToast\` must keep a stable identity — it sits in
// useCallback/useEffect dependency arrays, and a per-render arrow would
// re-fire those effects on every render.
const addToast = (message: string, type?: string) =>
  notify.toast(message, { type: type as 'success' | 'error' | 'info' })
`

const files = execSync(
  `git diff --name-only -- 'apps/web/**/*.ts' 'apps/web/**/*.tsx'`,
  { encoding: 'utf8' },
)
  .split('\n')
  .filter((f) => f && !f.includes('.test.'))

let hoisted = 0
let skipped = []
let failed = []

/** Index just past the last top-level import statement. */
function afterLastImport(lines) {
  let last = -1
  for (let i = 0; i < lines.length; i++) {
    if (!lines[i].startsWith('import ')) continue
    let depth = 0
    let j = i
    for (; j < lines.length; j++) {
      for (const ch of lines[j]) {
        if ('({['.includes(ch)) depth++
        else if (')}]'.includes(ch)) depth--
      }
      if (depth <= 0) break
    }
    last = j
    i = j
  }
  return last + 1
}

for (const file of files) {
  let src
  try {
    src = readFileSync(file, 'utf8')
  } catch {
    continue
  }
  const lines = src.split('\n')
  const hits = lines.filter((l) => l === ADAPTER).length
  if (hits === 0) continue

  // The hoisted declaration closes over `notify`, which must be a top-level import.
  const hasTopImport = lines.some((l) => /^import .*['"]@\/lib\/notify['"]/.test(l))
  if (!hasTopImport) {
    skipped.push(`${file} (no top-level @/lib/notify import, ${hits} hits)`)
    continue
  }
  // Never hoist into a file that already declares it at module scope.
  if (lines.some((l) => l.startsWith('const addToast'))) {
    skipped.push(`${file} (already has module-scope addToast)`)
    continue
  }

  const kept = lines.filter((l) => l !== ADAPTER)
  const at = afterLastImport(kept)
  if (at <= 0) {
    skipped.push(`${file} (no imports found)`)
    continue
  }
  kept.splice(at, 0, DECL)

  writeFileSync(file, kept.join('\n'))
  hoisted++
  console.log(`  hoisted ${hits}x  ${file}`)
}

console.log(`codemod-notify-stabilize: ${hoisted} files hoisted, ${skipped.length} skipped`)
for (const s of skipped) console.log(`  SKIP ${s}`)
if (failed.length) {
  for (const f of failed) console.log(`  FAIL ${f}`)
  process.exitCode = 1
}
