#!/usr/bin/env node
/**
 * codemod-notify-assertions — align test assertions with the notify library's
 * call shape.
 *
 * The migration changed producers from `addToast(message, type, verbose, …)` to
 * `notify.toast(message, { type, verbose, … })`. Tests mock the module with
 * `vi.mock('@/lib/notify', () => ({ notify: { toast: spy, … } }))`, so the spy
 * now receives an OPTIONS OBJECT as its second argument — every assertion still
 * expecting the old positional form fails.
 *
 * Per file:
 *   1. read the `toast:` binding out of the notify mock → the spy's identifier
 *      (`mockAddToast`, `mocks.mockAddToast`, …). Anonymous `vi.fn()` bindings
 *      have no assertions against them and are skipped.
 *   2. rewrite `expect(<spy>).toHaveBeenCalledWith(<args>)` — a bare string
 *      2nd arg ('success' | 'error' | 'info') becomes `{ type: … }`, and any
 *      further positional args (verbose, onUndo, key) fold into that object.
 *
 * Deliberately NOT touched: `expect(addToast).toHaveBeenCalledWith(…)` where
 * `addToast` is a prop/hook parameter — those components still take the legacy
 * `(message, type?)` shape via a local adapter, so their positional assertions
 * remain correct.
 *
 * Run: node scripts/codemod-notify-assertions.mjs [--root apps/web]
 */

import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { join, relative } from 'node:path'

const rootIdx = process.argv.indexOf('--root')
const ROOT = rootIdx !== -1 ? process.argv[rootIdx + 1] ?? 'apps/web' : 'apps/web'
const ABS_ROOT = join(process.cwd(), ROOT)

const EXCLUDE_DIRS = new Set(['node_modules', 'dist', 'dist-vite', '.git', 'notify'])
const OPT_NAMES = ['type', 'verbose', 'onUndo', 'key']
const TOAST_TYPES = new Set(['success', 'error', 'info'])

let changed = 0
let rewritten = 0
const notes = []

/** Split the top-level args of a call whose '(' is at openIdx (excluded). */
function splitArgs(text, openIdx) {
  const args = []
  let depth = 0
  let current = ''
  let quote = null
  let interpDepth = 0
  for (let i = openIdx + 1; i < text.length; i++) {
    const c = text[i]
    if (quote) {
      if (c === '\\') {
        current += c + (text[i + 1] ?? '')
        i++
        continue
      }
      if (quote === '`') {
        if (c === '$' && text[i + 1] === '{') {
          current += '${'
          i++
          depth++
          interpDepth++
          continue
        }
        if (interpDepth > 0 && c === '{') {
          depth++
          interpDepth++
          current += c
          continue
        }
        if (interpDepth > 0 && c === '}') {
          depth--
          interpDepth--
          current += c
          continue
        }
        if (c === '`' && interpDepth === 0) {
          quote = null
          current += c
          continue
        }
        current += c
        continue
      }
      current += c
      if (c === quote) quote = null
      continue
    }
    if (c === "'" || c === '"' || c === '`') {
      quote = c
      current += c
      continue
    }
    if (c === '(' || c === '[' || c === '{') {
      depth++
      current += c
      continue
    }
    if (c === ')' || c === ']' || c === '}') {
      if (depth === 0) {
        if (current.trim()) args.push(current.trim())
        return { args, endIdx: i }
      }
      depth--
      current += c
      continue
    }
    if (c === ',' && depth === 0) {
      if (current.trim()) args.push(current.trim())
      current = ''
      continue
    }
    current += c
  }
  throw new Error('unbalanced call while splitting args')
}

/** Pull the toast spy identifier out of the notify mock factory. */
function findSpyIdentifier(src) {
  const marker = "vi.mock('@/lib/notify'"
  const at = src.indexOf(marker)
  if (at === -1) return null
  const bodyStart = src.indexOf('{', src.indexOf('=>', at))
  if (bodyStart === -1) return null
  // match the factory object literal
  let depth = 0
  let end = -1
  for (let i = bodyStart; i < src.length; i++) {
    if (src[i] === '{') depth++
    else if (src[i] === '}') {
      depth--
      if (depth === 0) {
        end = i
        break
      }
    }
  }
  if (end === -1) return null
  const body = src.slice(bodyStart, end)
  const m = body.match(/toast:\s*([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)/)
  if (!m) return null
  const ident = m[1]
  // anonymous / inline mocks assert nothing
  if (ident === 'vi.fn' || ident === 'fn') return null
  return ident
}

/** Rewrite one `expect(<spy>).toHaveBeenCalledWith(` occurrence. */
function rewriteAssertion(src, start /* index of 'expect(' */) {
  const callOpen = src.indexOf('(', start + 'expect'.length) // '(' of toHaveBeenCalledWith
  const openIdx = src.indexOf('(', src.indexOf('toHaveBeenCalledWith', start))
  if (openIdx === -1) return null
  const { args, endIdx } = splitArgs(src, openIdx)
  if (args.length < 2) return null
  const second = args[1]
  const lit = second.match(/^(['"])(success|error|info)\1$/)
  if (!lit || !TOAST_TYPES.has(lit[2])) return null

  const opts = [`${OPT_NAMES[0]}: ${second}`]
  for (let i = 2; i < args.length && i - 1 < OPT_NAMES.length; i++) {
    opts.push(`${OPT_NAMES[i - 1]}: ${args[i]}`)
  }
  const output = `toHaveBeenCalledWith(${args[0]}, { ${opts.join(', ')} })`
  void callOpen
  return { output, openIdx, endIdx }
}

/** Rewrite every assertion on `spy` in the file. */
function rewriteFile(src, spy) {
  let out = ''
  let i = 0
  let count = 0
  const needle = `expect(${spy}).toHaveBeenCalledWith(`
  for (;;) {
    const idx = src.indexOf(needle, i)
    if (idx === -1) {
      out += src.slice(i)
      break
    }
    const res = rewriteAssertion(src, idx)
    if (!res) {
      out += src.slice(i, idx + needle.length)
      i = idx + needle.length
      continue
    }
    out += src.slice(i, res.openIdx + 1) // keep `expect(spy).toHaveBeenCalledWith(`
    out += res.output.slice('toHaveBeenCalledWith('.length)
    i = res.endIdx + 1
    count++
  }
  return { out, count }
}

function walk(dir, files = []) {
  for (const name of readdirSync(dir)) {
    if (EXCLUDE_DIRS.has(name)) continue
    const p = join(dir, name)
    const st = statSync(p)
    if (st.isDirectory()) walk(p, files)
    else if (/\.test\.tsx?$/.test(name)) files.push(p)
  }
  return files
}

for (const file of walk(ABS_ROOT)) {
  const rel = relative(process.cwd(), file)
  let src
  try {
    src = readFileSync(file, 'utf8')
  } catch {
    continue
  }
  if (!src.includes("vi.mock('@/lib/notify'")) continue
  const spy = findSpyIdentifier(src)
  if (!spy) continue
  const { out, count } = rewriteFile(src, spy)
  if (count > 0 && out !== src) {
    writeFileSync(file, out)
    changed++
    rewritten += count
    notes.push(`${rel}  spy=${spy} assertions=${count}`)
  }
}

console.log(`codemod-notify-assertions: ${changed} files, ${rewritten} assertions`)
for (const n of notes) console.log('  ' + n)
