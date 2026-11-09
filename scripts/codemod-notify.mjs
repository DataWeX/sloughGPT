#!/usr/bin/env node
/**
 * codemod-notify — migrate apps/web from lib/toast-store + lib/banner-store
 * to the notify library (lib/notify).
 *
 * Transformations (per file, in order):
 *   1. call sites      `addToast(a, b, …)`        → `notify.toast(a, { type: b, … })`
 *                      `store.addToast(` / `useToastStore.getState().addToast(`
 *                      `useBannerStore.getState().showBanner({…})` → `notify.banner({…})`
 *   2. dead bindings   `const addToast = useToastStore(s => s.addToast)` → removed
 *   3. imports         both store modules merged into `@/lib/notify`
 *   4. test mocks      `vi.mock('@/lib/toast-store', f)` → `vi.mock('@/lib/notify', f′)`
 *                      with `notify: { toast: <existing mock>, banner: vi.fn() }`
 *                      injected (+ useBannerStore stub when the test imports it)
 *
 * Run: node scripts/codemod-notify.mjs [--root apps/web]
 * Excluded: lib/notify/**, the store files themselves, node_modules, dist.
 */

import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { join, relative } from 'node:path'

const rootIdx = process.argv.indexOf('--root')
const ROOT = rootIdx !== -1 ? process.argv[rootIdx + 1] ?? 'apps/web' : 'apps/web'
const ABS_ROOT = join(process.cwd(), ROOT)

const EXCLUDE_DIRS = new Set(['node_modules', 'dist', 'dist-vite', '.git', 'notify'])
const SELF = /lib\/(toast-store|banner-store)/

let changed = 0
const notes = []

/* ── helpers ──────────────────────────────────────────────────── */

/** Split the top-level args of a call whose text starts at `openIdx` (the '('). */
function splitArgs(text, openIdx) {
  const args = []
  let depth = 0
  let current = ''
  let quote = null // ', ", `
  let interpDepth = 0 // brace depth inside a `${ … }` interpolation
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

const OPT_NAMES = ['type', 'verbose', 'onUndo', 'key']

/** Rewrite one addToast(...) call occurrence. Returns { text, endIdx } relative output. */
function rewriteToastCall(text, callStart /* index of 'addToast' */) {
  const openIdx = text.indexOf('(', callStart)
  const { args, endIdx } = splitArgs(text, openIdx)
  if (args.length === 0) return { output: 'notify.toast()', endIdx }
  if (args.length === 1) return { output: `notify.toast(${args[0]})`, endIdx }
  if (args.length > 5) throw new Error(`addToast with ${args.length} args unsupported`)
  const opts = []
  for (let i = 1; i < args.length; i++) {
    if (args[i] === 'undefined') continue // positional gap — omit
    const name = OPT_NAMES[i - 1]
    // shorthand when the arg is literally the same identifier (type: type → type)
    opts.push(args[i] === name ? name : `${name}: ${args[i]}`)
  }
  const second = opts.length ? `, { ${opts.join(', ')} }` : ''
  return { output: `notify.toast(${args[0]}${second})`, endIdx }
}

/**
 * `useToastStore . getState() . addToast( … )` chains (whitespace/newline
 * tolerant): rewrite the WHOLE chain including its args — replacing only the
 * callee would leave positional args behind.
 */
function rewriteStateChains(src) {
  const re = /useToastStore\s*\.\s*getState\s*\(\s*\)\s*\.\s*addToast\s*\(/g
  let out = ''
  let i = 0
  let m
  while ((m = re.exec(src)) !== null) {
    const addIdx = m.index + m[0].lastIndexOf('addToast')
    const { output, endIdx } = rewriteToastCall(src, addIdx)
    out += src.slice(i, m.index) + output // drop the ENTIRE chain prefix
    i = endIdx + 1
    re.lastIndex = i
  }
  return out + src.slice(i)
}

/** Replace every `[…].addToast(` / bare `addToast(` call with notify.toast form. */
function rewriteAllToastCalls(src) {
  // Known chains, handled exactly (whitespace/newline tolerant):
  src = rewriteStateChains(src)
  let out = ''
  let i = 0
  let rewrote = 0
  const warnings = []
  while (i < src.length) {
    const idx = src.indexOf('addToast(', i)
    if (idx === -1) {
      out += src.slice(i)
      break
    }
    const wordBefore = idx > 0 && /[\w$]/.test(src[idx - 1])
    const isCall = src[idx + 'addToast'.length] === '('
    // Optional single-level member prefix `ident.addToast(` (e.g. `store.addToast(`):
    // the whole `ident.` is consumed into the rewrite.
    let prefixStart = idx
    if (!wordBefore && idx > 0 && src[idx - 1] === '.') {
      let k = idx - 2
      while (k >= 0 && /[\w$]/.test(src[k])) k--
      const ident = src.slice(k + 1, idx - 1)
      const before = k >= 0 ? src[k] : ''
      if (ident && !/[\w$.)]/.test(before)) {
        prefixStart = k + 1
      } else {
        // Unsupported deeper chain — leave untouched, surface it loudly.
        warnings.push(`unsupported member chain: …${src.slice(Math.max(0, idx - 40), idx + 20).replace(/\s+/g, ' ')}`)
        out += src.slice(i, idx + 'addToast'.length)
        i = idx + 'addToast'.length
        continue
      }
    }
    if (!wordBefore && isCall) {
      const { output, endIdx } = rewriteToastCall(src, idx)
      out += src.slice(i, prefixStart) // everything BEFORE the call (or `ident.`) survives
      out += output
      rewrote++
      i = endIdx + 1
      continue
    }
    out += src.slice(i, idx + 'addToast'.length)
    i = idx + 'addToast'.length
  }
  return { out, rewrote, warnings }
}

/**
 * `const addToast = useToastStore(s => s.addToast)` bindings: if the name is
 * still referenced AFTER removal (threaded into hooks/props), replace the
 * binding with a typed adapter over notify.toast; otherwise drop it entirely.
 */
function transformBindings(src) {
  const lines = src.split('\n')
  const bindingRe =
    /^(\s*)const\s+(\w+)\s*=\s*useToastStore\([\s\S]*?\.addToast[\s\S]*?\)\s*(?:as\s+[^;]+)?;?\s*$/
  const hits = []
  lines.forEach((line, i) => {
    const m = line.match(bindingRe)
    if (m) hits.push({ line: i, indent: m[1], name: m[2] })
  })
  if (hits.length === 0) return { out: src, removed: 0, adapters: 0 }

  const hitLines = new Set(hits.map((h) => h.line))
  const without = lines.filter((_, i) => !hitLines.has(i)).join('\n')

  let removed = 0
  let adapters = 0
  const kept = []
  lines.forEach((line, i) => {
    const hit = hits.find((h) => h.line === i)
    if (hit) {
      const stillUsed = new RegExp(`\\b${hit.name}\\b`).test(without)
      if (stillUsed) {
        kept.push(
          `${hit.indent}const ${hit.name} = (message: string, type?: string) =>` +
            ` notify.toast(message, { type: type as 'success' | 'error' | 'info' })`,
        )
        adapters++
      } else {
        removed++
      }
      return
    }
    kept.push(line)
  })
  return { out: kept.join('\n'), removed, adapters }
}

/**
 * Extract the value expression assigned to `addToast:` inside a mock factory.
 *
 * The key also appears in TYPE ANNOTATIONS inside the same factory
 * (`(selector: (s: { addToast: (...a: unknown[]) => void }) => unknown)`), so
 * every occurrence is tried and type-shaped candidates are skipped:
 * `typeof X` or `(params) => RetType` without a `{ … }` body.
 */
function isTypeShaped(v) {
  if (v.startsWith('typeof ')) return true
  if (v.startsWith('(')) {
    const arrow = v.indexOf('=>')
    if (arrow !== -1) return !v.slice(arrow + 2).trimStart().startsWith('{')
  }
  return false
}

function extractMockValue(factoryBody) {
  let from = 0
  for (;;) {
    const keyIdx = factoryBody.indexOf('addToast:', from)
    if (keyIdx === -1) return null
    from = keyIdx + 'addToast:'.length
    let i = from
    while (i < factoryBody.length && /\s/.test(factoryBody[i])) i++
    let depth = 0
    let quote = null
    let value = ''
    for (; i < factoryBody.length; i++) {
      const c = factoryBody[i]
      if (quote) {
        value += c
        if (c === '\\') {
          value += factoryBody[i + 1] ?? ''
          i++
          continue
        }
        if (c === quote) quote = null
        continue
      }
      if (c === "'" || c === '"' || c === '`') {
        quote = c
        value += c
        continue
      }
      if ('(['.includes(c)) depth++
      if (')]'.includes(c)) {
        if (depth === 0) break
        depth--
        value += c
        continue
      }
      if ((c === ',' || c === '}') && depth === 0) break
      value += c
    }
    value = value.trim()
    if (value && !isTypeShaped(value)) return value
  }
}

const BANNER_STUB = `useBannerStore: Object.assign(
        (sel: any) => sel({ banners: [], dismissBanner: vi.fn(), clearBanners: vi.fn() }),
        { getState: () => ({ banners: [], dismissBanner: vi.fn(), clearBanners: vi.fn() }) },
      ),`

/** Transform `vi.mock('@/lib/toast-store', factory)` → notify equivalent. */
function rewriteMock(src, needsBannerStub) {
  const marker = "vi.mock('@/lib/toast-store', () => ({"
  const idx = src.indexOf(marker)
  if (idx === -1) return { out: src, did: false }
  const bodyStart = idx + marker.length
  // find matching end `}))` of this factory (depth-aware from bodyStart)
  let depth = 1
  let end = -1
  for (let i = bodyStart; i < src.length; i++) {
    const c = src[i]
    if (c === '{') depth++
    else if (c === '}') {
      depth--
      if (depth === 0) {
        end = i
        break
      }
    }
  }
  if (end === -1) return { out: src, did: false }
  const body = src.slice(bodyStart, end)
  const mockValue = extractMockValue(body) ?? 'vi.fn()'
  const inject = `notify: { toast: ${mockValue}, banner: vi.fn() },\n      ${needsBannerStub ? BANNER_STUB + '\n      ' : ''}`
  let out = src.slice(0, idx) + "vi.mock('@/lib/notify', () => ({" + '\n      ' + inject + body + src.slice(end)
  return { out, did: true }
}

/** Rewrite import statements from the two store modules into one notify import. */
function rewriteImports(src) {
  const specifiers = [] // raw specifiers, `type ` markers preserved
  const importRe =
    /import\s*\{([^}]*)\}\s*from\s*'(?:@\/lib\/|\.{1,2}\/)(?:toast-store|banner-store)';?\n?/g
  let m
  let had = false
  while ((m = importRe.exec(src)) !== null) {
    had = true
    for (const raw of m[1].split(',')) {
      const name = raw.trim()
      if (name && !specifiers.includes(name)) specifiers.push(name)
    }
  }
  if (!had) return { out: src, did: false }
  let out = src.replace(importRe, '')
  const needsNotify = /\bnotify\.(?:toast|banner)\(/.test(out)
  const bare = specifiers.map((s) => s.replace(/^type\s+/, ''))
  if (needsNotify && !bare.includes('notify')) specifiers.unshift('notify')
  if (specifiers.length === 0) return { out, did: true }
  const stmt = `import { ${specifiers.join(', ')} } from '@/lib/notify'\n`
  // place at first line previously occupied by a store import (top of file after directives)
  // simple strategy: prepend after leading 'use client' directive if present
  if (out.startsWith("'use client'")) {
    const nl = out.indexOf('\n')
    out = out.slice(0, nl + 1) + stmt + out.slice(nl + 1)
  } else {
    out = stmt + out
  }
  return { out, did: true }
}

/* ── walker ───────────────────────────────────────────────────── */

function walk(dir, files = []) {
  for (const name of readdirSync(dir)) {
    if (EXCLUDE_DIRS.has(name)) continue
    const p = join(dir, name)
    const st = statSync(p)
    if (st.isDirectory()) walk(p, files)
    else if (/\.tsx?$/.test(name)) files.push(p)
  }
  return files
}

for (const file of walk(ABS_ROOT)) {
  const rel = relative(process.cwd(), file)
  if (SELF.test(file)) continue // the store files + their old tests: deleted separately
  let src = readFileSync(file, 'utf8')
  const original = src

  if (src.includes('toast-store') || src.includes('banner-store')) {
    try {
      // Dynamic import binding: `const { useToastStore } = await import('./toast-store')`
      // → the notify facade (the chain rewrite below turns its .getState().addToast
      // into notify.toast, so the binding must be `notify`).
      src = src.replace(
        /const\s*\{\s*useToastStore\s*\}\s*=\s*await import\((['"])(?:@\/lib\/|\.{1,2}\/)toast-store\1\)/g,
        "const { notify } = await import('@/lib/notify')",
      )
      const needsBannerStub = /import\s*\{[^}]*useBannerStore/.test(src)
      const mocked = rewriteMock(src, needsBannerStub)
      src = mocked.out

      const calls = rewriteAllToastCalls(src)
      src = calls.out

      src = src.replaceAll('useBannerStore.getState().showBanner(', 'notify.banner(')

      const bindings = transformBindings(src)
      src = bindings.out

      const imports = rewriteImports(src)
      src = imports.out

      if (src !== original) {
        writeFileSync(file, src)
        changed++
        const warn = calls.warnings?.length ? ` WARNINGS=${calls.warnings.length}` : ''
        notes.push(
          `${rel}  (calls:${calls.rewrote} removed:${bindings.removed} adapters:${bindings.adapters} mock:${mocked.did} imports:${imports.did})${warn}`,
        )
        for (const w of calls.warnings ?? []) notes.push(`    ! ${w}`)
      }
    } catch (e) {
      notes.push(`!! ${rel} SKIPPED: ${e.message}`)
    }
  }
}

console.log(`codemod-notify: ${changed} files changed`)
for (const n of notes) console.log('  ' + n)
