#!/usr/bin/env node
// Phase 3: rewrite client next/* imports (and vi.mock sites) onto local shims.
// In scope: next/link, next/navigation, next/dynamic, next/web-vitals
// Out of scope: next/server, next/font, next-auth, next/image, next (types)

import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { join, extname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { dirname } from 'node:path'

const webRoot = join(dirname(fileURLToPath(import.meta.url)), '..')

const SKIP_DIRS = new Set([
  'node_modules',
  '.next',
  '.next-dev',
  'dist-vite',
  'cypress',
  '.git',
  'public',
])

/** specifier → shim module path (alias-relative via @/) */
const SPECIFIER_MAP = {
  'next/link': '@/vite/next-compat/link',
  'next/navigation': '@/vite/next-compat/navigation',
  'next/dynamic': '@/vite/next-compat/dynamic',
  'next/web-vitals': '@/vite/next-compat/web-vitals',
}

const EXTS = new Set(['.ts', '.tsx', '.mts', '.cts', '.js', '.jsx'])

function walk(dir, out) {
  for (const entry of readdirSync(dir)) {
    if (entry.startsWith('.') && entry !== '.') {
      if (SKIP_DIRS.has(entry)) continue
    }
    if (SKIP_DIRS.has(entry)) continue
    const full = join(dir, entry)
    let st
    try {
      st = statSync(full)
    } catch {
      continue
    }
    if (st.isDirectory()) walk(full, out)
    else if (EXTS.has(extname(entry))) out.push(full)
  }
}

function transform(src) {
  let out = src
  let changed = false

  for (const [spec, target] of Object.entries(SPECIFIER_MAP)) {
    // import ... from 'spec' | from "spec"
    const importRe = new RegExp(
      `(from\\s*)(['"])${escapeRe(spec)}\\2`,
      'g',
    )
    out = out.replace(importRe, (_m, from, q) => {
      changed = true
      return `${from}${q}${target}${q}`
    })

    // side-effect import 'spec'
    const sideRe = new RegExp(`(import\\s*)(['"])${escapeRe(spec)}\\2`, 'g')
    out = sideRe[Symbol.replace](out, (_m, imp, q) => {
      changed = true
      return `${imp}${q}${target}${q}`
    })

    // vi.mock('spec') / vi.mock("spec")
    const mockRe = new RegExp(
      `(vi\\.mock\\(\\s*)(['"])${escapeRe(spec)}\\2`,
      'g',
    )
    out = out.replace(mockRe, (_m, head, q) => {
      changed = true
      return `${head}${q}${target}${q}`
    })

    // import('spec') dynamic
    const dynRe = new RegExp(
      `(import\\(\\s*)(['"])${escapeRe(spec)}\\2`,
      'g',
    )
    out = out.replace(dynRe, (_m, head, q) => {
      changed = true
      return `${head}${q}${target}${q}`
    })
  }

  return { out, changed }
}

function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function main() {
  const dry = process.argv.includes('--dry')
  const files = []
  walk(webRoot, files)

  let rewritten = 0
  let touched = 0
  for (const file of files) {
    // never rewrite the shims themselves
    if (file.includes(`${join('vite', 'next-compat')}`)) continue
    if (file.includes('scripts/codemod-next-imports')) continue

    const src = readFileSync(file, 'utf-8')
    const { out, changed } = transform(src)
    if (!changed) continue
    touched++
    const n = (src.match(/next\/(link|navigation|dynamic|web-vitals)/g) || []).length
    rewritten += n
    if (!dry) writeFileSync(file, out, 'utf-8')
    console.log(`${dry ? 'would rewrite' : 'rewrote'}: ${file.slice(webRoot.length + 1)} (${n})`)
  }
  console.log(
    `${dry ? 'dry-run: ' : ''}files=${touched} specifier-hits≈${rewritten}`,
  )
}

main()
