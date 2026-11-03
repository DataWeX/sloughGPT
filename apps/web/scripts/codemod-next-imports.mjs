#!/usr/bin/env node
// Keep framework-native imports. Vite aliases next/* to local shims, while
// Next must use its own router context and runtime implementations.

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

/** shim module path → framework-native specifier */
const SHIM_MAP = {
  '@/vite/next-compat/link': 'next/link',
  '@/vite/next-compat/navigation': 'next/navigation',
  '@/vite/next-compat/dynamic': 'next/dynamic',
  '@/vite/next-compat/web-vitals': 'next/web-vitals',
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

  for (const [spec, target] of Object.entries(SHIM_MAP)) {
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
    const n = (src.match(/@\/vite\/next-compat\/(link|navigation|dynamic|web-vitals)/g) || []).length
    rewritten += n
    if (!dry) writeFileSync(file, out, 'utf-8')
    console.log(`${dry ? 'would rewrite' : 'rewrote'}: ${file.slice(webRoot.length + 1)} (${n})`)
  }
  console.log(
    `${dry ? 'dry-run: ' : ''}files=${touched} specifier-hits≈${rewritten}`,
  )
}

main()
