/**
 * Shared helpers for reading/writing planner data.
 *
 * Board:   <repo>/.kanban/board.json
 * Notes:   ~/.config/dev-notes/*.md  (YAML frontmatter + markdown body)
 *
 * These match the data format used by packages/planner/.
 */

import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync } from 'fs'
import { join, basename } from 'path'
import { homedir } from 'os'

// ── Paths ──────────────────────────────────────────────────────────────

const REPO_ROOT = process.cwd()

export function boardPath(): string {
  return join(REPO_ROOT, '.kanban', 'board.json')
}

export function notesDir(): string {
  const dir = join(homedir(), '.config', 'dev-notes')
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  return dir
}

// ── Board ──────────────────────────────────────────────────────────────

export interface BoardColumn {
  name: string
  wip_limit: number
  order: number
}

export interface BoardCard {
  id: string
  title: string
  description: string
  column: string
  priority: 'low' | 'medium' | 'high' | 'critical'
  tags: string[]
  created_at: string
  updated_at: string
  due_date: string
  assignee: string
  notes: { id: string; text: string; author: string }[]
}

export interface Board {
  name: string
  columns: BoardColumn[]
  cards: BoardCard[]
}

export function readBoard(): Board {
  try {
    const raw = readFileSync(boardPath(), 'utf-8')
    return JSON.parse(raw)
  } catch {
    return { name: 'board', columns: [], cards: [] }
  }
}

export function writeBoard(board: Board): void {
  const dir = join(REPO_ROOT, '.kanban')
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  writeFileSync(boardPath(), JSON.stringify(board, null, 2))
}

// ── Notes ──────────────────────────────────────────────────────────────

export interface Note {
  id: string
  title: string
  created_at: string
  updated_at: string
  tags: string[]
  status: string
  sprint: string
  gh: string
  body: string
}

function parseNoteFile(filename: string, content: string): Note {
  const id = filename.replace(/\.md$/, '')
  const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/)
  if (!match) {
    return { id, title: id, created_at: '', updated_at: '', tags: [], status: 'open', sprint: '', gh: '', body: content.trim() }
  }
  const [, yamlBody, markdownBody] = match
  const meta: Record<string, string> = {}
  for (const line of yamlBody.split('\n')) {
    const colon = line.indexOf(':')
    if (colon > 0) {
      const key = line.slice(0, colon).trim()
      const val = line.slice(colon + 1).trim()
      meta[key] = val
    }
  }
  const tagsRaw = meta.tags || ''
  const tags = tagsRaw ? tagsRaw.split(',').map(t => t.trim()).filter(Boolean) : []
  return {
    id,
    title: meta.title || id,
    created_at: meta.created || '',
    updated_at: meta.updated || '',
    tags,
    status: meta.status || 'open',
    sprint: meta.sprint || '',
    gh: meta.gh || '',
    body: markdownBody.trim(),
  }
}

function noteToMarkdown(note: Note): string {
  const tags = note.tags.join(', ')
  const lines = [
    '---',
    `title: ${note.title}`,
    `created: ${note.created_at}`,
    `updated: ${note.updated_at}`,
    `tags: ${tags}`,
    `status: ${note.status}`,
  ]
  if (note.sprint) lines.push(`sprint: ${note.sprint}`)
  if (note.gh) lines.push(`gh: ${note.gh}`)
  lines.push('---', '', note.body)
  return lines.join('\n')
}

export function listNotes(): Note[] {
  const dir = notesDir()
  try {
    const files = readdirSync(dir).filter(f => f.endsWith('.md'))
    return files.map(f => {
      const content = readFileSync(join(dir, f), 'utf-8')
      return parseNoteFile(f, content)
    }).sort((a, b) => b.created_at.localeCompare(a.created_at))
  } catch {
    return []
  }
}

export function getNote(id: string): Note | null {
  const dir = notesDir()
  const filepath = join(dir, `${id}.md`)
  if (!existsSync(filepath)) return null
  const content = readFileSync(filepath, 'utf-8')
  return parseNoteFile(`${id}.md`, content)
}

export function createNote(data: { title: string; tags?: string[]; status?: string; body?: string; sprint?: string; gh?: string }): Note {
  const now = new Date().toISOString()
  const slug = data.title.toLowerCase().replace(/[^\w\s-]/g, '').replace(/[\s_]+/g, '-').replace(/-+/g, '-').slice(0, 60)
  const id = `${now.replace(/[-:T]/g, '').slice(0, 15)}_${slug}`
  const note: Note = {
    id,
    title: data.title,
    created_at: now,
    updated_at: now,
    tags: data.tags || [],
    status: data.status || 'open',
    sprint: data.sprint || '',
    gh: data.gh || '',
    body: data.body || '',
  }
  writeFileSync(join(notesDir(), `${id}.md`), noteToMarkdown(note))
  return note
}

export function updateNote(id: string, data: Partial<Pick<Note, 'title' | 'tags' | 'status' | 'body' | 'sprint' | 'gh'>>): Note | null {
  const existing = getNote(id)
  if (!existing) return null
  const now = new Date().toISOString()
  const updated: Note = {
    ...existing,
    ...Object.fromEntries(Object.entries(data).filter(([_, v]) => v !== undefined)),
    updated_at: now,
  }
  writeFileSync(join(notesDir(), `${id}.md`), noteToMarkdown(updated))
  return updated
}

export function deleteNote(id: string): boolean {
  const filepath = join(notesDir(), `${id}.md`)
  if (!existsSync(filepath)) return false
  const { unlinkSync } = require('fs')
  unlinkSync(filepath)
  return true
}
