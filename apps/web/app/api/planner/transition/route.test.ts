import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import os from 'os'
import { join } from 'path'
import { mkdirSync, writeFileSync, readFileSync, existsSync, rmSync } from 'fs'
import type { NextRequest } from 'next/server'

const tmpDir = join(
  os.tmpdir(),
  `planner-transition-test-${Date.now()}-${Math.random().toString(36).slice(2)}`,
)
const tmpKanban = join(tmpDir, '.kanban')
const tmpBoard = join(tmpKanban, 'board.jsonl')

const originalCwd = process.cwd

const COLUMNS = [
  { name: 'todo', wip_limit: 5, order: 0 },
  { name: 'wip', wip_limit: 3, order: 1 },
  { name: 'review', wip_limit: 2, order: 2 },
  { name: 'done', wip_limit: 0, order: 3 },
]

function card(id: string, column: string, title = `Card ${id}`) {
  return {
    id,
    title,
    description: '',
    column,
    priority: 'medium',
    tags: [],
    due_date: '',
    assignee: '',
    sprint: '',
    gh: '',
    notes: [],
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
    root_hash: '',
  }
}

function writeFixture(cards: unknown[]) {
  const lines = [JSON.stringify({ schema: 'planner/1', name: 'Main', columns: COLUMNS })]
  for (const c of cards) lines.push(JSON.stringify(c))
  writeFileSync(tmpBoard, lines.join('\n') + '\n', 'utf-8')
}

function readCard(id: string): { column: string } {
  const lines = readFileSync(tmpBoard, 'utf-8').split('\n').filter(Boolean)
  for (const line of lines) {
    const obj = JSON.parse(line)
    if (obj.id === id) return obj
  }
  throw new Error(`card ${id} not found on disk`)
}

function makeRequest(body: unknown, headers: Record<string, string> = {}): NextRequest {
  return new Request('http://localhost/api/planner/transition', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify(body),
  }) as unknown as NextRequest
}

async function post(body: unknown) {
  const { POST } = await import('./route')
  const res = await POST(makeRequest(body))
  return { status: res.status, data: await res.json() }
}

beforeEach(() => {
  vi.resetModules()
  mkdirSync(tmpKanban, { recursive: true })
  process.cwd = () => tmpDir
})

afterEach(() => {
  process.cwd = originalCwd
  try {
    rmSync(tmpDir, { recursive: true, force: true })
  } catch {
    /* best-effort cleanup */
  }
})

describe('POST /api/planner/transition', () => {
  it('400 when card_id or to_state missing', async () => {
    writeFixture([card('a', 'todo')])
    const { status, data } = await post({ card_id: 'a' })
    expect(status).toBe(400)
    expect(data.ok).toBe(false)
  })

  it('404 for unknown card', async () => {
    writeFixture([card('a', 'todo')])
    const { status, data } = await post({ card_id: 'nope', to_state: 'wip' })
    expect(status).toBe(404)
    expect(data.event.type).toBe('transition_rejected')
    expect(data.event.metadata.reason).toBe('card_not_found')
  })

  it('409 for a column the board does not declare', async () => {
    writeFixture([card('a', 'todo')])
    const { status, data } = await post({ card_id: 'a', to_state: 'doing' })
    expect(status).toBe(409)
    expect(data.event.metadata.reason).toBe('unknown_state')
    expect(data.event.metadata.legal_targets).toEqual(['todo', 'wip', 'review', 'done'])
  })

  it('200 no-op when target equals current column', async () => {
    writeFixture([card('a', 'todo')])
    const { status, data } = await post({ card_id: 'a', to_state: 'todo' })
    expect(status).toBe(200)
    expect(data.ok).toBe(true)
    expect(data.event.metadata.noop).toBe(true)
    expect(readCard('a').column).toBe('todo')
  })

  it('200 lane-adjacent move persists to board.jsonl', async () => {
    writeFixture([card('a', 'todo')])
    const { status, data } = await post({ card_id: 'a', to_state: 'wip' })
    expect(status).toBe(200)
    expect(data.ok).toBe(true)
    expect(data.from_state).toBe('todo')
    expect(data.to_state).toBe('wip')
    expect(data.event.type).toBe('card_moved')
    expect(readCard('a').column).toBe('wip')
  })

  it('matches card id by prefix', async () => {
    writeFixture([card('20261022_abc123_fix-thing', 'todo')])
    const { status } = await post({ card_id: '20261022_abc', to_state: 'wip' })
    expect(status).toBe(200)
    expect(readCard('20261022_abc123_fix-thing').column).toBe('wip')
  })

  it('409 non-adjacent move with legal lane neighbours', async () => {
    writeFixture([card('a', 'todo')])
    const { status, data } = await post({ card_id: 'a', to_state: 'review' })
    expect(status).toBe(409)
    expect(data.event.metadata.reason).toBe('not_adjacent')
    expect(data.event.metadata.legal_targets).toEqual(['wip'])
    expect(readCard('a').column).toBe('todo')
  })

  it('409 done is terminal', async () => {
    writeFixture([card('c', 'done')])
    const { status, data } = await post({ card_id: 'c', to_state: 'review' })
    expect(status).toBe(409)
    expect(data.event.metadata.reason).toBe('terminal')
    expect(data.event.metadata.legal_targets).toEqual([])
    expect(data.error).toContain('terminal')
    expect(readCard('c').column).toBe('done')
  })

  it('409 wip_exceeded with counts when target is full', async () => {
    writeFixture([card('a', 'todo'), card('w1', 'wip'), card('w2', 'wip'), card('w3', 'wip')])
    const { status, data } = await post({ card_id: 'a', to_state: 'wip' })
    expect(status).toBe(409)
    expect(data.event.type).toBe('wip_exceeded')
    expect(data.wip_count).toBe(4)
    expect(data.wip_limit).toBe(3)
    expect(readCard('a').column).toBe('todo')
  })

  it('force=true bypasses an exceeded WIP limit', async () => {
    writeFixture([card('a', 'todo'), card('w1', 'wip'), card('w2', 'wip'), card('w3', 'wip')])
    const { status, data } = await post({ card_id: 'a', to_state: 'wip', force: true })
    expect(status).toBe(200)
    expect(data.ok).toBe(true)
    expect(data.event.metadata.forced).toBe(true)
    expect(readCard('a').column).toBe('wip')
  })

  it('0-limit column never blocks (done admits cards)', async () => {
    writeFixture([card('r', 'review'), card('d1', 'done'), card('d2', 'done')])
    const { status, data } = await post({ card_id: 'r', to_state: 'done' })
    expect(status).toBe(200)
    expect(data.wip_limit).toBe(0)
    expect(readCard('r').column).toBe('done')
  })

  it('rescues a card stuck in an undeclared column', async () => {
    const stuck = { ...card('x', 'legacy_col'), column: 'legacy_col' }
    writeFixture([stuck])
    const { status } = await post({ card_id: 'x', to_state: 'todo' })
    expect(status).toBe(200)
    expect(readCard('x').column).toBe('todo')
  })

  it('preserves unrelated rows (schema header, foreign lines)', async () => {
    writeFixture([card('a', 'todo')])
    const foreign = JSON.stringify({
      schema: 'planner/1',
      name: 'Main',
      columns: COLUMNS,
      slot_history: 'slot_history.jsonl',
    })
    const before = readFileSync(tmpBoard, 'utf-8')
    writeFileSync(tmpBoard, before + foreign + '\n', 'utf-8')

    await post({ card_id: 'a', to_state: 'wip' })

    const after = readFileSync(tmpBoard, 'utf-8')
    expect(after).toContain('slot_history')
    expect(after).toContain('"schema":"planner/1"')
    expect(existsSync(tmpBoard)).toBe(true)
  })
})
