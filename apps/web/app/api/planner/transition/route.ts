import { type NextRequest, NextResponse } from 'next/server'
import { existsSync, readFileSync, writeFileSync } from 'fs'
import { readBoard, boardPath, type Board, type BoardCard } from '../helpers'

// POST /api/planner/transition — engine-validated move against the live board.
//
// The Python KanbanEngine (packages/planner/engine.py) speaks the researched
// 7-state vocabulary (inbox/ready/doing/...), but the live board's schema
// declares an ordered linear workflow (todo/wip/review/done). This route is
// the projection of the same engine contract onto the board's own columns:
//   - states = board columns, ordered by `order`
//   - legal = lane-adjacent (prev/next), last column (done) is terminal
//   - WIP enforced at transition time on the target (after-move count),
//     409 wip_exceeded unless force=true
//   - idempotent same-column moves are a no-op
// Response shape mirrors TransitionResult in lib/planner-client.ts.

interface TransitionEvent {
  type: string
  card_id: string
  from_state: string
  to_state: string
  actor: string
  timestamp: string
  metadata: Record<string, unknown>
}

function makeEvent(
  type: string,
  cardId: string,
  fromState: string,
  toState: string,
  actor: string,
  metadata: Record<string, unknown> = {},
): TransitionEvent {
  return {
    type,
    card_id: cardId,
    from_state: fromState,
    to_state: toState,
    actor,
    timestamp: new Date().toISOString(),
    metadata,
  }
}

function orderedColumnNames(columns: Board['columns']): string[] {
  return columns
    .slice()
    .sort((a, b) => a.order - b.order)
    .map((c) => c.name)
}

// Surgical single-card rewrite: only the moved card's line is touched —
// schema headers, foreign rows (slot_history/tray_roots), and every other
// card line keep their exact bytes. Unlike writeBoard, nothing is dropped.
function persistMove(
  board: Board,
  card: BoardCard,
  toState: string,
  workspaceId?: string,
): boolean {
  const bp = boardPath(workspaceId)
  if (!existsSync(bp)) return false

  const now = new Date().toISOString()
  const raw = readFileSync(bp, 'utf-8')
  const lines = raw.split('\n')
  let found = false

  const out = lines.map((line) => {
    if (!line.trim()) return line
    try {
      const obj = JSON.parse(line)
      if (obj.id === card.id && obj.title) {
        found = true
        obj.column = toState
        obj.updated_at = now
        return JSON.stringify(obj)
      }
    } catch {
      // foreign/unparseable line — keep verbatim
    }
    return line
  })

  if (!found) {
    // Card came from the app-planner feed overlay — append it to the
    // canonical board (same promotion semantics as writeBoard).
    const appended = JSON.stringify({ ...card, column: toState, updated_at: now })
    const lastIdx = out.length - 1
    if (out[lastIdx] === '') out.splice(lastIdx, 0, appended)
    else out.push(appended)
  }

  writeFileSync(bp, out.join('\n'), 'utf-8')
  return true
}

export async function POST(request: NextRequest) {
  const body = (await request.json().catch(() => null)) as {
    card_id?: string
    id?: string
    to_state?: string
    column?: string
    actor?: string
    force?: boolean
  } | null

  const cardId = (body?.card_id ?? body?.id ?? '').trim()
  const toState = (body?.to_state ?? body?.column ?? '').trim()
  const actor = typeof body?.actor === 'string' ? body.actor : ''
  const force = Boolean(body?.force)

  if (!cardId || !toState) {
    return NextResponse.json(
      { ok: false, error: 'card_id and to_state are required' },
      { status: 400 },
    )
  }

  const workspaceId = request.headers.get('x-workspace-id') || undefined
  const board = readBoard(workspaceId)
  const colNames = orderedColumnNames(board.columns)

  const card =
    board.cards.find((c) => c.id === cardId) ?? board.cards.find((c) => c.id.startsWith(cardId))

  const respond = (status: number, payload: Record<string, unknown>) =>
    NextResponse.json(payload, { status })

  if (!card) {
    return respond(404, {
      ok: false,
      card_id: cardId,
      from_state: '',
      to_state: toState,
      error: `Card '${cardId}' not found`,
      wip_count: 0,
      wip_limit: 0,
      event: makeEvent('transition_rejected', cardId, '', toState, actor, {
        reason: 'card_not_found',
      }),
    })
  }

  const fromState = card.column

  if (!colNames.includes(toState)) {
    return respond(409, {
      ok: false,
      card_id: card.id,
      from_state: fromState,
      to_state: toState,
      error: `Unknown state '${toState}'. Legal: [${colNames.join(', ')}]`,
      wip_count: 0,
      wip_limit: 0,
      event: makeEvent('transition_rejected', card.id, fromState, toState, actor, {
        reason: 'unknown_state',
        legal_targets: colNames,
      }),
    })
  }

  if (fromState === toState) {
    return respond(200, {
      ok: true,
      card_id: card.id,
      from_state: fromState,
      to_state: toState,
      error: '',
      wip_count: 0,
      wip_limit: 0,
      event: makeEvent('card_moved', card.id, fromState, toState, actor, { noop: true }),
    })
  }

  const fromIdx = colNames.indexOf(fromState)
  const terminal = colNames[colNames.length - 1]

  let legalTargets: string[]
  if (fromIdx === -1) {
    // Card sits in a column no longer declared on the board — allow rescue
    // into any declared column (WIP still applies).
    legalTargets = colNames.filter((n) => n !== fromState)
  } else if (fromIdx === colNames.length - 1) {
    // Last column (done) is terminal — no outbound moves.
    legalTargets = []
  } else {
    legalTargets = [...(fromIdx > 0 ? [colNames[fromIdx - 1]] : []), colNames[fromIdx + 1]]
  }

  if (!legalTargets.includes(toState)) {
    const terminalNote = fromState === terminal ? ' (done is terminal)' : ''
    return respond(409, {
      ok: false,
      card_id: card.id,
      from_state: fromState,
      to_state: toState,
      error: `Illegal transition: ${fromState} → ${toState}${terminalNote}. Legal: [${legalTargets.join(', ')}]`,
      wip_count: 0,
      wip_limit: 0,
      event: makeEvent('transition_rejected', card.id, fromState, toState, actor, {
        reason: fromState === terminal ? 'terminal' : 'not_adjacent',
        legal_targets: legalTargets,
      }),
    })
  }

  const targetCol = board.columns.find((c) => c.name === toState)
  const wipLimit = targetCol?.wip_limit ?? 0
  const wipCount = board.cards.filter((c) => c.column === toState && c.id !== card.id).length + 1

  if (!force && wipLimit > 0 && wipCount > wipLimit) {
    return respond(409, {
      ok: false,
      card_id: card.id,
      from_state: fromState,
      to_state: toState,
      error: `WIP limit exceeded for '${toState}': ${wipCount}/${wipLimit}`,
      wip_count: wipCount,
      wip_limit: wipLimit,
      event: makeEvent('wip_exceeded', card.id, fromState, toState, actor, {
        wip_count: wipCount,
        wip_limit: wipLimit,
      }),
    })
  }

  const moved = persistMove(board, card, toState, workspaceId)
  if (!moved) {
    return respond(500, {
      ok: false,
      card_id: card.id,
      from_state: fromState,
      to_state: toState,
      error: 'Failed to persist move',
      wip_count: 0,
      wip_limit: 0,
      event: makeEvent('transition_rejected', card.id, fromState, toState, actor, {
        reason: 'persist_failed',
      }),
    })
  }

  return respond(200, {
    ok: true,
    card_id: card.id,
    from_state: fromState,
    to_state: toState,
    error: '',
    wip_count: wipCount,
    wip_limit: wipLimit,
    event: makeEvent('card_moved', card.id, fromState, toState, actor, {
      forced: force && wipLimit > 0 && wipCount > wipLimit,
    }),
  })
}
