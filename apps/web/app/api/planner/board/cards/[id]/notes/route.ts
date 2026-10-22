import type { NextRequest } from 'next/server'
import { NextResponse } from 'next/server'
import { addCardNote, updateCardNote, deleteCardNote } from '../../../../helpers'

// POST /api/planner/board/cards/[id]/notes
// Single endpoint, action-based — every action writes card.notes AND
// the hash tree (note_add/note_edit/note_delete + rehash) in one save.
//   { action: 'add',    text, author? }
//   { action: 'update', note_id, text }
//   { action: 'delete', note_id }
export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await params
    const data = await request.json()
    const workspaceId = request.headers.get('x-workspace-id') || undefined

    let card = null
    switch (data.action) {
      case 'add':
        if (!data.text?.trim()) {
          return NextResponse.json({ error: 'text is required' }, { status: 400 })
        }
        card = addCardNote(id, data.text, data.author || '', workspaceId)
        break
      case 'update':
        if (!data.note_id || !data.text?.trim()) {
          return NextResponse.json({ error: 'note_id and text are required' }, { status: 400 })
        }
        card = updateCardNote(id, data.note_id, data.text, workspaceId)
        break
      case 'delete':
        if (!data.note_id) {
          return NextResponse.json({ error: 'note_id is required' }, { status: 400 })
        }
        card = deleteCardNote(id, data.note_id, workspaceId)
        break
      default:
        return NextResponse.json({ error: 'Unknown action' }, { status: 400 })
    }

    if (!card) {
      return NextResponse.json({ error: 'Card not found' }, { status: 404 })
    }
    return NextResponse.json({ card })
  } catch {
    return NextResponse.json({ error: 'Failed to mutate card note' }, { status: 500 })
  }
}
