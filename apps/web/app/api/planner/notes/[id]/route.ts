import { NextRequest, NextResponse } from 'next/server'
import { updateNote, deleteNote } from '../../helpers'

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params
    const data = await request.json()
    const workspaceId = request.headers.get('x-workspace-id') || undefined
    const note = updateNote(id, data, workspaceId)
    if (!note) {
      return NextResponse.json(
        { error: 'Note not found' },
        { status: 404 },
      )
    }
    return NextResponse.json({ note })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to update note' },
      { status: 500 },
    )
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params
    const workspaceId = request.headers.get('x-workspace-id') || undefined
    const success = deleteNote(id, workspaceId)
    if (!success) {
      return NextResponse.json(
        { error: 'Note not found' },
        { status: 404 },
      )
    }
    return NextResponse.json({ success: true })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to delete note' },
      { status: 500 },
    )
  }
}
