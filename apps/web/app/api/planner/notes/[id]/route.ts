import { NextRequest, NextResponse } from 'next/server'
import { updateNote, deleteNote } from '../../helpers'

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params
    const data = await request.json()
    const note = updateNote(id, data)
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
    const success = deleteNote(id)
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
