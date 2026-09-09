import { NextRequest, NextResponse } from 'next/server'
import { updateCard, deleteCard } from '../../../helpers'

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params
    const data = await request.json()
    const workspaceId = request.headers.get('x-workspace-id') || undefined
    const card = updateCard(id, data, workspaceId)
    if (!card) {
      return NextResponse.json(
        { error: 'Card not found' },
        { status: 404 },
      )
    }
    return NextResponse.json({ card })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to update card' },
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
    const success = deleteCard(id, workspaceId)
    if (!success) {
      return NextResponse.json(
        { error: 'Card not found' },
        { status: 404 },
      )
    }
    return NextResponse.json({ success: true })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to delete card' },
      { status: 500 },
    )
  }
}
