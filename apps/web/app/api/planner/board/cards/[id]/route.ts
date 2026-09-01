import { NextRequest, NextResponse } from 'next/server'
import { updateCard, deleteCard } from '../../../helpers'

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params
    const data = await request.json()
    const card = updateCard(id, data)
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
    const success = deleteCard(id)
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
