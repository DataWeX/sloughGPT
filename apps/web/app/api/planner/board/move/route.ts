import { NextRequest, NextResponse } from 'next/server'
import { moveCard } from '../../helpers'

export async function POST(request: NextRequest) {
  try {
    const { card_id, column } = await request.json()
    if (!card_id || !column) {
      return NextResponse.json(
        { error: 'card_id and column are required' },
        { status: 400 },
      )
    }
    const success = moveCard(card_id, column)
    if (!success) {
      return NextResponse.json(
        { error: 'Card not found' },
        { status: 404 },
      )
    }
    return NextResponse.json({ success: true })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to move card' },
      { status: 500 },
    )
  }
}
