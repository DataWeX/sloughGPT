import { NextRequest, NextResponse } from 'next/server'
import { createCard } from '../../helpers'

export async function POST(request: NextRequest) {
  try {
    const data = await request.json()
    if (!data.title) {
      return NextResponse.json(
        { error: 'title is required' },
        { status: 400 },
      )
    }
    const card = createCard(data)
    return NextResponse.json({ card }, { status: 201 })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to create card' },
      { status: 500 },
    )
  }
}
