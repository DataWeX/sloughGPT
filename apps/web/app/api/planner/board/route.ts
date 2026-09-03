import { NextResponse } from 'next/server'
import { readBoard } from '../helpers'

export async function GET() {
  try {
    const board = readBoard()
    return NextResponse.json({ board })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to read board' },
      { status: 500 },
    )
  }
}
