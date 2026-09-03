import { NextResponse } from 'next/server'
import { getStats } from '../helpers'

export async function GET() {
  try {
    const stats = getStats()
    return NextResponse.json({ stats })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to read stats' },
      { status: 500 },
    )
  }
}
