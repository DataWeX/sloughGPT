import { NextResponse } from 'next/server'
import { getAllTags } from '../helpers'

export async function GET() {
  try {
    const tags = getAllTags()
    return NextResponse.json({ tags })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to read tags' },
      { status: 500 },
    )
  }
}
