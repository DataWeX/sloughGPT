import { NextResponse } from 'next/server'
import { getAllTags } from '../helpers'

export async function GET(request: Request) {
  try {
    const workspaceId = request.headers.get('x-workspace-id') || undefined
    const tags = getAllTags(workspaceId)
    return NextResponse.json({ tags })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to read tags' },
      { status: 500 },
    )
  }
}
