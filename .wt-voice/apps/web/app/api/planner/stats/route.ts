import { NextResponse } from 'next/server'
import { getStats } from '../helpers'

export async function GET(request: Request) {
  try {
    const workspaceId = request.headers.get('x-workspace-id') || undefined
    const stats = getStats(workspaceId)
    return NextResponse.json({ stats })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to read stats' },
      { status: 500 },
    )
  }
}
