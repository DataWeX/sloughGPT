import { NextResponse } from 'next/server'
import { readBoard } from '../helpers'

export async function GET(request: Request) {
  try {
    const workspaceId = request.headers.get('x-workspace-id') || undefined
    const board = readBoard(workspaceId)
    return NextResponse.json({ board })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to read board' },
      { status: 500 },
    )
  }
}
