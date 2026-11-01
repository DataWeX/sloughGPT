import type { NextRequest } from 'next/server'
import { NextResponse } from 'next/server'
import { readEvents, createEvent, getEventsByDate } from '../helpers'

export async function GET(request: Request) {
  try {
    const workspaceId = request.headers.get('x-workspace-id') || undefined
    const url = new URL(request.url)
    const date = url.searchParams.get('date')

    if (date) {
      const events = getEventsByDate(date, workspaceId)
      return NextResponse.json({ events })
    }

    const events = readEvents(workspaceId)
    return NextResponse.json({ events })
  } catch (_error) {
    return NextResponse.json({ error: 'Failed to read events' }, { status: 500 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const data = await request.json()
    if (!data.title || !data.date) {
      return NextResponse.json({ error: 'title and date are required' }, { status: 400 })
    }
    const workspaceId = request.headers.get('x-workspace-id') || undefined
    const event = createEvent(data, workspaceId)
    return NextResponse.json({ event }, { status: 201 })
  } catch (_error) {
    return NextResponse.json({ error: 'Failed to create event' }, { status: 500 })
  }
}
