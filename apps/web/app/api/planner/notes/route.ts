import { NextRequest, NextResponse } from 'next/server'
import { readNotes, createNote } from '../helpers'

export async function GET(request: Request) {
  try {
    const workspaceId = request.headers.get('x-workspace-id') || undefined
    const notes = readNotes(workspaceId)
    return NextResponse.json({ notes })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to read notes' },
      { status: 500 },
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const data = await request.json()
    if (!data.title) {
      return NextResponse.json(
        { error: 'title is required' },
        { status: 400 },
      )
    }
    const workspaceId = request.headers.get('x-workspace-id') || undefined
    const note = createNote(data, workspaceId)
    return NextResponse.json({ note }, { status: 201 })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to create note' },
      { status: 500 },
    )
  }
}
