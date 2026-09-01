import { NextRequest, NextResponse } from 'next/server'
import { readNotes, createNote, updateNote, deleteNote } from '../helpers'

export async function GET() {
  try {
    const notes = readNotes()
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
    const note = createNote(data)
    return NextResponse.json({ note }, { status: 201 })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to create note' },
      { status: 500 },
    )
  }
}
