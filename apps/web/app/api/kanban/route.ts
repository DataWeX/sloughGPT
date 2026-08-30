import { NextResponse } from 'next/server'
import { readFileSync } from 'fs'
import { join } from 'path'

export async function GET() {
  try {
    const boardPath = join(process.cwd(), '..', '..', '.kanban', 'board.json')
    const raw = readFileSync(boardPath, 'utf-8')
    const board = JSON.parse(raw)
    return NextResponse.json(board)
  } catch (e) {
    return NextResponse.json({ name: 'board', columns: [], cards: [] })
  }
}
