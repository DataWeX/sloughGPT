import { NextResponse } from 'next/server'

export async function POST() {
  // Sync is a no-op for now — notes are read directly
  return NextResponse.json({ added: 0, updated: 0, total: 0 })
}
