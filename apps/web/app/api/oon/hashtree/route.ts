import { NextRequest, NextResponse } from 'next/server'
import { getHashTree, createHashTree, readHashTrees, writeHashTrees, addNoteToTree } from '../../planner/helpers'

// GET /api/oon/hashtree?cardId=xxx — get hash tree for a card
// GET /api/oon/hashtree — list all hash trees
export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url)
  const cardId = searchParams.get('cardId')

  if (cardId) {
    const tree = getHashTree(cardId)
    if (!tree) {
      return NextResponse.json({ error: 'Hash tree not found' }, { status: 404 })
    }
    return NextResponse.json(tree)
  }

  const trees = readHashTrees()
  const list = Array.from(trees.values())
  return NextResponse.json({ trees: list, count: list.length })
}

// POST /api/oon/hashtree — create hash tree for a card
export async function POST(req: NextRequest) {
  const body = await req.json()
  const { cardId, cardContent, tray, position } = body

  if (!cardId || !cardContent || !tray) {
    return NextResponse.json({ error: 'cardId, cardContent, and tray are required' }, { status: 400 })
  }

  const existing = getHashTree(cardId)
  if (existing) {
    return NextResponse.json(existing)
  }

  const tree = createHashTree(cardId, cardContent, tray, position || 0)
  const trees = readHashTrees()
  trees.set(cardId, tree)
  writeHashTrees(trees)

  return NextResponse.json(tree, { status: 201 })
}

// DELETE /api/oon/hashtree?cardId=xxx — delete hash tree
export async function DELETE(req: NextRequest) {
  const { searchParams } = new URL(req.url)
  const cardId = searchParams.get('cardId')

  if (!cardId) {
    return NextResponse.json({ error: 'cardId is required' }, { status: 400 })
  }

  const trees = readHashTrees()
  if (!trees.has(cardId)) {
    return NextResponse.json({ error: 'Hash tree not found' }, { status: 404 })
  }

  trees.delete(cardId)
  writeHashTrees(trees)

  return NextResponse.json({ deleted: true })
}
