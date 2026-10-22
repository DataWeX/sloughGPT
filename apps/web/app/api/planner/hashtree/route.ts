import type { NextRequest } from 'next/server'
import { NextResponse } from 'next/server'
import { getHashTree, readHashTrees } from '../helpers'

// GET /api/planner/hashtree?cardId=xxx — hash tree for one card
// GET /api/planner/hashtree — all trees
export async function GET(req: NextRequest) {
  try {
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
  } catch {
    return NextResponse.json({ error: 'Failed to read hash trees' }, { status: 500 })
  }
}
