import { NextRequest, NextResponse } from 'next/server'
import {
  readBoard, writeBoard, createCard, updateCard, deleteCard, moveCard,
  readNotes, createNote, updateNote, deleteNote, getAllTags, getStats,
} from '../../planner/helpers'
import { readHashTrees, writeHashTrees, createHashTree, addNoteToTree, removeNoteFromTree, rehashTree } from '../../planner/helpers'

// POST /api/oon/card — single endpoint for all card operations
// Body: { action: "list" | "create" | "update" | "delete" | "move" | "board" | "tags" | "stats", ... }
export async function POST(req: NextRequest) {
  const body = await req.json()
  const { action, ...data } = body

  switch (action) {
    case 'board': {
      const board = readBoard()
      return NextResponse.json({ board })
    }

    case 'tags': {
      const tags = getAllTags()
      return NextResponse.json({ tags })
    }

    case 'stats': {
      const stats = getStats()
      return NextResponse.json({ stats })
    }

    case 'list': {
      const notes = readNotes()
      return NextResponse.json({ notes })
    }

    case 'create': {
      const { title, description, priority, tags, due_date, assignee, sprint, gh, column } = data
      if (!title) {
        return NextResponse.json({ error: 'title is required' }, { status: 400 })
      }
      const card = createCard({ title, description, priority, tags, due_date, assignee, sprint, gh, column })

      // Create hash tree
      const trees = readHashTrees()
      const tree = createHashTree(card.id, title, card.column, 0)
      trees.set(card.id, tree)
      writeHashTrees(trees)
      card.root_hash = tree.root.root

      return NextResponse.json({ card }, { status: 201 })
    }

    case 'update': {
      const { id, ...updates } = data
      if (!id) {
        return NextResponse.json({ error: 'id is required' }, { status: 400 })
      }
      const card = updateCard(id, updates)
      if (!card) {
        return NextResponse.json({ error: 'Card not found' }, { status: 404 })
      }
      return NextResponse.json({ card })
    }

    case 'delete': {
      const { id } = data
      if (!id) {
        return NextResponse.json({ error: 'id is required' }, { status: 400 })
      }
      const success = deleteCard(id)
      if (!success) {
        return NextResponse.json({ error: 'Card not found' }, { status: 404 })
      }
      // Clean up hash tree
      const trees = readHashTrees()
      trees.delete(id)
      writeHashTrees(trees)
      return NextResponse.json({ deleted: true })
    }

    case 'move': {
      const { id, column } = data
      if (!id || !column) {
        return NextResponse.json({ error: 'id and column are required' }, { status: 400 })
      }
      const success = moveCard(id, column)
      if (!success) {
        return NextResponse.json({ error: 'Card not found' }, { status: 404 })
      }
      return NextResponse.json({ moved: true })
    }

    case 'sync': {
      const notes = readNotes()
      const board = readBoard()
      const trees = readHashTrees()
      const existing = new Map(board.cards.map(c => [c.title, c]))
      let added = 0
      let moved = 0

      for (const note of notes) {
        const title = note.title || '(untitled)'
        const col = statusToColumn(note.status)
        const card = existing.get(title)

        if (!card) {
          const newCard = createCard({ title, column: col, tags: note.tags, description: note.body })
          const tree = createHashTree(newCard.id, title, col, 0)
          if (note.body) addNoteToTree(tree, note.id, note.body)
          trees.set(newCard.id, tree)
          newCard.root_hash = tree.root.root
          added++
          continue
        }

        if (card.column !== col) {
          moveCard(card.id, col)
          card.column = col
          moved++
        }

        const tree = trees.get(card.id)
        if (tree && note.body) {
          addNoteToTree(tree, note.id, note.body)
          trees.set(card.id, tree)
        }
      }

      writeHashTrees(trees)
      const total = readBoard().cards.length
      return NextResponse.json({ added, moved, total })
    }

    case 'create_note': {
      const { title, body, status, tags, sprint, gh, card_id } = data
      if (!title) {
        return NextResponse.json({ error: 'title is required' }, { status: 400 })
      }
      const note = createNote({ title, body, status, tags, sprint, gh })
      // Add to card's hash tree if card_id provided
      if (card_id && body) {
        const trees = readHashTrees()
        const tree = trees.get(card_id)
        if (tree) {
          addNoteToTree(tree, note.id, body)
          rehashTree(tree)
          writeHashTrees(trees)
        }
      }
      return NextResponse.json({ note }, { status: 201 })
    }

    case 'update_note': {
      const { id, card_id, ...updates } = data
      if (!id) {
        return NextResponse.json({ error: 'id is required' }, { status: 400 })
      }
      const note = updateNote(id, updates)
      if (!note) {
        return NextResponse.json({ error: 'Note not found' }, { status: 404 })
      }
      // Rehash card's tree if card_id provided
      if (card_id) {
        const trees = readHashTrees()
        const tree = trees.get(card_id)
        if (tree) {
          if (updates.body) addNoteToTree(tree, id, updates.body)
          rehashTree(tree)
          writeHashTrees(trees)
        }
      }
      return NextResponse.json({ note })
    }

    case 'delete_note': {
      const { id, card_id } = data
      if (!id) {
        return NextResponse.json({ error: 'id is required' }, { status: 400 })
      }
      const success = deleteNote(id)
      if (!success) {
        return NextResponse.json({ error: 'Note not found' }, { status: 404 })
      }
      // Remove from card's hash tree and rehash
      if (card_id) {
        const trees = readHashTrees()
        const tree = trees.get(card_id)
        if (tree) {
          removeNoteFromTree(tree, id)
          rehashTree(tree)
          writeHashTrees(trees)
        }
      }
      return NextResponse.json({ deleted: true })
    }

    default:
      return NextResponse.json({ error: `Unknown action: ${action}` }, { status: 400 })
  }
}

function statusToColumn(status: string): string {
  const map: Record<string, string> = {
    open: 'todo', todo: 'todo', wip: 'in_progress', in_progress: 'in_progress',
    review: 'review', done: 'done', blocked: 'review',
  }
  return map[(status || '').toLowerCase()] || 'todo'
}
