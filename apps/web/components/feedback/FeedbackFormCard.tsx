'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'

interface FeedbackFormCardProps {
  onSubmit?: (feedback: { rating: 'thumbs_up' | 'thumbs_down'; comment?: string; category?: string }) => void
  submitting?: boolean
}

const CATEGORIES = ['quality', 'accuracy', 'helpfulness', 'speed', 'other']

export function FeedbackFormCard({ onSubmit, submitting = false }: FeedbackFormCardProps) {
  const [rating, setRating] = useState<'thumbs_up' | 'thumbs_down' | null>(null)
  const [comment, setComment] = useState('')
  const [category, setCategory] = useState<string>('')

  const handleSubmit = () => {
    if (!rating) return
    onSubmit?.({ rating, comment: comment.trim() || undefined, category: category || undefined })
    setRating(null)
    setComment('')
    setCategory('')
  }

  return (
    <Card data-testid="feedback-form">
      <CardHeader><CardTitle className="text-base">Submit Feedback</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <div>
          <p className="text-[10px] text-muted-foreground mb-1.5">Rating</p>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setRating('thumbs_up')}
              className={`flex items-center gap-1 px-3 py-1.5 rounded-md border text-xs transition-colors ${
                rating === 'thumbs_up'
                  ? 'bg-success/15 text-success border-success/30'
                  : 'text-muted-foreground border-border hover:text-foreground'
              }`}
              aria-label="Thumbs up"
            >
              👍 Good
            </button>
            <button
              type="button"
              onClick={() => setRating('thumbs_down')}
              className={`flex items-center gap-1 px-3 py-1.5 rounded-md border text-xs transition-colors ${
                rating === 'thumbs_down'
                  ? 'bg-destructive/15 text-destructive border-destructive/30'
                  : 'text-muted-foreground border-border hover:text-foreground'
              }`}
              aria-label="Thumbs down"
            >
              👎 Bad
            </button>
          </div>
        </div>

        <div>
          <p className="text-[10px] text-muted-foreground mb-1.5">Category (optional)</p>
          <div className="flex flex-wrap gap-1">
            {CATEGORIES.map(cat => (
              <button
                key={cat}
                type="button"
                onClick={() => setCategory(category === cat ? '' : cat)}
                className={`px-2 py-0.5 text-[10px] rounded border transition-colors ${
                  category === cat
                    ? 'bg-primary/15 text-primary border-primary/30'
                    : 'text-muted-foreground border-border hover:text-foreground'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        <div>
          <p className="text-[10px] text-muted-foreground mb-1.5">Comment (optional)</p>
          <textarea
            value={comment}
            onChange={e => setComment(e.target.value)}
            placeholder="Tell us more about your experience..."
            rows={2}
            className="w-full text-xs border border-border/40 rounded-md p-2 bg-background resize-none"
            aria-label="Feedback comment"
          />
        </div>

        <Button
          size="sm"
          className="h-7 text-[11px]"
          onClick={handleSubmit}
          disabled={!rating || submitting}
        >
          {submitting ? 'Submitting...' : 'Submit Feedback'}
        </Button>
      </CardContent>
    </Card>
  )
}
