'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, cn } from '@sloughgpt/strui'

interface Doc {
  _id: string
}

interface DocstoreEditorCardProps {
  doc: Doc | null
  editMode: boolean
  editContent: string
  onEditToggle: () => void
  onEditContentChange: (content: string) => void
  onSave: () => void
  onDelete: () => void
}

export function DocstoreEditorCard({
  doc,
  editMode,
  editContent,
  onEditToggle,
  onEditContentChange,
  onSave,
  onDelete,
}: DocstoreEditorCardProps) {
  if (!doc) {
    return (
      <Card data-testid="docstore-editor">
        <CardHeader>
          <CardTitle className="text-base">Document</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-muted-foreground">No document selected.</div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card data-testid="docstore-editor">
      <CardHeader>
        <CardTitle className="text-base">
          <code className="text-xs font-mono">{doc._id}</code>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex gap-1">
          {editMode ? (
            <>
              <Button size="sm" className="h-7 text-xs" onClick={onSave} data-testid="save-btn">
                Save
              </Button>
              <Button size="sm" variant="ghost" className="h-7 text-xs" onClick={onEditToggle} data-testid="cancel-btn">
                Cancel
              </Button>
            </>
          ) : (
            <>
              <Button size="sm" variant="ghost" className="h-7 text-xs" onClick={onEditToggle} data-testid="edit-btn">
                Edit
              </Button>
              <Button size="sm" variant="ghost" className="h-7 text-xs text-destructive" onClick={onDelete} data-testid="delete-btn">
                Delete
              </Button>
            </>
          )}
        </div>

        {editMode ? (
          <textarea
            className="w-full min-h-[200px] rounded border border-border bg-background p-2 text-xs font-mono resize-y"
            value={editContent}
            onChange={e => onEditContentChange(e.target.value)}
            data-testid="edit-textarea"
          />
        ) : (
          <pre className="text-xs font-mono whitespace-pre-wrap rounded border border-border bg-background p-2 overflow-auto max-h-[400px]">
            {editContent}
          </pre>
        )}
      </CardContent>
    </Card>
  )
}
