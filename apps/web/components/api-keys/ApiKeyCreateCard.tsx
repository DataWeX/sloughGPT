'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent, Button, Input } from '@sloughgpt/strui'

interface ApiKeyCreateCardProps {
  onCreate?: (name: string) => Promise<void>
}

export function ApiKeyCreateCard({ onCreate }: ApiKeyCreateCardProps) {
  const [name, setName] = useState('')
  const [creating, setCreating] = useState(false)

  const handleCreate = async () => {
    if (!name.trim() || !onCreate) return
    setCreating(true)
    try {
      await onCreate(name.trim())
      setName('')
    } finally { setCreating(false) }
  }

  return (
    <Card data-testid="api-key-create">
      <CardHeader>
        <CardTitle className="text-base">Create API Key</CardTitle>
        <CardDescription>Create a new key for programmatic access.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex gap-2">
          <Input
            value={name}
            onChange={e => setName(e.target.value)}
            placeholder="Key name"
            maxLength={100}
            onKeyDown={e => e.key === 'Enter' && handleCreate()}
            data-testid="api-key-name"
          />
          <Button onClick={handleCreate} disabled={!name.trim() || creating}>
            {creating ? 'Creating...' : 'Create'}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
