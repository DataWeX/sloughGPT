'use client'
export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { Card, CardHeader, CardTitle, CardContent, Badge, Button, Skeleton } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { apiGet } from '@/lib/http-client'
import { knowledgeController, type KnowledgeItem } from '@/lib/knowledge-controller'
import { useToastStore } from '@/lib/toast-store'
import { formatDate } from '@/lib/conversations-utils'
import {
  ArrowLeft, Brain, Tag, Clock, Star, ExternalLink, Edit,
  Trash2, BarChart3, Lightbulb
} from 'lucide-react'

const TOPIC_COLORS: Record<string, string> = {
  personal: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
  preferences: 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400',
  technical: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
  planning: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400',
  interests: 'bg-pink-100 text-pink-700 dark:bg-pink-900/30 dark:text-pink-400',
  food: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
  general: 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400',
}

export default function KnowledgeDetailPage() {
  const params = useParams()
  const router = useRouter()
  const itemId = params.id as string
  const addToast = useToastStore(s => s.addToast)

  const [item, setItem] = useState<KnowledgeItem | null>(null)
  const [loading, setLoading] = useState(true)
  const [editing, setEditing] = useState(false)
  const [editContent, setEditContent] = useState('')
  const [editTopic, setEditTopic] = useState('')
  const [saving, setSaving] = useState(false)

  const fetchItem = useCallback(async () => {
    setLoading(true)
    try {
      // Try direct endpoint first, fallback to list
      try {
        const data = await apiGet<KnowledgeItem>(`/knowledge/${itemId}`)
        setItem(data)
      } catch {
        const items = await knowledgeController.list()
        const found = items.find(i => i.id === itemId)
        if (found) {
          setItem(found)
        } else {
          addToast('Knowledge item not found', 'error')
        }
      }
    } catch {
      addToast('Failed to load knowledge item', 'error')
    } finally {
      setLoading(false)
    }
  }, [itemId, addToast])

  useEffect(() => { fetchItem() }, [fetchItem])

  const handleSave = async () => {
    if (!item) return
    setSaving(true)
    try {
      await knowledgeController.update(item.id, {
        content: editContent,
        topic: editTopic,
      })
      setItem({ ...item, content: editContent, topic: editTopic })
      setEditing(false)
      addToast('Updated', 'success')
    } catch {
      addToast('Update failed', 'error')
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async () => {
    if (!item) return
    try {
      await knowledgeController.delete(item.id)
      addToast('Deleted', 'success')
      router.push('/knowledge')
    } catch {
      addToast('Delete failed', 'error')
    }
  }

  const topicColor = item ? TOPIC_COLORS[item.topic] || TOPIC_COLORS.general : ''

  return (
    <PageContainer
      title={item ? 'Knowledge Item' : 'Loading...'}
      loading={loading}
      loadingCards={3}
      headerRight={
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => router.push('/knowledge')}>
            <ArrowLeft className="h-4 w-4 mr-1" /> Back
          </Button>
          <Button variant="outline" size="sm" onClick={fetchItem}>
            <IconRefresh className="h-4 w-4" />
          </Button>
        </div>
      }
    >
      {!loading && item && (
        <div className="space-y-6">
          {/* Content Card */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Brain className="h-4 w-4" />
                  Content
                </CardTitle>
                <div className="flex items-center gap-1">
                  {editing ? (
                    <>
                      <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => setEditing(false)}>
                        Cancel
                      </Button>
                      <Button size="sm" className="h-6 text-[10px]" onClick={handleSave} disabled={saving}>
                        {saving ? '...' : 'Save'}
                      </Button>
                    </>
                  ) : (
                    <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => {
                      setEditing(true)
                      setEditContent(item.content)
                      setEditTopic(item.topic)
                    }}>
                      <Edit className="h-3 w-3 mr-1" /> Edit
                    </Button>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {editing ? (
                <div className="space-y-3">
                  <textarea
                    value={editContent}
                    onChange={e => setEditContent(e.target.value)}
                    className="w-full h-32 p-3 text-sm rounded-md border border-border bg-background resize-none focus:outline-none focus:ring-1 focus:ring-primary/30"
                  />
                  <input
                    value={editTopic}
                    onChange={e => setEditTopic(e.target.value)}
                    placeholder="Topic"
                    className="w-full h-8 px-3 text-sm rounded-md border border-border bg-background focus:outline-none focus:ring-1 focus:ring-primary/30"
                  />
                </div>
              ) : (
                <p className="text-sm leading-relaxed whitespace-pre-wrap">{item.content}</p>
              )}
            </CardContent>
          </Card>

          {/* Metadata */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="py-3">
                <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
                  <Tag className="h-3 w-3" /> Topic
                </div>
                <Badge className={topicColor}>{item.topic}</Badge>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="py-3">
                <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
                  <Star className="h-3 w-3" /> Importance
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary rounded-full"
                      style={{ width: `${item.importance * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium">{(item.importance * 100).toFixed(0)}%</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="py-3">
                <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
                  <BarChart3 className="h-3 w-3" /> Score
                </div>
                <span className="text-xl font-bold">{item.score.toFixed(2)}</span>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="py-3">
                <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
                  <Clock className="h-3 w-3" /> Created
                </div>
                <span className="text-sm">{formatDate(new Date(item.timestamp).toISOString())}</span>
              </CardContent>
            </Card>
          </div>

          {/* Source & Actions */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Details</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Source</span>
                <span className="font-medium">{item.source || 'manual'}</span>
              </div>
              {item.url && (
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">URL</span>
                  <a href={item.url} target="_blank" rel="noopener noreferrer" className="font-medium text-primary hover:underline flex items-center gap-1">
                    {item.url.slice(0, 50)}... <ExternalLink className="h-3 w-3" />
                  </a>
                </div>
              )}
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">ID</span>
                <span className="font-mono text-xs">{item.id}</span>
              </div>

              <div className="pt-3 border-t flex gap-2">
                <Button variant="outline" size="sm" className="h-7 text-[11px]" onClick={() => router.push(`/chat?context=${item.id}`)}>
                  <Lightbulb className="h-3 w-3 mr-1" /> Use in Chat
                </Button>
                <Button variant="outline" size="sm" className="h-7 text-[11px] text-destructive border-destructive/30" onClick={handleDelete}>
                  <Trash2 className="h-3 w-3 mr-1" /> Delete
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </PageContainer>
  )
}
