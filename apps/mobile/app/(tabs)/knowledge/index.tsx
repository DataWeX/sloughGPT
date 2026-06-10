import { useState, useEffect, useCallback } from 'react'
import {
  FlatList,
  RefreshControl,
  KeyboardAvoidingView,
  Platform,
} from 'react-native'
import {
  YStack,
  XStack,
  Text,
  Input,
  Button,
  Card,
  Paragraph,
  Sheet,
  Label,
  ScrollView,
} from 'tamagui'
import {
  Search,
  Plus,
  Trash2,
  Edit3,
  BookOpen,
  X,
} from '@tamagui/lucide-icons'
import * as Haptics from 'expo-haptics'
import { apiGet, apiPost, apiPatch, apiDelete } from '@/lib/api-client'

interface KnowledgeItem {
  id: string
  content: string
  topic: string
  source: string
  importance: number
  score: number
  timestamp: number
}

export default function KnowledgeScreen() {
  const [items, setItems] = useState<KnowledgeItem[]>([])
  const [search, setSearch] = useState('')
  const [topics, setTopics] = useState<string[]>([])
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [loading, setLoading] = useState(true)

  const [showAddSheet, setShowAddSheet] = useState(false)
  const [newContent, setNewContent] = useState('')
  const [newTopic, setNewTopic] = useState('')
  const [saving, setSaving] = useState(false)

  const [editItem, setEditItem] = useState<KnowledgeItem | null>(null)
  const [editContent, setEditContent] = useState('')
  const [editTopic, setEditTopic] = useState('')

  const fetchItems = useCallback(async () => {
    try {
      if (search.trim()) {
        const data = await apiGet<{ results: KnowledgeItem[] }>(
          `/knowledge/search?query=${encodeURIComponent(search)}`
        )
        setItems(data.results || [])
      } else {
        const params = new URLSearchParams({ limit: '100', offset: '0' })
        if (selectedTopic) params.set('topic', selectedTopic)
        const data = await apiGet<KnowledgeItem[]>(`/knowledge?${params}`)
        setItems(Array.isArray(data) ? data : [])
      }
    } catch {
      setItems([])
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [search, selectedTopic])

  const fetchTopics = useCallback(async () => {
    try {
      const data = await apiGet<string[]>('/knowledge/topics')
      setTopics(Array.isArray(data) ? data : [])
    } catch {
      setTopics([])
    }
  }, [])

  useEffect(() => {
    fetchItems()
    fetchTopics()
  }, [fetchItems, fetchTopics])

  const onRefresh = () => {
    setRefreshing(true)
    fetchItems()
    fetchTopics()
  }

  const handleAdd = async () => {
    if (!newContent.trim()) return
    setSaving(true)
    try {
      await apiPost('/knowledge', {
        content: newContent.trim(),
        topic: newTopic.trim() || undefined,
      })
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success)
      setNewContent('')
      setNewTopic('')
      setShowAddSheet(false)
      fetchItems()
    } catch {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error)
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (id: string) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium)
    try {
      await apiDelete(`/knowledge/${id}`)
      setItems((prev) => prev.filter((i) => i.id !== id))
    } catch {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error)
    }
  }

  const handleEdit = async () => {
    if (!editItem || !editContent.trim()) return
    setSaving(true)
    try {
      await apiPatch(`/knowledge/${editItem.id}`, {
        content: editContent.trim(),
        topic: editTopic.trim() || undefined,
      })
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success)
      setEditItem(null)
      fetchItems()
    } catch {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error)
    } finally {
      setSaving(false)
    }
  }

  function renderItem({ item }: { item: KnowledgeItem }) {
    return (
      <Card
        backgroundColor="$backgroundStrong"
        borderRadius="$4"
        padding="$3"
        marginHorizontal="$3"
        marginVertical="$1"
      >
        <Text color="$color" fontSize="$3" numberOfLines={3}>
          {item.content}
        </Text>
        <XStack justifyContent="space-between" alignItems="center" marginTop="$2">
          <XStack gap="$2" alignItems="center">
            {item.topic && (
              <XStack
                backgroundColor="$primary"
                borderRadius="$6"
                paddingHorizontal="$2"
                paddingVertical={2}
                opacity={0.8}
              >
                <Text color="$background" fontSize="$1">
                  {item.topic}
                </Text>
              </XStack>
            )}
            <XStack gap="$1" alignItems="center">
              {Array.from({ length: Math.min(5, Math.round(item.importance * 5)) }).map(
                (_, i) => (
                  <XStack
                    key={i}
                    width={4}
                    height={4}
                    borderRadius={2}
                    backgroundColor="$accent"
                  />
                )
              )}
            </XStack>
          </XStack>
          <XStack gap="$1">
            <Button
              size="$2"
              chromeless
              icon={<Edit3 size={14} />}
              onPress={() => {
                setEditItem(item)
                setEditContent(item.content)
                setEditTopic(item.topic || '')
              }}
            />
            <Button
              size="$2"
              chromeless
              icon={<Trash2 size={14} color="$destructive" />}
              onPress={() => handleDelete(item.id)}
            />
          </XStack>
        </XStack>
      </Card>
    )
  }

  return (
    <YStack flex={1} backgroundColor="$background">
      {/* Header */}
      <XStack
        paddingHorizontal="$3"
        paddingVertical="$2"
        alignItems="center"
        justifyContent="space-between"
        borderBottomWidth={1}
        borderBottomColor="$borderColor"
        paddingTop={56}
      >
        <Text fontSize="$6" fontWeight="700" color="$color">
          Knowledge
        </Text>
        <Button
          size="$3"
          theme="active"
          icon={<Plus size={16} />}
          onPress={() => setShowAddSheet(true)}
        >
          Add
        </Button>
      </XStack>

      {/* Search */}
      <XStack paddingHorizontal="$3" paddingVertical="$2">
        <Input
          flex={1}
          size="$3"
          placeholder="Search knowledge..."
          value={search}
          onChangeText={(text) => {
            setSearch(text)
            setTimeout(fetchItems, 300)
          }}
        />
      </XStack>

      {/* Topic chips */}
      {topics.length > 0 && (
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          <XStack gap="$2" paddingHorizontal="$3" paddingBottom="$2">
            <Button
              size="$2"
              chromeless
              borderRadius="$6"
              backgroundColor={!selectedTopic ? '$primary' : '$backgroundStrong'}
              onPress={() => {
                setSelectedTopic(null)
              }}
            >
              <Text
                color={!selectedTopic ? '$background' : '$color'}
                fontSize="$2"
              >
                All
              </Text>
            </Button>
            {topics.map((topic) => {
              const isActive = topic === selectedTopic
              return (
                <Button
                  key={topic}
                  size="$2"
                  chromeless
                  borderRadius="$6"
                  backgroundColor={isActive ? '$primary' : '$backgroundStrong'}
                  onPress={() => setSelectedTopic(isActive ? null : topic)}
                >
                  <Text
                    color={isActive ? '$background' : '$color'}
                    fontSize="$2"
                  >
                    {topic}
                  </Text>
                </Button>
              )
            })}
          </XStack>
        </ScrollView>
      )}

      {/* List */}
      <FlatList
        data={items}
        renderItem={renderItem}
        keyExtractor={(item) => item.id}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
        ListEmptyComponent={
          <YStack alignItems="center" paddingVertical="$8">
            <BookOpen size={48} color="$placeholderColor" opacity={0.3} />
            <Paragraph color="$placeholderColor" marginTop="$3">
              {loading ? 'Loading...' : 'No knowledge items yet'}
            </Paragraph>
          </YStack>
        }
        contentContainerStyle={{ paddingBottom: 100 }}
      />

      {/* Add Sheet */}
      <Sheet
        modal
        open={showAddSheet}
        onOpenChange={setShowAddSheet}
        snapPoints={[50]}
        snapPointsMode="percent"
        dismissOnSnapToBottom
      >
        <Sheet.Overlay backgroundColor="rgba(0,0,0,0.4)" />
        <Sheet.Frame backgroundColor="$background" padding="$4">
          <KeyboardAvoidingView
            behavior={Platform.OS === 'ios' ? 'padding' : undefined}
            style={{ flex: 1 }}
          >
            <YStack flex={1} gap="$3">
              <XStack justifyContent="space-between" alignItems="center">
                <Text fontSize="$5" fontWeight="700" color="$color">
                  Add Knowledge
                </Text>
                <Button
                  size="$2"
                  chromeless
                  icon={<X size={18} />}
                  onPress={() => setShowAddSheet(false)}
                />
              </XStack>

              <YStack gap="$1">
                <Label>Content</Label>
                <Input
                  size="$4"
                  placeholder="What should I remember?"
                  value={newContent}
                  onChangeText={setNewContent}
                  multiline
                  numberOfLines={4}
                  minHeight={100}
                />
              </YStack>

              <YStack gap="$1">
                <Label>Topic (optional)</Label>
                <Input
                  size="$4"
                  placeholder="e.g. tech, personal, work"
                  value={newTopic}
                  onChangeText={setNewTopic}
                />
              </YStack>

              <Button
                size="$5"
                theme="active"
                onPress={handleAdd}
                disabled={saving || !newContent.trim()}
              >
                {saving ? 'Saving...' : 'Save'}
              </Button>
            </YStack>
          </KeyboardAvoidingView>
        </Sheet.Frame>
      </Sheet>

      {/* Edit Sheet */}
      <Sheet
        modal
        open={!!editItem}
        onOpenChange={(open: boolean) => !open && setEditItem(null)}
        snapPoints={[50]}
        snapPointsMode="percent"
        dismissOnSnapToBottom
      >
        <Sheet.Overlay backgroundColor="rgba(0,0,0,0.4)" />
        <Sheet.Frame backgroundColor="$background" padding="$4">
          <KeyboardAvoidingView
            behavior={Platform.OS === 'ios' ? 'padding' : undefined}
            style={{ flex: 1 }}
          >
            <YStack flex={1} gap="$3">
              <XStack justifyContent="space-between" alignItems="center">
                <Text fontSize="$5" fontWeight="700" color="$color">
                  Edit Knowledge
                </Text>
                <Button
                  size="$2"
                  chromeless
                  icon={<X size={18} />}
                  onPress={() => setEditItem(null)}
                />
              </XStack>

              <YStack gap="$1">
                <Label>Content</Label>
                <Input
                  size="$4"
                  value={editContent}
                  onChangeText={setEditContent}
                  multiline
                  numberOfLines={4}
                  minHeight={100}
                />
              </YStack>

              <YStack gap="$1">
                <Label>Topic</Label>
                <Input
                  size="$4"
                  value={editTopic}
                  onChangeText={setEditTopic}
                />
              </YStack>

              <Button
                size="$5"
                theme="active"
                onPress={handleEdit}
                disabled={saving || !editContent.trim()}
              >
                {saving ? 'Saving...' : 'Update'}
              </Button>
            </YStack>
          </KeyboardAvoidingView>
        </Sheet.Frame>
      </Sheet>
    </YStack>
  )
}
