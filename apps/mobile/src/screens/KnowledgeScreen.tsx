import React, {useEffect, useState, useCallback, useRef} from 'react';
import {
  FlatList,
  TextInput,
  TouchableOpacity,
  RefreshControl,
  Modal,
  Keyboard,
  Share,
  View,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {api} from '../services/api-client';
import {StatusBadge} from '../components/StatusBadge';
import {Icon} from '../components/Icon';
import type {KnowledgeItem} from '../types';

export function KnowledgeScreen() {
  const [items, setItems] = useState<KnowledgeItem[]>([]);
  const [topics, setTopics] = useState<string[]>([]);
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [addModalVisible, setAddModalVisible] = useState(false);
  const [editItem, setEditItem] = useState<KnowledgeItem | null>(null);
  const [formContent, setFormContent] = useState('');
  const [formTopic, setFormTopic] = useState('');
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [selectMode, setSelectMode] = useState(false);
  const searchTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchItems = useCallback(async () => {
    try {
      let data: KnowledgeItem[];
      if (search.trim()) {
        const result = await api.get<{items: KnowledgeItem[]}>(
          `/knowledge/search?query=${encodeURIComponent(search)}`,
        );
        data = result.items || [];
      } else {
        const params = new URLSearchParams({limit: '100', offset: '0'});
        if (selectedTopic) params.set('topic', selectedTopic);
        const result = await api.get<{items: KnowledgeItem[]}>(
          `/knowledge?${params}`,
        );
        data = result.items || [];
      }
      setItems(data);
    } catch {
      setItems([]);
    }
  }, [search, selectedTopic]);

  const fetchTopics = useCallback(async () => {
    try {
      const result = await api.get<{topics: string[]}>('/knowledge/topics');
      setTopics(result.topics || []);
    } catch {}
  }, []);

  useEffect(() => {
    fetchItems();
  }, [fetchItems]);

  useEffect(() => {
    fetchTopics();
  }, []);

  const onRefresh = async () => {
    setRefreshing(true);
    await Promise.all([fetchItems(), fetchTopics()]);
    setRefreshing(false);
  };

  const debouncedSearch = (text: string) => {
    setSearch(text);
    if (searchTimeout.current) clearTimeout(searchTimeout.current);
    searchTimeout.current = setTimeout(() => {}, 300);
  };

  const handleAdd = async () => {
    if (!formContent.trim()) return;
    try {
      await api.post('/knowledge', {
        content: formContent.trim(),
        topic: formTopic.trim() || undefined,
      });
      setAddModalVisible(false);
      setFormContent('');
      setFormTopic('');
      await fetchItems();
      await fetchTopics();
    } catch {}
  };

  const handleEdit = async () => {
    if (!editItem || !formContent.trim()) return;
    try {
      await api.patch(`/knowledge/${editItem.id}`, {
        content: formContent.trim(),
        topic: formTopic.trim() || undefined,
      });
      setEditItem(null);
      setFormContent('');
      setFormTopic('');
      await fetchItems();
    } catch {}
  };

  const handleDelete = async (id: string) => {
    try {
      await api.delete(`/knowledge/${id}`);
      await fetchItems();
    } catch {}
  };

  const handleBatchDelete = async () => {
    const ids = Array.from(selectedIds);
    if (ids.length === 0) return;
    try {
      await Promise.all(ids.map(id => api.delete(`/knowledge/${id}`)));
      setSelectedIds(new Set());
      setSelectMode(false);
      await fetchItems();
    } catch {}
  };

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleExport = async () => {
    try {
      const data = items.map(item => ({
        content: item.content,
        topic: item.topic,
        importance: item.importance,
      }));
      const json = JSON.stringify(data, null, 2);
      await Share.share({title: 'Knowledge Export', message: json});
    } catch {}
  };

  const openEdit = (item: KnowledgeItem) => {
    setEditItem(item);
    setFormContent(item.content);
    setFormTopic(item.topic || '');
  };

  const [importModalVisible, setImportModalVisible] = useState(false);
  const [importText, setImportText] = useState('');
  const [importing, setImporting] = useState(false);

  const handleImport = async () => {
    if (!importText.trim()) return;
    setImporting(true);
    try {
      const lines = importText.split('\n').filter(l => l.trim().length > 2);
      await Promise.all(
        lines.map(line =>
          api.post('/knowledge', {content: line.trim(), topic: formTopic.trim() || undefined}),
        ),
      );
      setImportModalVisible(false);
      setImportText('');
      setFormTopic('');
      await fetchItems();
      await fetchTopics();
    } catch {}
    setImporting(false);
  };

  const renderItem = ({item}: {item: KnowledgeItem}) => {
    const isSelected = selectedIds.has(item.id);
    return (
      <YStack
        backgroundColor={isSelected ? '#7C52C410' : 'white'}
        borderRadius={8}
        padding={12}
        borderWidth={isSelected ? 1 : 0}
        borderColor={isSelected ? '$color9' : 'transparent'}
        onLongPress={() => { setSelectMode(true); setSelectedIds(new Set([item.id])); }}
        onPress={() => {
          if (selectMode) toggleSelect(item.id);
          else openEdit(item);
        }}>
        <XStack alignItems="flex-start" gap={8} marginBottom={8}>
          {selectMode && (
            <YStack
              width={20} height={20} borderRadius={4} borderWidth={2}
              borderColor={isSelected ? '$color9' : '$borderColor'}
              backgroundColor={isSelected ? '$color9' : 'transparent'}
              alignItems="center" justifyContent="center" marginTop={2}>
              {isSelected && <View style={{width: 10, height: 10, borderRadius: 2, backgroundColor: 'white'}} />}
            </YStack>
          )}
          <Text fontSize={14} color="$color" flex={1} numberOfLines={3}>
            {item.content}
          </Text>
        </XStack>
        <XStack alignItems="center" justifyContent="space-between">
          <XStack alignItems="center" gap={8} flex={1}>
            {item.topic && <StatusBadge label={item.topic} variant="info" />}
            <XStack gap={3}>
              {Array.from({length: 5}).map((_, i) => (
                <View
                  key={i}
                  style={{
                    width: 6, height: 6, borderRadius: 3,
                    backgroundColor: i < item.importance ? '#F0935C' : '#E0DCE8',
                  }}
                />
              ))}
            </XStack>
          </XStack>
          {!selectMode && (
            <XStack gap={4}>
              <YStack onPress={() => openEdit(item)}>
                <StatusBadge label="Edit" variant="info" />
              </YStack>
              <YStack onPress={() => handleDelete(item.id)}>
                <StatusBadge label="Del" variant="error" />
              </YStack>
            </XStack>
          )}
        </XStack>
      </YStack>
    );
  };

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: '#F5F0FF'}} edges={['top']}>
      <YStack flex={1}>
        <XStack alignItems="center" justifyContent="space-between" paddingHorizontal={16} paddingVertical={12}>
          {selectMode ? (
            <XStack alignItems="center" justifyContent="space-between" flex={1}>
              <YStack onPress={() => { setSelectMode(false); setSelectedIds(new Set()); }}>
                <Text fontSize={13} fontWeight="600" color="$color9">Cancel</Text>
              </YStack>
              <Text fontSize={14} fontWeight="600" color="$color">{selectedIds.size} selected</Text>
              <YStack onPress={handleBatchDelete} opacity={selectedIds.size === 0 ? 0.4 : 1}>
                <Text fontSize={13} fontWeight="600" color="#D44C56">
                  Delete ({selectedIds.size})
                </Text>
              </YStack>
            </XStack>
          ) : (
            <>
              <Text fontSize={20} fontWeight="600" letterSpacing={-0.2} color="$color">
                What AI Knows About Me
              </Text>
              <XStack gap={8}>
                {items.length > 0 && (
                  <YStack backgroundColor="white" borderRadius={8} paddingHorizontal={12} paddingVertical={6} borderWidth={1} borderColor="$borderColor" onPress={handleExport}>
                    <Text fontSize={13} fontWeight="600" color="$color11">Export</Text>
                  </YStack>
                )}
                <YStack backgroundColor="white" borderRadius={8} paddingHorizontal={12} paddingVertical={6} borderWidth={1} borderColor="$borderColor" onPress={() => { setImportText(''); setImportModalVisible(true); }}>
                  <Text fontSize={13} fontWeight="600" color="$color11">Import</Text>
                </YStack>
                <YStack backgroundColor="$color9" borderRadius={8} paddingHorizontal={12} paddingVertical={6} onPress={() => { setFormContent(''); setFormTopic(''); setAddModalVisible(true); }}>
                  <Text fontSize={13} fontWeight="600" color="white">+ Add</Text>
                </YStack>
              </XStack>
            </>
          )}
        </XStack>

        {!selectMode && (
          <XStack paddingHorizontal={16} marginBottom={8}>
            <TextInput
              style={{flex: 1, fontSize: 14, color: '#1A1625', backgroundColor: 'white', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 8, borderWidth: 1, borderColor: '#E0DCE8'}}
              value={search}
              onChangeText={debouncedSearch}
              placeholder="Search knowledge..."
              placeholderTextColor="#9B95A8"
              returnKeyType="search"
              onSubmitEditing={() => Keyboard.dismiss()}
            />
          </XStack>
        )}

        {!selectMode && topics.length > 0 && (
          <FlatList
            horizontal
            data={[null, ...topics]}
            renderItem={({item: topic}) => (
              <YStack
                paddingHorizontal={12} paddingVertical={4} borderRadius={9999}
                backgroundColor={selectedTopic === topic ? '$color9' : 'white'}
                borderWidth={1}
                borderColor={selectedTopic === topic ? '$color9' : '$borderColor'}
                onPress={() => setSelectedTopic(topic)}>
                <Text fontSize={11} fontWeight="500" color={selectedTopic === topic ? 'white' : '$color11'}>
                  {topic || 'All'}
                </Text>
              </YStack>
            )}
            keyExtractor={item => item || 'all'}
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={{paddingHorizontal: 16, gap: 4, marginBottom: 8}}
          />
        )}

        <FlatList
          data={items}
          renderItem={renderItem}
          keyExtractor={item => item.id}
          contentContainerStyle={{padding: 16, gap: 8}}
          keyboardDismissMode="on-drag"
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
          ListEmptyComponent={
            <YStack alignItems="center" paddingVertical={64}>
              <Icon name="book-open" size={48} color="#9B95A8" />
              <Text fontSize={14} color="$color10">No knowledge items yet</Text>
            </YStack>
          }
        />
      </YStack>

      <Modal visible={addModalVisible} animationType="slide" transparent>
        <TouchableOpacity activeOpacity={1} onPress={() => setAddModalVisible(false)} style={{flex: 1, backgroundColor: 'rgba(0,0,0,0.4)', justifyContent: 'flex-end'}}>
          <TouchableOpacity activeOpacity={1} style={{backgroundColor: '#F5F0FF', borderTopLeftRadius: 16, borderTopRightRadius: 16, padding: 20, gap: 12}} onPress={() => {}}>
            <Text fontSize={16} fontWeight="600" color="$color">Add Knowledge</Text>
            <TextInput
              style={{fontSize: 14, color: '#1A1625', backgroundColor: 'white', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 12, minHeight: 100, textAlignVertical: 'top', borderWidth: 1, borderColor: '#E0DCE8'}}
              value={formContent}
              onChangeText={setFormContent}
              placeholder="Content..."
              placeholderTextColor="#9B95A8"
              multiline
              autoFocus
            />
            <TextInput
              style={{fontSize: 14, color: '#1A1625', backgroundColor: 'white', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 12, borderWidth: 1, borderColor: '#E0DCE8'}}
              value={formTopic}
              onChangeText={setFormTopic}
              placeholder="Topic (optional)"
              placeholderTextColor="#9B95A8"
              returnKeyType="done"
            />
            <XStack gap={8} justifyContent="flex-end">
              <YStack onPress={() => setAddModalVisible(false)} paddingHorizontal={20} paddingVertical={8} borderRadius={8}>
                <Text fontSize={13} fontWeight="600" color="$color11">Cancel</Text>
              </YStack>
              <YStack onPress={handleAdd} opacity={!formContent.trim() ? 0.4 : 1} backgroundColor="$color9" paddingHorizontal={24} paddingVertical={8} borderRadius={8}>
                <Text fontSize={13} fontWeight="600" color="white">Add</Text>
              </YStack>
            </XStack>
          </TouchableOpacity>
        </TouchableOpacity>
      </Modal>

      <Modal visible={!!editItem} animationType="slide" transparent>
        <TouchableOpacity activeOpacity={1} onPress={() => setEditItem(null)} style={{flex: 1, backgroundColor: 'rgba(0,0,0,0.4)', justifyContent: 'flex-end'}}>
          <TouchableOpacity activeOpacity={1} style={{backgroundColor: '#F5F0FF', borderTopLeftRadius: 16, borderTopRightRadius: 16, padding: 20, gap: 12}} onPress={() => {}}>
            <Text fontSize={16} fontWeight="600" color="$color">Edit Knowledge</Text>
            <TextInput
              style={{fontSize: 14, color: '#1A1625', backgroundColor: 'white', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 12, minHeight: 100, textAlignVertical: 'top', borderWidth: 1, borderColor: '#E0DCE8'}}
              value={formContent}
              onChangeText={setFormContent}
              placeholder="Content..."
              placeholderTextColor="#9B95A8"
              multiline
            />
            <TextInput
              style={{fontSize: 14, color: '#1A1625', backgroundColor: 'white', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 12, borderWidth: 1, borderColor: '#E0DCE8'}}
              value={formTopic}
              onChangeText={setFormTopic}
              placeholder="Topic (optional)"
              placeholderTextColor="#9B95A8"
              returnKeyType="done"
            />
            <XStack gap={8} justifyContent="flex-end">
              <YStack onPress={() => setEditItem(null)} paddingHorizontal={20} paddingVertical={8} borderRadius={8}>
                <Text fontSize={13} fontWeight="600" color="$color11">Cancel</Text>
              </YStack>
              <YStack onPress={handleEdit} opacity={!formContent.trim() ? 0.4 : 1} backgroundColor="$color9" paddingHorizontal={24} paddingVertical={8} borderRadius={8}>
                <Text fontSize={13} fontWeight="600" color="white">Save</Text>
              </YStack>
            </XStack>
          </TouchableOpacity>
        </TouchableOpacity>
      </Modal>

      <Modal visible={importModalVisible} animationType="slide" transparent>
        <TouchableOpacity activeOpacity={1} onPress={() => setImportModalVisible(false)} style={{flex: 1, backgroundColor: 'rgba(0,0,0,0.4)', justifyContent: 'flex-end'}}>
          <TouchableOpacity activeOpacity={1} style={{backgroundColor: '#F5F0FF', borderTopLeftRadius: 16, borderTopRightRadius: 16, padding: 20, gap: 12}} onPress={() => {}}>
            <Text fontSize={16} fontWeight="600" color="$color">Import Knowledge</Text>
            <Text fontSize={13} color="$color10" marginBottom={4}>
              Paste one item per line. Each line becomes a knowledge entry.
            </Text>
            <TextInput
              style={{fontSize: 14, color: '#1A1625', backgroundColor: 'white', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 12, minHeight: 100, textAlignVertical: 'top', borderWidth: 1, borderColor: '#E0DCE8'}}
              value={importText}
              onChangeText={setImportText}
              placeholder="Line 1\nLine 2\nLine 3..."
              placeholderTextColor="#9B95A8"
              multiline
              autoFocus
            />
            <TextInput
              style={{fontSize: 14, color: '#1A1625', backgroundColor: 'white', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 12, borderWidth: 1, borderColor: '#E0DCE8'}}
              value={formTopic}
              onChangeText={setFormTopic}
              placeholder="Topic for all (optional)"
              placeholderTextColor="#9B95A8"
              returnKeyType="done"
            />
            <XStack gap={8} justifyContent="flex-end">
              <YStack onPress={() => setImportModalVisible(false)} paddingHorizontal={20} paddingVertical={8} borderRadius={8}>
                <Text fontSize={13} fontWeight="600" color="$color11">Cancel</Text>
              </YStack>
              <YStack onPress={handleImport} opacity={(!importText.trim() || importing) ? 0.4 : 1} backgroundColor="$color9" paddingHorizontal={24} paddingVertical={8} borderRadius={8}>
                <Text fontSize={13} fontWeight="600" color="white">
                  {importing ? 'Importing...' : `Import (${importText.split('\n').filter(l => l.trim().length > 2).length})`}
                </Text>
              </YStack>
            </XStack>
          </TouchableOpacity>
        </TouchableOpacity>
      </Modal>
    </SafeAreaView>
  );
}
