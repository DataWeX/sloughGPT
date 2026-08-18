import React, {useEffect, useState, useCallback, useRef} from 'react';
import {
  FlatList,
  TextInput as RNTextInput,
  Pressable,
  RefreshControl,
  Modal,
  Keyboard,
  Share,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {triggerHaptic} from '../services/haptics';
import {pickDocument} from '../services/file-upload';
import type {KnowledgeItem} from '../types';

export function KnowledgeScreen() {
  const colors = useColors();
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
      triggerHaptic('success');
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
      triggerHaptic('light');
      await api.delete(`/knowledge/${id}`);
      await fetchItems();
    } catch {}
  };

  const handleBatchDelete = async () => {
    const ids = Array.from(selectedIds);
    if (ids.length === 0) return;
    try {
      triggerHaptic('medium');
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
      triggerHaptic('light');
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
      triggerHaptic('success');
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

  const handleImportFile = async () => {
    try {
      const file = await pickDocument();
      if (!file) return;
      setImporting(true);
      const RNFS = require('react-native-fs');
      const content = await RNFS.readFile(decodeURIComponent(file.uri.replace('file://', '')));
      const data = JSON.parse(content);
      const entries = Array.isArray(data) ? data : [data];
      await Promise.all(
        entries.map((entry: any) => {
          const text = typeof entry === 'string' ? entry : entry.content || entry.text || entry.knowledge || '';
          const topic = typeof entry === 'object' ? (entry.topic || entry.category || formTopic.trim() || undefined) : (formTopic.trim() || undefined);
          return text.trim() ? api.post('/knowledge', {content: text.trim(), topic}) : Promise.resolve();
        }),
      );
      setImportModalVisible(false);
      setImportText('');
      setFormTopic('');
      await fetchItems();
      await fetchTopics();
    } catch (e: any) {
      // JSON parse or file read error — silently handled
    }
    setImporting(false);
  };

  const accent = colors.primary;
  const bgMuted = colors.primaryAlpha(0.06);

  const renderItem = ({item}: {item: KnowledgeItem}) => {
    const isSelected = selectedIds.has(item.id);
    return (
      <YStack
        backgroundColor={isSelected ? bgMuted : 'transparent'}
        borderRadius={12}
        padding={14}
        borderWidth={isSelected ? 1 : 0.5}
        borderColor={isSelected ? accent : colors.border}
        onLongPress={() => { setSelectMode(true); setSelectedIds(new Set([item.id])); }}
        onPress={() => {
          if (selectMode) toggleSelect(item.id);
          else openEdit(item);
        }}
        pressStyle={{opacity: 0.85}}>
        <XStack alignItems="flex-start" gap={10} marginBottom={8}>
          {selectMode && (
            <YStack
              width={22} height={22} borderRadius={6} borderWidth={2}
              borderColor={isSelected ? accent : colors.border}
              backgroundColor={isSelected ? accent : 'transparent'}
              alignItems="center" justifyContent="center" marginTop={1}>
              {isSelected && (
                <Icon name="check" size={12} color="white" />
              )}
            </YStack>
          )}
          <Text fontSize={14} color={colors.text} flex={1} numberOfLines={3}>
            {item.content}
          </Text>
        </XStack>
        <XStack alignItems="center" justifyContent="space-between">
          <XStack alignItems="center" gap={8} flex={1}>
            {item.topic && (
              <YStack backgroundColor={bgMuted} paddingHorizontal={8} paddingVertical={2} borderRadius={6}>
                <Text fontSize={10} fontWeight="500" color={colors.primary}>{item.topic}</Text>
              </YStack>
            )}
            <XStack gap={3}>
              {Array.from({length: 5}).map((_, i) => (
                <YStack
                  key={i}
                  width={6} height={6} borderRadius={3}
                  backgroundColor={i < item.importance ? accent : colors.border}
                />
              ))}
            </XStack>
          </XStack>
          {!selectMode && (
            <XStack gap={6}>
              <Pressable onPress={() => openEdit(item)} accessibilityLabel="Edit knowledge item">
                <YStack width={28} height={28} borderRadius={8} backgroundColor={bgMuted} alignItems="center" justifyContent="center">
                  <Icon name="edit" size={12} color={accent} />
                </YStack>
              </Pressable>
              <Pressable onPress={() => handleDelete(item.id)} accessibilityLabel="Delete knowledge item">
                <YStack width={28} height={28} borderRadius={8} backgroundColor={colors.errorAlpha(0.08)} alignItems="center" justifyContent="center">
                  <Icon name="trash-2" size={12} color={colors.error} />
                </YStack>
              </Pressable>
            </XStack>
          )}
        </XStack>
      </YStack>
    );
  };

  const renderBottomSheet = (
    visible: boolean,
    onClose: () => void,
    title: string,
    children: React.ReactNode,
  ) => (
    <Modal visible={visible} animationType="slide" transparent>
      <YStack flex={1} justifyContent="flex-end">
        <Pressable style={{flex: 1}} onPress={onClose} />
        <YStack
          backgroundColor={colors.background}
          borderTopLeftRadius={24}
          borderTopRightRadius={24}
          padding={20}
          gap={14}>
          <XStack alignItems="center" justifyContent="space-between" marginBottom={4}>
            <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color={colors.text}>{title}</Text>
            <Pressable onPress={onClose} accessibilityLabel="Close">
              <YStack width={28} height={28} borderRadius={9} alignItems="center" justifyContent="center">
                <Icon name="x" size={14} color={colors.textSecondary} />
              </YStack>
            </Pressable>
          </XStack>
          {children}
        </YStack>
      </YStack>
    </Modal>
  );

  return (
    <SafeAreaView style={{flex: 1}} edges={['top']}>
      <YStack flex={1}>
        <XStack alignItems="center" justifyContent="space-between" paddingHorizontal={16} paddingVertical={12}>
          {selectMode ? (
            <XStack alignItems="center" justifyContent="space-between" flex={1}>
              <Pressable onPress={() => { setSelectMode(false); setSelectedIds(new Set()); }}>
                <Text fontSize={13} fontWeight="600" color={colors.primary}>Cancel</Text>
              </Pressable>
              <Text fontSize={14} fontWeight="600" color={colors.text}>{selectedIds.size} selected</Text>
              <Pressable onPress={handleBatchDelete} style={{opacity: selectedIds.size === 0 ? 0.4 : 1}}>
                <Text fontSize={13} fontWeight="600" color={colors.error}>
                  Delete ({selectedIds.size})
                </Text>
              </Pressable>
            </XStack>
          ) : (
            <>
              <Text fontSize={20} fontWeight="600" letterSpacing={-0.2} color={colors.text}>
                What AI Knows About Me
              </Text>
              <XStack gap={8}>
                {items.length > 0 && (
                  <Pressable onPress={handleExport}>
                    <YStack backgroundColor={bgMuted} borderRadius={8} paddingHorizontal={12} paddingVertical={8}>
                      <Text fontSize={12} fontWeight="600" color={colors.primary}>Export</Text>
                    </YStack>
                  </Pressable>
                )}
                <Pressable onPress={() => { setImportText(''); setImportModalVisible(true); }}>
                  <YStack backgroundColor={bgMuted} borderRadius={8} paddingHorizontal={12} paddingVertical={8}>
                    <Text fontSize={12} fontWeight="600" color={colors.primary}>Import</Text>
                  </YStack>
                </Pressable>
                <Pressable onPress={() => { setFormContent(''); setFormTopic(''); setAddModalVisible(true); }}>
                  <YStack backgroundColor={accent} borderRadius={8} paddingHorizontal={14} paddingVertical={8}>
                    <Text fontSize={12} fontWeight="600" color="white">+ Add</Text>
                  </YStack>
                </Pressable>
              </XStack>
            </>
          )}
        </XStack>

        {!selectMode && (
          <YStack paddingHorizontal={16} marginBottom={8}>
            <RNTextInput
              style={{
                flex: 1, fontSize: 14, color={colors.text},
                backgroundColor: colors.primaryAlpha(0.04),
                borderRadius: 10,
                paddingHorizontal: 14,
                paddingVertical: 10,
                borderWidth: 0.5,
                borderColor: colors.primaryAlpha(0.12),
              }}
              value={search}
              onChangeText={debouncedSearch}
              placeholder="Search knowledge..."
              placeholderTextColor={colors.textMuted}
              returnKeyType="search"
              onSubmitEditing={() => Keyboard.dismiss()}
            />
          </YStack>
        )}

        {!selectMode && topics.length > 0 && (
          <FlatList
            horizontal
            data={[null, ...topics]}
            renderItem={({item: topic}) => (
              <YStack
                paddingHorizontal={12} paddingVertical={5} borderRadius={999}
                backgroundColor={selectedTopic === topic ? accent : 'transparent'}
                borderWidth={0.5}
                borderColor={selectedTopic === topic ? accent : colors.border}
                onPress={() => setSelectedTopic(topic)}>
                <Text fontSize={11} fontWeight="500" color={selectedTopic === topic ? 'white' : colors.textMuted}>
                  {topic || 'All'}
                </Text>
              </YStack>
            )}
            keyExtractor={item => item || 'all'}
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={{paddingHorizontal: 16, gap: 6, marginBottom: 8, paddingVertical: 4}}
          />
        )}

        <FlatList
          data={items}
          renderItem={renderItem}
          keyExtractor={item => item.id}
          contentContainerStyle={{padding: 16, gap: 8}}
          keyboardDismissMode="on-drag"
          removeClippedSubviews
          maxToRenderPerBatch={10}
          windowSize={11}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              tintColor={accent}
            />
          }
          ListEmptyComponent={
            <YStack alignItems="center" paddingVertical={64}>
              <Icon name="book-open" size={48} color={colors.textMuted} />
              <Text fontSize={14} color={colors.textSecondary}>No knowledge items yet</Text>
            </YStack>
          }
        />
      </YStack>

      {renderBottomSheet(addModalVisible, () => setAddModalVisible(false), 'Add Knowledge', (
        <>
          <RNTextInput
            style={{
              fontSize: 14, color={colors.text},
              backgroundColor: colors.primaryAlpha(0.04),
              borderRadius: 10,
              paddingHorizontal: 14,
              paddingVertical: 12,
              minHeight: 100,
              textAlignVertical: 'top',
              borderWidth: 0.5,
              borderColor: colors.primaryAlpha(0.12),
            }}
            value={formContent}
            onChangeText={setFormContent}
            placeholder="Content..."
            placeholderTextColor={colors.textMuted}
            multiline
            autoFocus
          />
          <RNTextInput
            style={{
              fontSize: 14, color={colors.text},
              backgroundColor: colors.primaryAlpha(0.04),
              borderRadius: 10,
              paddingHorizontal: 14,
              paddingVertical: 12,
              borderWidth: 0.5,
              borderColor: colors.primaryAlpha(0.12),
            }}
            value={formTopic}
            onChangeText={setFormTopic}
            placeholder="Topic (optional)"
            placeholderTextColor={colors.textMuted}
            returnKeyType="done"
          />
          <XStack gap={8} justifyContent="flex-end">
            <Pressable onPress={() => setAddModalVisible(false)}>
              <YStack paddingHorizontal={20} paddingVertical={10} borderRadius={8}>
                <Text fontSize={13} fontWeight="600" color={colors.textMuted}>Cancel</Text>
              </YStack>
            </Pressable>
            <Pressable onPress={handleAdd} disabled={!formContent.trim()}>
              <YStack
                backgroundColor={accent}
                opacity={!formContent.trim() ? 0.4 : 1}
                paddingHorizontal={24} paddingVertical={10} borderRadius={8}
              >
                <Text fontSize={13} fontWeight="600" color="white">Add</Text>
              </YStack>
            </Pressable>
          </XStack>
        </>
      ))}

      {renderBottomSheet(!!editItem, () => setEditItem(null), 'Edit Knowledge', (
        <>
          <RNTextInput
            style={{
              fontSize: 14, color={colors.text},
              backgroundColor: colors.primaryAlpha(0.04),
              borderRadius: 10,
              paddingHorizontal: 14,
              paddingVertical: 12,
              minHeight: 100,
              textAlignVertical: 'top',
              borderWidth: 0.5,
              borderColor: colors.primaryAlpha(0.12),
            }}
            value={formContent}
            onChangeText={setFormContent}
            placeholder="Content..."
            placeholderTextColor={colors.textMuted}
            multiline
          />
          <RNTextInput
            style={{
              fontSize: 14, color={colors.text},
              backgroundColor: colors.primaryAlpha(0.04),
              borderRadius: 10,
              paddingHorizontal: 14,
              paddingVertical: 12,
              borderWidth: 0.5,
              borderColor: colors.primaryAlpha(0.12),
            }}
            value={formTopic}
            onChangeText={setFormTopic}
            placeholder="Topic (optional)"
            placeholderTextColor={colors.textMuted}
            returnKeyType="done"
          />
          <XStack gap={8} justifyContent="flex-end">
            <Pressable onPress={() => setEditItem(null)}>
              <YStack paddingHorizontal={20} paddingVertical={10} borderRadius={8}>
                <Text fontSize={13} fontWeight="600" color={colors.textMuted}>Cancel</Text>
              </YStack>
            </Pressable>
            <Pressable onPress={handleEdit} disabled={!formContent.trim()}>
              <YStack
                backgroundColor={accent}
                opacity={!formContent.trim() ? 0.4 : 1}
                paddingHorizontal={24} paddingVertical={10} borderRadius={8}
              >
                <Text fontSize={13} fontWeight="600" color="white">Save</Text>
              </YStack>
            </Pressable>
          </XStack>
        </>
      ))}

      {renderBottomSheet(importModalVisible, () => setImportModalVisible(false), 'Import Knowledge', (
        <>
          <Text fontSize={13} color={colors.textSecondary} marginBottom={4}>
            Paste one item per line. Each line becomes a knowledge entry.
          </Text>
          <RNTextInput
            style={{
              fontSize: 14, color={colors.text},
              backgroundColor: colors.primaryAlpha(0.04),
              borderRadius: 10,
              paddingHorizontal: 14,
              paddingVertical: 12,
              minHeight: 100,
              textAlignVertical: 'top',
              borderWidth: 0.5,
              borderColor: colors.primaryAlpha(0.12),
            }}
            value={importText}
            onChangeText={setImportText}
            placeholder="Line 1\nLine 2\nLine 3..."
            placeholderTextColor={colors.textMuted}
            multiline
            autoFocus
          />
          <RNTextInput
            style={{
              fontSize: 14, color={colors.text},
              backgroundColor: colors.primaryAlpha(0.04),
              borderRadius: 10,
              paddingHorizontal: 14,
              paddingVertical: 12,
              borderWidth: 0.5,
              borderColor: colors.primaryAlpha(0.12),
            }}
            value={formTopic}
            onChangeText={setFormTopic}
            placeholder="Topic for all (optional)"
            placeholderTextColor={colors.textMuted}
            returnKeyType="done"
          />
          <Pressable onPress={handleImportFile} disabled={importing}>
            <XStack
              alignItems="center" gap={8}
              paddingHorizontal={14} paddingVertical={12}
              borderRadius={10}
              backgroundColor={colors.primaryAlpha(0.06)}
              borderWidth={0.5} borderColor={colors.primaryAlpha(0.12)}>
              <Icon name="upload" size={16} color={accent} />
              <Text fontSize={13} fontWeight="500" color={colors.textMuted}>
                Pick JSON file instead
              </Text>
            </XStack>
          </Pressable>
          <XStack gap={8} justifyContent="flex-end">
            <Pressable onPress={() => setImportModalVisible(false)}>
              <YStack paddingHorizontal={20} paddingVertical={10} borderRadius={8}>
                <Text fontSize={13} fontWeight="600" color={colors.textMuted}>Cancel</Text>
              </YStack>
            </Pressable>
            <Pressable onPress={handleImport} disabled={!importText.trim() || importing}>
              <YStack
                backgroundColor={accent}
                opacity={(!importText.trim() || importing) ? 0.4 : 1}
                paddingHorizontal={24} paddingVertical={10} borderRadius={8}
              >
                <Text fontSize={13} fontWeight="600" color="white">
                  {importing ? 'Importing...' : `Import (${importText.split('\n').filter(l => l.trim().length > 2).length})`}
                </Text>
              </YStack>
            </Pressable>
          </XStack>
        </>
      ))}
    </SafeAreaView>
  );
}
