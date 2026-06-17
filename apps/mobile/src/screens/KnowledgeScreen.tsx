import React, {useEffect, useState, useCallback} from 'react';
import {
  View,
  Text,
  FlatList,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  RefreshControl,
  Modal,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {api} from '../../services/api-client';
import {StatusBadge} from '../../components/StatusBadge';
import {colors, spacing, radii, typography} from '../../theme';
import type {KnowledgeItem} from '../../types';

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

  const openEdit = (item: KnowledgeItem) => {
    setEditItem(item);
    setFormContent(item.content);
    setFormTopic(item.topic || '');
  };

  const renderItem = ({item}: {item: KnowledgeItem}) => (
    <View style={styles.itemCard}>
      <View style={styles.itemHeader}>
        <Text style={styles.itemContent} numberOfLines={3}>
          {item.content}
        </Text>
      </View>
      <View style={styles.itemFooter}>
        <View style={styles.itemMeta}>
          {item.topic && <StatusBadge label={item.topic} variant="info" />}
          <View style={styles.importanceDots}>
            {Array.from({length: 5}).map((_, i) => (
              <View
                key={i}
                style={[
                  styles.dot,
                  i < item.importance && styles.dotFilled,
                ]}
              />
            ))}
          </View>
        </View>
        <View style={styles.itemActions}>
          <TouchableOpacity
            style={styles.actionBtn}
            onPress={() => openEdit(item)}>
            <Text style={styles.actionText}>Edit</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.actionBtn, styles.deleteBtn]}
            onPress={() => handleDelete(item.id)}>
            <Text style={[styles.actionText, styles.deleteText]}>Del</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <View style={styles.header}>
        <Text style={styles.title}>Knowledge</Text>
        <TouchableOpacity
          style={styles.addBtn}
          onPress={() => {
            setFormContent('');
            setFormTopic('');
            setAddModalVisible(true);
          }}>
          <Text style={styles.addBtnText}>+ Add</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.searchRow}>
        <TextInput
          style={styles.searchInput}
          value={search}
          onChangeText={setSearch}
          placeholder="Search knowledge..."
          placeholderTextColor={colors.textMuted}
        />
      </View>

      {topics.length > 0 && (
        <FlatList
          horizontal
          data={[null, ...topics]}
          renderItem={({item: topic}) => (
            <TouchableOpacity
              style={[
                styles.topicChip,
                selectedTopic === topic && styles.topicChipActive,
              ]}
              onPress={() => setSelectedTopic(topic)}>
              <Text
                style={[
                  styles.topicChipText,
                  selectedTopic === topic && styles.topicChipTextActive,
                ]}>
                {topic || 'All'}
              </Text>
            </TouchableOpacity>
          )}
          keyExtractor={item => item || 'all'}
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.topicRow}
        />
      )}

      <FlatList
        data={items}
        renderItem={renderItem}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.list}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={styles.emptyIcon}>📚</Text>
            <Text style={styles.emptyText}>No knowledge items yet</Text>
          </View>
        }
      />

      <Modal visible={addModalVisible} animationType="slide" transparent>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Add Knowledge</Text>
            <TextInput
              style={styles.modalInput}
              value={formContent}
              onChangeText={setFormContent}
              placeholder="Content..."
              placeholderTextColor={colors.textMuted}
              multiline
              textAlignVertical="top"
            />
            <TextInput
              style={styles.modalInput}
              value={formTopic}
              onChangeText={setFormTopic}
              placeholder="Topic (optional)"
              placeholderTextColor={colors.textMuted}
            />
            <View style={styles.modalActions}>
              <TouchableOpacity
                style={styles.cancelBtn}
                onPress={() => setAddModalVisible(false)}>
                <Text style={styles.cancelText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.saveBtn} onPress={handleAdd}>
                <Text style={styles.saveText}>Add</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      <Modal visible={!!editItem} animationType="slide" transparent>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Edit Knowledge</Text>
            <TextInput
              style={styles.modalInput}
              value={formContent}
              onChangeText={setFormContent}
              placeholder="Content..."
              placeholderTextColor={colors.textMuted}
              multiline
              textAlignVertical="top"
            />
            <TextInput
              style={styles.modalInput}
              value={formTopic}
              onChangeText={setFormTopic}
              placeholder="Topic (optional)"
              placeholderTextColor={colors.textMuted}
            />
            <View style={styles.modalActions}>
              <TouchableOpacity
                style={styles.cancelBtn}
                onPress={() => setEditItem(null)}>
                <Text style={styles.cancelText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.saveBtn} onPress={handleEdit}>
                <Text style={styles.saveText}>Save</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: colors.background,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  title: {
    ...typography.h1,
    color: colors.text,
  },
  addBtn: {
    backgroundColor: colors.primary,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radii.md,
  },
  addBtnText: {
    ...typography.caption,
    color: colors.white,
    fontWeight: '600',
  },
  searchRow: {
    paddingHorizontal: spacing.lg,
    marginBottom: spacing.sm,
  },
  searchInput: {
    ...typography.body,
    color: colors.text,
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  topicRow: {
    paddingHorizontal: spacing.lg,
    gap: spacing.xs,
    marginBottom: spacing.sm,
  },
  topicChip: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: radii.full,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  topicChipActive: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  topicChipText: {
    ...typography.small,
    color: colors.textSecondary,
  },
  topicChipTextActive: {
    color: colors.white,
  },
  list: {
    padding: spacing.lg,
    gap: spacing.sm,
  },
  itemCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    padding: spacing.md,
  },
  itemHeader: {
    marginBottom: spacing.sm,
  },
  itemContent: {
    ...typography.body,
    color: colors.text,
  },
  itemFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  itemMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    flex: 1,
  },
  importanceDots: {
    flexDirection: 'row',
    gap: 3,
  },
  dot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: colors.border,
  },
  dotFilled: {
    backgroundColor: colors.accent,
  },
  itemActions: {
    flexDirection: 'row',
    gap: spacing.xs,
  },
  actionBtn: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 3,
    borderRadius: radii.sm,
  },
  actionText: {
    ...typography.small,
    color: colors.primary,
  },
  deleteBtn: {},
  deleteText: {
    color: colors.error,
  },
  empty: {
    alignItems: 'center',
    paddingVertical: spacing.xxxl * 2,
  },
  emptyIcon: {
    fontSize: 48,
    marginBottom: spacing.lg,
  },
  emptyText: {
    ...typography.body,
    color: colors.textMuted,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: colors.background,
    borderTopLeftRadius: radii.xl,
    borderTopRightRadius: radii.xl,
    padding: spacing.xl,
    gap: spacing.md,
  },
  modalTitle: {
    ...typography.h3,
    color: colors.text,
  },
  modalInput: {
    ...typography.body,
    color: colors.text,
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    minHeight: 80,
  },
  modalActions: {
    flexDirection: 'row',
    gap: spacing.sm,
    justifyContent: 'flex-end',
  },
  cancelBtn: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: radii.md,
  },
  cancelText: {
    ...typography.caption,
    color: colors.textSecondary,
    fontWeight: '600',
  },
  saveBtn: {
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.sm,
    borderRadius: radii.md,
    backgroundColor: colors.primary,
  },
  saveText: {
    ...typography.caption,
    color: colors.white,
    fontWeight: '600',
  },
});
