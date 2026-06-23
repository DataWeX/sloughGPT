import React, {useEffect, useState, useCallback, useRef} from 'react';
import {
  View,
  FlatList,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  RefreshControl,
  Modal,
  Keyboard,
  Share,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {api} from '../services/api-client';
import {StatusBadge} from '../components/StatusBadge';
import {colors, spacing, radii, typography} from '../theme';
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
      await Share.share({
        title: 'Knowledge Export',
        message: json,
      });
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
          api.post('/knowledge', {
            content: line.trim(),
            topic: formTopic.trim() || undefined,
          }),
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
      <TouchableOpacity
        style={[styles.itemCard, isSelected && styles.itemCardSelected]}
        onLongPress={() => {
          setSelectMode(true);
          setSelectedIds(new Set([item.id]));
        }}
        onPress={() => {
          if (selectMode) {
            toggleSelect(item.id);
          } else {
            openEdit(item);
          }
        }}
        activeOpacity={0.7}>
        <View style={styles.itemHeader}>
          {selectMode && (
            <View style={[styles.checkbox, isSelected && styles.checkboxActive]}>
              {isSelected && <View style={styles.checkInner} />}
            </View>
          )}
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
                  style={[styles.dot, i < item.importance && styles.dotFilled]}
                />
              ))}
            </View>
          </View>
          {!selectMode && (
            <View style={styles.itemActions}>
              <TouchableOpacity
                style={styles.actionBtn}
                onPress={() => openEdit(item)}>
                <StatusBadge label="Edit" variant="info" />
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.actionBtn}
                onPress={() => handleDelete(item.id)}>
                <StatusBadge label="Del" variant="error" />
              </TouchableOpacity>
            </View>
          )}
        </View>
      </TouchableOpacity>
    );
  };

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <View style={styles.header}>
        {selectMode ? (
          <View style={styles.selectHeader}>
            <TouchableOpacity onPress={() => { setSelectMode(false); setSelectedIds(new Set()); }}>
              <Text style={styles.cancelText}>Cancel</Text>
            </TouchableOpacity>
            <Text style={styles.selectCount}>{selectedIds.size} selected</Text>
            <TouchableOpacity
              onPress={handleBatchDelete}
              disabled={selectedIds.size === 0}>
              <Text style={[styles.deleteText, selectedIds.size === 0 && styles.disabledText]}>
                Delete ({selectedIds.size})
              </Text>
            </TouchableOpacity>
          </View>
        ) : (
          <>
            <Text style={styles.title}>Knowledge</Text>
            <View style={styles.headerActions}>
              {items.length > 0 && (
                <TouchableOpacity style={styles.exportBtn} onPress={handleExport}>
                  <Text style={styles.exportBtnText}>Export</Text>
                </TouchableOpacity>
              )}
              <TouchableOpacity
                style={styles.importBtn}
                onPress={() => { setImportText(''); setImportModalVisible(true); }}>
                <Text style={styles.importBtnText}>Import</Text>
              </TouchableOpacity>
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
          </>
        )}
      </View>

      {!selectMode && (
        <View style={styles.searchRow}>
          <TextInput
            style={styles.searchInput}
            value={search}
            onChangeText={debouncedSearch}
            placeholder="Search knowledge..."
            placeholderTextColor={colors.textMuted}
            returnKeyType="search"
            onSubmitEditing={() => Keyboard.dismiss()}
          />
        </View>
      )}

      {!selectMode && topics.length > 0 && (
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
        keyboardDismissMode="on-drag"
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
        <TouchableOpacity
          style={styles.modalOverlay}
          activeOpacity={1}
          onPress={() => setAddModalVisible(false)}>
          <TouchableOpacity activeOpacity={1} style={styles.modalContent} onPress={() => {}}>
            <Text style={styles.modalTitle}>Add Knowledge</Text>
            <TextInput
              style={styles.modalInput}
              value={formContent}
              onChangeText={setFormContent}
              placeholder="Content..."
              placeholderTextColor={colors.textMuted}
              multiline
              textAlignVertical="top"
              autoFocus
            />
            <TextInput
              style={styles.modalInputShort}
              value={formTopic}
              onChangeText={setFormTopic}
              placeholder="Topic (optional)"
              placeholderTextColor={colors.textMuted}
              returnKeyType="done"
            />
            <View style={styles.modalActions}>
              <TouchableOpacity style={styles.cancelBtn} onPress={() => setAddModalVisible(false)}>
                <Text style={styles.cancelBtnText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.saveBtn, !formContent.trim() && styles.saveBtnDisabled]}
                onPress={handleAdd}
                disabled={!formContent.trim()}>
                <Text style={styles.saveText}>Add</Text>
              </TouchableOpacity>
            </View>
          </TouchableOpacity>
        </TouchableOpacity>
      </Modal>

      <Modal visible={!!editItem} animationType="slide" transparent>
        <TouchableOpacity
          style={styles.modalOverlay}
          activeOpacity={1}
          onPress={() => setEditItem(null)}>
          <TouchableOpacity activeOpacity={1} style={styles.modalContent} onPress={() => {}}>
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
              style={styles.modalInputShort}
              value={formTopic}
              onChangeText={setFormTopic}
              placeholder="Topic (optional)"
              placeholderTextColor={colors.textMuted}
              returnKeyType="done"
            />
            <View style={styles.modalActions}>
              <TouchableOpacity style={styles.cancelBtn} onPress={() => setEditItem(null)}>
                <Text style={styles.cancelBtnText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.saveBtn, !formContent.trim() && styles.saveBtnDisabled]}
                onPress={handleEdit}
                disabled={!formContent.trim()}>
                <Text style={styles.saveText}>Save</Text>
              </TouchableOpacity>
            </View>
          </TouchableOpacity>
        </TouchableOpacity>
      </Modal>

      {/* Import modal */}
      <Modal visible={importModalVisible} animationType="slide" transparent>
        <TouchableOpacity
          style={styles.modalOverlay}
          activeOpacity={1}
          onPress={() => setImportModalVisible(false)}>
          <TouchableOpacity activeOpacity={1} style={styles.modalContent} onPress={() => {}}>
            <Text style={styles.modalTitle}>Import Knowledge</Text>
            <Text style={styles.importHint}>
              Paste one item per line. Each line becomes a knowledge entry.
            </Text>
            <TextInput
              style={styles.modalInput}
              value={importText}
              onChangeText={setImportText}
              placeholder="Line 1\nLine 2\nLine 3..."
              placeholderTextColor={colors.textMuted}
              multiline
              textAlignVertical="top"
              autoFocus
            />
            <TextInput
              style={styles.modalInputShort}
              value={formTopic}
              onChangeText={setFormTopic}
              placeholder="Topic for all (optional)"
              placeholderTextColor={colors.textMuted}
              returnKeyType="done"
            />
            <View style={styles.modalActions}>
              <TouchableOpacity style={styles.cancelBtn} onPress={() => setImportModalVisible(false)}>
                <Text style={styles.cancelBtnText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.saveBtn, (!importText.trim() || importing) && styles.saveBtnDisabled]}
                onPress={handleImport}
                disabled={!importText.trim() || importing}>
                <Text style={styles.saveText}>
                  {importing ? 'Importing...' : `Import (${importText.split('\n').filter(l => l.trim().length > 2).length})`}
                </Text>
              </TouchableOpacity>
            </View>
          </TouchableOpacity>
        </TouchableOpacity>
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
  headerActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  title: {
    ...typography.h1,
    color: colors.text,
  },
  importBtn: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radii.md,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  importBtnText: {
    ...typography.caption,
    color: colors.textSecondary,
    fontWeight: '600',
  },
  exportBtn: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radii.md,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  exportBtnText: {
    ...typography.caption,
    color: colors.textSecondary,
    fontWeight: '600',
  },
  importHint: {
    ...typography.caption,
    color: colors.textMuted,
    marginBottom: spacing.xs,
  },
  selectHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    flex: 1,
  },
  selectCount: {
    ...typography.body,
    color: colors.text,
    fontWeight: '600',
  },
  cancelText: {
    ...typography.caption,
    color: colors.primary,
    fontWeight: '600',
  },
  deleteText: {
    ...typography.caption,
    color: colors.error,
    fontWeight: '600',
  },
  disabledText: {
    opacity: 0.4,
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
  itemCardSelected: {
    backgroundColor: colors.primary + '10',
    borderWidth: 1,
    borderColor: colors.primary,
  },
  itemHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.sm,
    marginBottom: spacing.sm,
  },
  checkbox: {
    width: 20,
    height: 20,
    borderRadius: radii.sm,
    borderWidth: 2,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 2,
  },
  checkboxActive: {
    borderColor: colors.primary,
    backgroundColor: colors.primary,
  },
  checkInner: {
    width: 10,
    height: 10,
    borderRadius: 2,
    backgroundColor: colors.white,
  },
  itemContent: {
    ...typography.body,
    color: colors.text,
    flex: 1,
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
  actionBtn: {},
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
    minHeight: 100,
  },
  modalInputShort: {
    ...typography.body,
    color: colors.text,
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
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
  cancelBtnText: {
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
  saveBtnDisabled: {
    opacity: 0.4,
  },
  saveText: {
    ...typography.caption,
    color: colors.white,
    fontWeight: '600',
  },
});
