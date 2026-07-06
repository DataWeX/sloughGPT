import React, {useEffect, useState} from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Modal,
  FlatList,
  TextInput,
} from 'react-native';
import {
  getQuickPrompts,
  getQuickPromptsByCategory,
  addQuickPrompt,
  deleteQuickPrompt,
  type QuickPrompt,
} from '../services/quick-prompts';
import {triggerHaptic} from '../services/haptics';
import {colors, spacing, radii, typography} from '../theme';

const CATEGORIES = ['all', 'general', 'coding', 'writing', 'analysis', 'custom'] as const;

interface Props {
  visible: boolean;
  onClose: () => void;
  onSelect: (prompt: string) => void;
}

export function QuickPromptPicker({visible, onClose, onSelect}: Props) {
  const [prompts, setPrompts] = useState<QuickPrompt[]>([]);
  const [category, setCategory] = useState<string>('all');
  const [showAdd, setShowAdd] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newPrompt, setNewPrompt] = useState('');

  useEffect(() => {
    if (visible) {
      loadPrompts();
    }
  }, [visible, category]);

  const loadPrompts = async () => {
    const data = await getQuickPromptsByCategory(category);
    setPrompts(data);
  };

  const handleSelect = (prompt: QuickPrompt) => {
    triggerHaptic('light');
    onSelect(prompt.prompt);
    onClose();
  };

  const handleAdd = async () => {
    if (!newTitle.trim() || !newPrompt.trim()) return;
    await addQuickPrompt(newTitle.trim(), newPrompt.trim(), 'custom');
    setNewTitle('');
    setNewPrompt('');
    setShowAdd(false);
    await loadPrompts();
    triggerHaptic('success');
  };

  const handleDelete = async (id: string) => {
    await deleteQuickPrompt(id);
    await loadPrompts();
    triggerHaptic('light');
  };

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <View style={styles.overlay}>
        <View style={styles.container}>
          {/* Header */}
          <View style={styles.header}>
            <Text style={styles.title}>Quick Prompts</Text>
            <TouchableOpacity onPress={onClose}>
              <Text style={styles.closeBtn}>✕</Text>
            </TouchableOpacity>
          </View>

          {/* Category chips */}
          <FlatList
            horizontal
            showsHorizontalScrollIndicator={false}
            data={[...CATEGORIES]}
            keyExtractor={c => c}
            contentContainerStyle={styles.chipRow}
            renderItem={({item: c}) => (
              <TouchableOpacity
                style={[styles.chip, category === c && styles.chipActive]}
                onPress={() => setCategory(c)}>
                <Text style={[styles.chipText, category === c && styles.chipTextActive]}>
                  {c.charAt(0).toUpperCase() + c.slice(1)}
                </Text>
              </TouchableOpacity>
            )}
          />

          {/* Prompts list */}
          <FlatList
            data={prompts}
            keyExtractor={p => p.id}
            contentContainerStyle={styles.list}
            renderItem={({item: p}) => (
              <TouchableOpacity
                style={styles.promptItem}
                onPress={() => handleSelect(p)}
                onLongPress={() => handleDelete(p.id)}
                activeOpacity={0.7}>
                <View style={styles.promptHeader}>
                  <Text style={styles.promptTitle}>{p.title}</Text>
                  <Text style={styles.promptCategory}>{p.category}</Text>
                </View>
                <Text style={styles.promptPreview} numberOfLines={2}>
                  {p.prompt}
                </Text>
              </TouchableOpacity>
            )}
            ListEmptyComponent={
              <Text style={styles.empty}>No prompts yet. Tap + to add one.</Text>
            }
          />

          {/* Add button */}
          <TouchableOpacity
            style={styles.addBtn}
            onPress={() => setShowAdd(!showAdd)}>
            <Text style={styles.addBtnText}>{showAdd ? '✕ Cancel' : '+ New Prompt'}</Text>
          </TouchableOpacity>

          {/* Add form */}
          {showAdd && (
            <View style={styles.addForm}>
              <TextInput
                style={styles.input}
                placeholder="Title"
                placeholderTextColor={colors.textMuted}
                value={newTitle}
                onChangeText={setNewTitle}
              />
              <TextInput
                style={[styles.input, styles.inputMultiline]}
                placeholder="Prompt text (use {variable} for placeholders)"
                placeholderTextColor={colors.textMuted}
                value={newPrompt}
                onChangeText={setNewPrompt}
                multiline
                numberOfLines={3}
              />
              <TouchableOpacity
                style={[styles.saveBtn, (!newTitle.trim() || !newPrompt.trim()) && styles.saveBtnDisabled]}
                onPress={handleAdd}
                disabled={!newTitle.trim() || !newPrompt.trim()}>
                <Text style={styles.saveBtnText}>Save</Text>
              </TouchableOpacity>
            </View>
          )}
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  container: {
    backgroundColor: colors.background,
    borderTopLeftRadius: radii.xl,
    borderTopRightRadius: radii.xl,
    maxHeight: '80%',
    paddingBottom: 34,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.lg,
    paddingBottom: spacing.sm,
  },
  title: {
    ...typography.h2,
    color: colors.text,
  },
  closeBtn: {
    fontSize: 20,
    color: colors.textMuted,
    padding: spacing.xs,
  },
  chipRow: {
    paddingHorizontal: spacing.lg,
    gap: spacing.xs,
    paddingBottom: spacing.sm,
  },
  chip: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: radii.full,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  chipActive: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  chipText: {
    ...typography.small,
    color: colors.textSecondary,
  },
  chipTextActive: {
    color: colors.white,
  },
  list: {
    paddingHorizontal: spacing.lg,
    gap: spacing.xs,
  },
  promptItem: {
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border,
  },
  promptHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  promptTitle: {
    ...typography.body,
    color: colors.text,
    fontWeight: '600',
  },
  promptCategory: {
    ...typography.small,
    color: colors.textMuted,
    textTransform: 'capitalize',
  },
  promptPreview: {
    ...typography.small,
    color: colors.textSecondary,
    lineHeight: 18,
  },
  empty: {
    ...typography.body,
    color: colors.textMuted,
    textAlign: 'center',
    paddingVertical: spacing.xxl,
  },
  addBtn: {
    marginHorizontal: spacing.lg,
    marginTop: spacing.sm,
    padding: spacing.md,
    borderRadius: radii.md,
    backgroundColor: colors.primary + '15',
    alignItems: 'center',
  },
  addBtnText: {
    ...typography.body,
    color: colors.primary,
    fontWeight: '600',
  },
  addForm: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    gap: spacing.sm,
  },
  input: {
    ...typography.body,
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm + 2,
    color: colors.text,
    borderWidth: 1,
    borderColor: colors.border,
  },
  inputMultiline: {
    minHeight: 80,
    textAlignVertical: 'top',
  },
  saveBtn: {
    padding: spacing.md,
    borderRadius: radii.md,
    backgroundColor: colors.primary,
    alignItems: 'center',
  },
  saveBtnDisabled: {
    opacity: 0.5,
  },
  saveBtnText: {
    ...typography.body,
    color: colors.white,
    fontWeight: '600',
  },
});
