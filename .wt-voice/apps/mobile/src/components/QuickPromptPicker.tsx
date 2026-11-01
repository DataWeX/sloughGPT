import React, {useEffect, useState} from 'react';
import {
  Pressable,
  Modal,
  FlatList,
  TextInput,
} from 'react-native';
import {YStack, XStack, Text} from 'tamagui';
import {TamaguiProvider} from '../theme/TamaguiProvider';
import {useColors} from '../theme/colors';
import {
  getQuickPromptsByCategory,
  addQuickPrompt,
  deleteQuickPrompt,
  type QuickPrompt,
} from '../services/quick-prompts';
import {triggerHaptic} from '../services/haptics';
import {Icon} from './Icon';

const CATEGORIES = ['all', 'general', 'coding', 'writing', 'analysis', 'custom'] as const;

interface Props {
  visible: boolean;
  onClose: () => void;
  onSelect: (prompt: string) => void;
}

export function QuickPromptPicker({visible, onClose, onSelect}: Props) {
  const colors = useColors();
  const [prompts, setPrompts] = useState<QuickPrompt[]>([]);
  const [category, setCategory] = useState<string>('all');
  const [showAdd, setShowAdd] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newPrompt, setNewPrompt] = useState('');

  const accent = colors.primary;

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
      <TamaguiProvider>
      <YStack flex={1} backgroundColor={colors.overlay(0.5)} justifyContent="flex-end">
        <YStack
          backgroundColor={colors.background}
          borderTopLeftRadius={24}
          borderTopRightRadius={24}
          maxHeight="80%"
          paddingBottom={34}>
          <XStack
            justifyContent="space-between"
            alignItems="center"
            paddingHorizontal={20}
            paddingTop={14}
            paddingBottom={8}>
            <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color={colors.text}>
              Quick Prompts
            </Text>
            <Pressable onPress={onClose} accessible accessibilityRole="button" accessibilityLabel="Close quick prompts">
              <YStack width={28} height={28} borderRadius={9} alignItems="center" justifyContent="center">
                <Icon name="x" size={16} color={colors.textSecondary} />
              </YStack>
            </Pressable>
          </XStack>

          {/* Category chips */}
          <FlatList
            horizontal
            showsHorizontalScrollIndicator={false}
            data={[...CATEGORIES]}
            keyExtractor={c => c}
            contentContainerStyle={{paddingHorizontal: 16, gap: 4, paddingBottom: 8}}
            renderItem={({item: c}) => (
              <Pressable onPress={() => setCategory(c)}>
                <YStack
                  paddingHorizontal={12}
                  paddingVertical={5}
                  borderRadius={999}
                  backgroundColor={category === c ? accent : colors.background}
                  borderWidth={0.5}
                  borderColor={category === c ? accent : colors.border}>
                  <Text
                    fontSize={11}
                    fontWeight="500"
                    letterSpacing={0.2}
                    color={category === c ? 'white' : colors.textMuted}>
                    {c.charAt(0).toUpperCase() + c.slice(1)}
                  </Text>
                </YStack>
              </Pressable>
            )}
          />

          {/* Prompts list */}
          <FlatList
            data={prompts}
            keyExtractor={p => p.id}
            contentContainerStyle={{paddingHorizontal: 16, gap: 4}}
            renderItem={({item: p}) => (
              <Pressable onPress={() => handleSelect(p)} onLongPress={() => handleDelete(p.id)}>
                <YStack backgroundColor={colors.background} borderRadius={10} padding={12} borderWidth={0.5} borderColor={colors.border}>
                  <XStack justifyContent="space-between" alignItems="center" marginBottom={4}>
                    <Text fontSize={15} fontWeight="600" color={colors.text}>
                      {p.title}
                    </Text>
                    <Text fontSize={11} fontWeight="500" letterSpacing={0.2} color={colors.textSecondary} textTransform="capitalize">
                      {p.category}
                    </Text>
                  </XStack>
                  <Text
                    fontSize={11}
                    fontWeight="500"
                    letterSpacing={0.2}
                    color={colors.textMuted}
                    lineHeight={18}
                    numberOfLines={2}>
                    {p.prompt}
                  </Text>
                </YStack>
              </Pressable>
            )}
            ListEmptyComponent={
              <Text fontSize={15} fontWeight="400" color={colors.textSecondary} textAlign="center" paddingVertical={24}>
                No prompts yet. Tap + to add one.
              </Text>
            }
          />

          {/* Add button */}
          <Pressable onPress={() => setShowAdd(!showAdd)}>
            <YStack marginHorizontal={16} marginTop={8} padding={12} borderRadius={10} backgroundColor={accent} alignItems="center">
              <XStack alignItems="center" gap={6}>
                <Icon name={showAdd ? 'x' : 'plus'} size={16} color="white" />
                <Text fontSize={15} fontWeight="600" color="white">
                  {showAdd ? 'Cancel' : 'New Prompt'}
                </Text>
              </XStack>
            </YStack>
          </Pressable>

          {/* Add form */}
          {showAdd && (
            <YStack paddingHorizontal={16} paddingTop={8} gap={8}>
              <TextInput
                placeholder="Title"
                placeholderTextColor={colors.textMuted}
                value={newTitle}
                onChangeText={setNewTitle}
                style={{
                  fontSize: 14,
                  backgroundColor: colors.primaryAlpha(0.04),
                  borderRadius: 10,
                  paddingHorizontal: 14,
                  paddingVertical: 10,
                  color: colors.text,
                  borderWidth: 0.5,
                  borderColor: colors.primaryAlpha(0.12),
                }}
              />
              <TextInput
                placeholder="Prompt text (use {variable} for placeholders)"
                placeholderTextColor={colors.textMuted}
                value={newPrompt}
                onChangeText={setNewPrompt}
                multiline
                numberOfLines={3}
                style={{
                  fontSize: 14,
                  backgroundColor: colors.primaryAlpha(0.04),
                  borderRadius: 10,
                  paddingHorizontal: 14,
                  paddingVertical: 10,
                  color: colors.text,
                  borderWidth: 0.5,
                  borderColor: colors.primaryAlpha(0.12),
                  minHeight: 80,
                  textAlignVertical: 'top',
                }}
              />
              <Pressable onPress={handleAdd} disabled={!newTitle.trim() || !newPrompt.trim()}>
                <YStack
                  padding={12}
                  borderRadius={10}
                  backgroundColor={accent}
                  alignItems="center"
                  opacity={(!newTitle.trim() || !newPrompt.trim()) ? 0.5 : 1}>
                  <Text fontSize={15} fontWeight="600" color="white">Save</Text>
                </YStack>
              </Pressable>
            </YStack>
          )}
        </YStack>
      </YStack>
      </TamaguiProvider>
    </Modal>
  );
}
