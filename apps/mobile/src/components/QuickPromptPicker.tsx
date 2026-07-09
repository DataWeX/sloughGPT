import React, {useEffect, useState} from 'react';
import {
  TouchableOpacity,
  Modal,
  FlatList,
  TextInput,
} from 'react-native';
import {YStack, XStack, Text} from 'tamagui';
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
      <YStack flex={1} backgroundColor="rgba(0,0,0,0.5)" justifyContent="flex-end">
        <YStack
          backgroundColor="$background"
          borderTopLeftRadius={16}
          borderTopRightRadius={16}
          maxHeight="80%"
          paddingBottom={34}>
          {/* Header */}
          <XStack
            justifyContent="space-between"
            alignItems="center"
            paddingHorizontal={16}
            paddingTop={16}
            paddingBottom={8}>
            <Text fontSize={20} fontWeight="600" letterSpacing={-0.2} color="$color">
              Quick Prompts
            </Text>
            <TouchableOpacity onPress={onClose} style={{padding: 4}}>
              <Icon name="x" size={20} color="$color10" />
            </TouchableOpacity>
          </XStack>

          {/* Category chips */}
          <FlatList
            horizontal
            showsHorizontalScrollIndicator={false}
            data={[...CATEGORIES]}
            keyExtractor={c => c}
            contentContainerStyle={{paddingHorizontal: 16, gap: 4, paddingBottom: 8}}
            renderItem={({item: c}) => (
              <TouchableOpacity
                onPress={() => setCategory(c)}
                style={{
                  paddingHorizontal: 12,
                  paddingVertical: 4,
                  borderRadius: 9999,
                  backgroundColor: category === c ? '$color9' : 'white',
                  borderWidth: 1,
                  borderColor: '$borderColor',
                }}>
                <Text
                  fontSize={11}
                  fontWeight="500"
                  letterSpacing={0.2}
                  color={category === c ? 'white' : '$color11'}>
                  {c.charAt(0).toUpperCase() + c.slice(1)}
                </Text>
              </TouchableOpacity>
            )}
          />

          {/* Prompts list */}
          <FlatList
            data={prompts}
            keyExtractor={p => p.id}
            contentContainerStyle={{paddingHorizontal: 16, gap: 4}}
            renderItem={({item: p}) => (
              <TouchableOpacity
                onPress={() => handleSelect(p)}
                onLongPress={() => handleDelete(p.id)}
                activeOpacity={0.7}
                style={{
                  backgroundColor: 'white',
                  borderRadius: 8,
                  padding: 12,
                  borderWidth: 1,
                  borderColor: '$borderColor',
                }}>
                <XStack justifyContent="space-between" alignItems="center" marginBottom={4}>
                  <Text fontSize={15} fontWeight="600" color="$color">
                    {p.title}
                  </Text>
                  <Text fontSize={11} fontWeight="500" letterSpacing={0.2} color="$color10" textTransform="capitalize">
                    {p.category}
                  </Text>
                </XStack>
                <Text
                  fontSize={11}
                  fontWeight="500"
                  letterSpacing={0.2}
                  color="$color11"
                  lineHeight={18}
                  numberOfLines={2}>
                  {p.prompt}
                </Text>
              </TouchableOpacity>
            )}
            ListEmptyComponent={
              <Text fontSize={15} fontWeight="400" color="$color10" textAlign="center" paddingVertical={24}>
                No prompts yet. Tap + to add one.
              </Text>
            }
          />

          {/* Add button */}
          <TouchableOpacity
            onPress={() => setShowAdd(!showAdd)}
            style={{
              marginHorizontal: 16,
              marginTop: 8,
              padding: 12,
              borderRadius: 8,
              backgroundColor: '$color9',
              alignItems: 'center',
            }}>
            <XStack alignItems="center" gap={6}>
              <Icon name={showAdd ? 'x' : 'plus'} size={16} color="white" />
              <Text fontSize={15} fontWeight="600" color="white">
                {showAdd ? 'Cancel' : 'New Prompt'}
              </Text>
            </XStack>
          </TouchableOpacity>

          {/* Add form */}
          {showAdd && (
            <YStack paddingHorizontal={16} paddingTop={8} gap={8}>
              <TextInput
                placeholder="Title"
                placeholderTextColor="$color10"
                value={newTitle}
                onChangeText={setNewTitle}
                style={{
                  fontSize: 15,
                  backgroundColor: 'white',
                  borderRadius: 8,
                  paddingHorizontal: 12,
                  paddingVertical: 10,
                  color: '$color',
                  borderWidth: 1,
                  borderColor: '$borderColor',
                }}
              />
              <TextInput
                placeholder="Prompt text (use {variable} for placeholders)"
                placeholderTextColor="$color10"
                value={newPrompt}
                onChangeText={setNewPrompt}
                multiline
                numberOfLines={3}
                style={{
                  fontSize: 15,
                  backgroundColor: 'white',
                  borderRadius: 8,
                  paddingHorizontal: 12,
                  paddingVertical: 10,
                  color: '$color',
                  borderWidth: 1,
                  borderColor: '$borderColor',
                  minHeight: 80,
                  textAlignVertical: 'top',
                }}
              />
              <TouchableOpacity
                onPress={handleAdd}
                disabled={!newTitle.trim() || !newPrompt.trim()}
                style={{
                  padding: 12,
                  borderRadius: 8,
                  backgroundColor: '$color9',
                  alignItems: 'center',
                  opacity: (!newTitle.trim() || !newPrompt.trim()) ? 0.5 : 1,
                }}>
                <Text fontSize={15} fontWeight="600" color="white">
                  Save
                </Text>
              </TouchableOpacity>
            </YStack>
          )}
        </YStack>
      </YStack>
    </Modal>
  );
}
