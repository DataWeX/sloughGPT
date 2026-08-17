import React, {useMemo, useState, useCallback} from 'react';
import {Modal, FlatList, Pressable} from 'react-native';
import {YStack, XStack, Text, useTheme} from 'tamagui';
import {Icon} from './Icon';
import {triggerHaptic} from '../services/haptics';
import {getAllCommands} from '../services/chat-commands';
import type {ChatCommand} from '../services/chat-commands';

interface Props {
  visible: boolean;
  query: string;
  onSelect: (command: string) => void;
  onExecute: (command: ChatCommand, args: string[]) => void;
  onClose: () => void;
}

function fuzzyScore(query: string, target: string): number {
  const q = query.toLowerCase();
  const t = target.toLowerCase();
  let qi = 0;
  let score = 0;
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] === q[qi]) {
      score += 1 + (qi === 0 ? 5 : 0);
      qi++;
    }
  }
  return qi === q.length ? score : -1;
}

export function SlashCommandPicker({visible, query, onSelect, onExecute, onClose}: Props) {
  const theme = useTheme();
  const allCommands = useMemo(() => getAllCommands(), []);
  const [selectedIndex, setSelectedIndex] = useState(0);

  const filtered = useMemo(() => {
    const q = query.startsWith('/') ? query.slice(1) : query;
    const firstWord = q.split(/\s+/)[0] || '';

    if (!q) {
      return allCommands.map(c => ({command: c, score: 0}));
    }
    return allCommands
      .map(c => {
        const nameScore = fuzzyScore(firstWord, c.command.slice(1));
        const descScore = fuzzyScore(q, c.description);
        const score = Math.max(nameScore, descScore ?? -1);
        return {command: c, score};
      })
      .filter(x => x.score >= 0)
      .sort((a, b) => b.score - a.score);
  }, [query, allCommands]);

  const handleSelect = useCallback(
    (item: {command: ChatCommand}) => {
      const args = query
        .trim()
        .split(/\s+/)
        .slice(1)
        .filter(Boolean);
      triggerHaptic('light');
      onExecute(item.command, args);
      onClose();
    },
    [query, onExecute, onClose],
  );

  const bg = theme.background?.val || '#FFFFFF';
  const border = theme.borderColor?.val || '#E4E0F2';
  const primary = theme.color9?.val || '#7C52C4';
  const textMuted = theme.color10?.val || '#827A96';
  const textColor = theme.color?.val || '#1A1625';

  if (!visible || filtered.length === 0) return null;

  return (
    <Modal transparent animationType="fade" visible={visible} onRequestClose={onClose}>
      <Pressable style={{flex: 1, justifyContent: 'flex-end', backgroundColor: 'rgba(0,0,0,0.3)'}} onPress={onClose}>
        <YStack
          backgroundColor={bg}
          borderTopLeftRadius={16}
          borderTopRightRadius={16}
          maxHeight="60%"
          paddingBottom={20}
          borderWidth={0.5}
          borderColor={border}
          borderBottomWidth={0}>
          <XStack padding={16} paddingBottom={8} alignItems="center" justifyContent="space-between">
            <Text fontSize={13} fontWeight="600" color={textMuted} letterSpacing={0.5} textTransform="uppercase">
              Commands
            </Text>
            <Pressable onPress={onClose} hitSlop={8}>
              <Icon name="x" size={18} color={textMuted} />
            </Pressable>
          </XStack>

          <FlatList
            data={filtered}
            keyExtractor={item => item.command.command}
            keyboardShouldPersistTaps="handled"
            renderItem={({item, index}) => {
              const isActive = index === selectedIndex;
              return (
                <Pressable
                  onPress={() => handleSelect(item)}
                  onPressIn={() => setSelectedIndex(index)}
                  style={{
                    paddingHorizontal: 16,
                    paddingVertical: 12,
                    flexDirection: 'row',
                    alignItems: 'center',
                    gap: 12,
                    backgroundColor: isActive ? 'rgba(124, 82, 196, 0.08)' : 'transparent',
                  }}>
                  <Text
                    style={{
                      fontSize: 14,
                      fontFamily: 'JetBrainsMono-Regular',
                      fontWeight: '500',
                      color: primary,
                      minWidth: 80,
                    }}>
                    {item.command.command}
                  </Text>
                  <YStack flex={1}>
                    <Text fontSize={13} color={textColor}>
                      {item.command.description}
                    </Text>
                    {item.command.usage !== item.command.command && (
                      <Text fontSize={11} color={textMuted} marginTop={2}>
                        {item.command.usage}
                      </Text>
                    )}
                  </YStack>
                </Pressable>
              );
            }}
          />
        </YStack>
      </Pressable>
    </Modal>
  );
}
