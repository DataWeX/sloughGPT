import React from 'react';
import {FlatList, Alert, TextInput as RNTextInput, Modal} from 'react-native';
import {YStack, XStack, Text} from 'tamagui';
import {TamaguiProvider} from '../theme/TamaguiProvider';
import {useColors} from '../theme/colors';
import {useChatStore} from '../stores/chat-store';
import {triggerHaptic} from '../services/haptics';
import {toast} from '../services/toast';
import * as starsService from '../services/stars';
import * as labelsService from '../services/labels';
import {Icon} from './Icon';
import type {Session} from '../types';
import {Platform} from 'react-native';

interface ChatDrawerProps {
  visible: boolean;
  onClose: () => void;
  sortedActiveSessions: Session[];
  archivedSessions: Session[];
  sessionLabels: Record<string, string[]>;
  allLabels: string[];
  labelFilter: string | null;
  setLabelFilter: (f: string | null) => void;
  showArchived: boolean;
  setShowArchived: (v: boolean) => void;
  labelInput: string;
  setLabelInput: (v: string) => void;
  starredIds: string[];
  setStarredIds: (fn: (prev: string[]) => string[]) => void;
  setSessionLabels: (fn: (prev: Record<string, string[]>) => Record<string, string[]>) => void;
  setAllLabels: (v: string[]) => void;
}

export function ChatDrawer({
  visible,
  onClose,
  sortedActiveSessions,
  archivedSessions,
  sessionLabels,
  allLabels,
  labelFilter,
  setLabelFilter,
  showArchived,
  setShowArchived,
  labelInput,
  setLabelInput,
  starredIds,
  setStarredIds,
  setSessionLabels,
  setAllLabels,
}: ChatDrawerProps) {
  const colors = useColors();
  const {loadSession, deleteSession, renameSession, archiveSession} = useChatStore();

  return (
    <Modal visible={visible} animationType="slide" transparent>
      <TamaguiProvider>
        <YStack flex={1} justifyContent="flex-end">
          <YStack
            flex={1}
            backgroundColor={colors.overlay(0.3)}
            onPress={onClose}
          />
          <YStack
            backgroundColor="$background"
            borderTopLeftRadius={24}
            borderTopRightRadius={24}
            maxHeight="85%"
            overflow="hidden">
            {/* Handle bar */}
            <YStack alignItems="center" paddingTop={10} paddingBottom={4}>
              <YStack width={40} height={5} borderRadius={3} backgroundColor="$borderColor" opacity={0.5} />
            </YStack>

            <XStack
              paddingHorizontal={20} paddingVertical={14}
              paddingBottom={12}
              borderBottomWidth={0.5} borderBottomColor="$borderColor"
              alignItems="center" justifyContent="space-between">
              <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color="$color">Conversations</Text>
              <YStack
                width={32} height={32} borderRadius={10}
                alignItems="center" justifyContent="center"
                onPress={onClose}
                pressStyle={{opacity: 0.6}}>
                <Icon name="x" size={16} color={colors.textSecondary} />
              </YStack>
            </XStack>

            {/* Session list */}
            <FlatList
              data={labelFilter ? sortedActiveSessions.filter(s => (sessionLabels[s.id] || []).includes(labelFilter!)) : sortedActiveSessions}
              keyExtractor={item => item.id}
              renderItem={({item: session}) => {
                const isStarred = starredIds.includes(session.id);
                const isActive = session.id === useChatStore.getState().activeSessionId;
                return (
                <XStack
                  paddingHorizontal={16} paddingVertical={12}
                  marginHorizontal={12} marginVertical={2}
                  borderRadius={12}
                  alignItems="center" justifyContent="space-between"
                  backgroundColor={isActive ? colors.primaryAlpha(0.08) : 'transparent'}
                  onPress={() => {
                    loadSession(session.id);
                    onClose();
                  }}
                  pressStyle={{backgroundColor: colors.primaryAlpha(0.06), scale: 0.98}}>
                  <YStack flex={1} marginRight={8}>
                    <YStack
                      onLongPress={() => {
                        triggerHaptic('light');
                        const currentTitle = session.name || 'New conversation';
                        Alert.prompt('Rename', 'Enter a new title:', (newTitle: string) => {
                          if (newTitle && newTitle.trim() && newTitle.trim() !== currentTitle) {
                            renameSession(session.id, newTitle.trim());
                          }
                        }, 'plain-text', currentTitle);
                      }}>
                      <XStack alignItems="center" gap={6}>
                        {isStarred && <Icon name="star" size={12} color={colors.warning} />}
                        <Text fontSize={14} fontWeight={isActive ? '600' : '400'} color="$color" numberOfLines={1}>
                          {session.name || 'New conversation'}
                        </Text>
                      </XStack>
                    </YStack>
                    <XStack alignItems="center" gap={6} marginTop={3}>
                      <Text fontSize={11} color="$color10">
                        {session.message_count || 0} messages
                      </Text>
                      {(sessionLabels[session.id] || []).length > 0 && (
                        <XStack flexWrap="wrap" gap={3}>
                          {(sessionLabels[session.id] || []).slice(0, 2).map(label => (
                            <YStack key={label} backgroundColor="$color9" opacity={0.12} paddingHorizontal={5} paddingVertical={1} borderRadius={4}>
                              <Text fontSize={9} color="$color9">{label}</Text>
                            </YStack>
                          ))}
                        </XStack>
                      )}
                    </XStack>
                  </YStack>
                  <XStack alignItems="center" gap={2}>
                    <YStack
                      width={28} height={28} borderRadius={8}
                      alignItems="center" justifyContent="center"
                      onPress={async () => {
                        try {
                          if (isStarred) {
                            await starsService.unstarSession(session.id);
                            setStarredIds(prev => prev.filter(id => id !== session.id));
                          } else {
                            await starsService.starSession(session.id);
                            setStarredIds(prev => [session.id, ...prev]);
                          }
                          triggerHaptic('light');
                        } catch (e) {
                          toast.error('Failed to update star');
                        }
                      }}
                      pressStyle={{opacity: 0.6}}>
                      <Icon name={isStarred ? 'star' : 'star-outline'} size={14} color={colors.warning} />
                    </YStack>
                    <YStack
                      width={28} height={28} borderRadius={8}
                      alignItems="center" justifyContent="center"
                      onPress={() => {
                        Alert.alert('Delete', 'Delete this conversation?', [
                          {text: 'Cancel', style: 'cancel'},
                          {
                            text: 'Delete',
                            style: 'destructive',
                            onPress: () => deleteSession(session.id),
                          },
                        ]);
                      }}
                      pressStyle={{opacity: 0.6}}>
                      <Icon name="trash-2" size={14} color={colors.error} />
                    </YStack>
                  </XStack>
                </XStack>
              )}}
              ListEmptyComponent={
                <YStack padding={40} alignItems="center">
                  <Icon name="message-circle" size={24} color={colors.textMuted} />
                  <Text fontSize={13} color="$color10" marginTop={8}>No conversations yet</Text>
                </YStack>
              }
              removeClippedSubviews
              maxToRenderPerBatch={10}
              windowSize={11}
            />
            {archivedSessions.length > 0 && (
              <XStack
                paddingHorizontal={16} paddingVertical={10}
                borderTopWidth={0.5} borderTopColor="$borderColor"
                alignItems="center" justifyContent="space-between"
                onPress={() => setShowArchived(!showArchived)}
                pressStyle={{backgroundColor: colors.primaryAlpha(0.04)}}>
                <XStack alignItems="center" gap={8}>
                  <YStack width={24} height={24} borderRadius={8} backgroundColor={colors.primaryAlpha(0.08)} alignItems="center" justifyContent="center">
                    <Icon name="archive" size={12} color="$color9" />
                  </YStack>
                  <Text fontSize={13} fontWeight="500" color="$color">Archived</Text>
                  <YStack paddingHorizontal={6} paddingVertical={2} borderRadius={6} backgroundColor={colors.primaryAlpha(0.08)}>
                    <Text fontSize={10} fontWeight="600" color="$color9">{archivedSessions.length}</Text>
                  </YStack>
                </XStack>
                <Text fontSize={12} color="$color10">{showArchived ? 'Hide' : 'Show'}</Text>
              </XStack>
            )}
            {showArchived && archivedSessions.length > 0 && (
              <YStack maxHeight={200}>
                <FlatList
                  data={archivedSessions}
                  keyExtractor={item => item.id}
                  renderItem={({item: session}) => (
                    <XStack
                      paddingHorizontal={16} paddingVertical={10}
                      marginHorizontal={12} marginVertical={1}
                      borderRadius={10}
                      alignItems="center" justifyContent="space-between"
                      backgroundColor={session.id === useChatStore.getState().activeSessionId ? colors.primaryAlpha(0.08) : 'transparent'}
                      onPress={() => {
                        loadSession(session.id);
                        onClose();
                      }}>
                      <YStack flex={1}>
                        <Text fontSize={13} fontWeight="400" color="$color" numberOfLines={1}>
                          {session.name || 'New conversation'}
                        </Text>
                        <Text fontSize={11} color="$color10">
                          {session.message_count || 0} messages
                        </Text>
                      </YStack>
                      <YStack
                        paddingHorizontal={10} paddingVertical={4} borderRadius={8}
                        backgroundColor={colors.primaryAlpha(0.08)}
                        onPress={() => archiveSession(session.id, false)}>
                        <Text fontSize={11} fontWeight="500" color="$color9">Restore</Text>
                      </YStack>
                    </XStack>
                  )}
                />
              </YStack>
            )}
            {/* Label filter chips — below list */}
            {allLabels.length > 0 && (
              <XStack
                flexWrap="wrap" gap={4}
                paddingHorizontal={16} paddingVertical={8}
                paddingBottom={12}
                borderTopWidth={0.5} borderTopColor="$borderColor">
                <YStack
                  paddingHorizontal={10} paddingVertical={4} borderRadius={999}
                  backgroundColor={labelFilter === null ? '$color9' : 'transparent'}
                  borderWidth={0.5}
                  borderColor={labelFilter === null ? '$color9' : '$borderColor'}
                  onPress={() => setLabelFilter(null)}>
                  <Text fontSize={11} color={labelFilter === null ? 'white' : '$color11'}
                    fontWeight={labelFilter === null ? '600' : '400'}>All</Text>
                </YStack>
                {allLabels.map(label => (
                  <YStack
                    key={label}
                    paddingHorizontal={10} paddingVertical={4} borderRadius={999}
                    backgroundColor={labelFilter === label ? '$color9' : 'transparent'}
                    borderWidth={0.5}
                    borderColor={labelFilter === label ? '$color9' : '$borderColor'}
                    onPress={() => setLabelFilter(label === labelFilter ? null : label)}>
                    <Text fontSize={11}
                      color={labelFilter === label ? 'white' : '$color11'}
                      fontWeight={labelFilter === label ? '600' : '400'}>{label}</Text>
                  </YStack>
                ))}
              </XStack>
            )}
            <YStack height={20} />
          </YStack>
        </YStack>
      </TamaguiProvider>
    </Modal>
  );
}
