import React from 'react';
import {Modal, FlatList, TextInput as RNTextInput, Platform} from 'react-native';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {useChatStore} from '../stores/chat-store';
import {useSettingsStore} from '../stores/settings-store';
import {useModelStore} from '../stores/model-store';
import {toast} from '../services/toast';
import * as labelsService from '../services/labels';
import {Icon} from './Icon';
import {SearchSessionsModal} from './SearchSessionsModal';
import {StatusBadge} from './StatusBadge';
import type {Message, Session} from '../types';

const BG_PRESETS = [
  {label: 'None', value: ''},
  {label: 'Navy', value: '#1a1a2e'},
  {label: 'Plum', value: '#2d1b2e'},
  {label: 'Forest', value: '#1b2e1a'},
  {label: 'Amber', value: '#2e2e1b'},
  {label: 'Maroon', value: '#2a1a1a'},
  {label: 'Cream', value: '#f5f0e8'},
  {label: 'Ice', value: '#e8f0f5'},
  {label: 'Lavender', value: '#f0e8f5'},
  {label: 'Rose', value: '#f5e8e8'},
];

interface ChatBottomSheetsProps {
  showInfo: boolean;
  setShowInfo: (v: boolean) => void;
  showSearchSessions: boolean;
  setShowSearchSessions: (v: boolean) => void;
  handleSelectSearchSession: (id: string) => void;
  showSoulPicker: boolean;
  setShowSoulPicker: (v: boolean) => void;
  showSettings: boolean;
  setShowSettings: (v: boolean) => void;
  showChatSettings: boolean;
  setShowChatSettings: (v: boolean) => void;
  showSystemPrompt: boolean;
  setShowSystemPrompt: (v: boolean) => void;
  forwardTo: Message | null;
  setForwardTo: (m: Message | null) => void;
  safeSessions: Session[];
  activeSessionId: string | null;
  messages: Message[];
  currentSoul: any;
  souls: any[];
  switchSoul: (name: string) => void;
  isConnected: boolean;
  chatBackground: string;
  updateTheme: (patch: any) => void;
  themeMode: string;
  sessionLabels: Record<string, string[]>;
  setSessionLabels: (fn: (prev: Record<string, string[]>) => Record<string, string[]>) => void;
  allLabels: string[];
  setAllLabels: (v: string[]) => void;
  labelInput: string;
  setLabelInput: (v: string) => void;
  forwardMessage: (content: string, sessionId: string) => Promise<void>;
  createSession: () => Promise<any>;
  handleExportChat: () => void;
  setShowSearch: (v: boolean) => void;
}

export function ChatBottomSheets({
  showInfo,
  setShowInfo,
  showSearchSessions,
  setShowSearchSessions,
  handleSelectSearchSession,
  showSoulPicker,
  setShowSoulPicker,
  showSettings,
  setShowSettings,
  showChatSettings,
  setShowChatSettings,
  showSystemPrompt,
  setShowSystemPrompt,
  forwardTo,
  setForwardTo,
  safeSessions,
  activeSessionId,
  messages,
  currentSoul,
  souls,
  switchSoul,
  isConnected,
  chatBackground,
  updateTheme,
  themeMode,
  sessionLabels,
  setSessionLabels,
  allLabels,
  setAllLabels,
  labelInput,
  setLabelInput,
  forwardMessage,
  createSession,
  handleExportChat,
  setShowSearch,
}: ChatBottomSheetsProps) {
  const colors = useColors();

  return (
    <>
      {/* Conversation Info — bottom sheet */}
      <Modal visible={showInfo} animationType="slide" transparent onRequestClose={() => setShowInfo(false)}>
        <YStack flex={1} justifyContent="flex-end">
          <YStack
            flex={1}
            backgroundColor={colors.overlay(0.3)}
            onPress={() => setShowInfo(false)}
          />
          <YStack
            backgroundColor={colors.background}
            borderTopLeftRadius={24}
            borderTopRightRadius={24}
            maxHeight="85%"
            overflow="hidden">
            <YStack alignItems="center" paddingTop={12} paddingBottom={2}>
              <YStack width={40} height={5} borderRadius={3} backgroundColor={colors.border} opacity={0.4} />
            </YStack>

            <XStack
              paddingHorizontal={20} paddingVertical={14}
              borderBottomWidth={0.5} borderBottomColor={colors.border}
              alignItems="center" justifyContent="space-between">
              <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color={colors.text}>Details</Text>
              <YStack
                width={28} height={28} borderRadius={9}
                alignItems="center" justifyContent="center"
                onPress={() => setShowInfo(false)}
                pressStyle={{opacity: 0.6}}>
                <Icon name="x" size={14} color={colors.textSecondary} />
              </YStack>
            </XStack>

            <XStack paddingHorizontal={20} paddingVertical={16} gap={10}>
              {[
                {label: 'Messages', value: String(messages.length)},
                {label: 'Words', value: String(messages.reduce((sum, m) => sum + (m.content?.split(/\s+/).length || 0), 0))},
                {label: 'Characters', value: String(messages.reduce((sum, m) => sum + (m.content?.length || 0), 0))},
              ].map(stat => (
                <YStack key={stat.label} flex={1} paddingVertical={14} paddingHorizontal={10} borderRadius={14}
                  backgroundColor={colors.primaryAlpha(0.04)} alignItems="center">
                  <Text fontSize={20} fontWeight="700" color={colors.primary}>{stat.value}</Text>
                  <Text fontSize={10} fontWeight="500" color={colors.textSecondary} marginTop={4}>{stat.label}</Text>
                </YStack>
              ))}
            </XStack>

            <YStack paddingHorizontal={20} paddingBottom={4}>
              <XStack justifyContent="space-between" paddingVertical={12}
                borderBottomWidth={0.5} borderBottomColor={colors.border}>
                <Text fontSize={13} color={colors.textMuted}>Session</Text>
                <Text fontSize={13} color={colors.text} fontWeight="500" numberOfLines={1}>
                  {activeSessionId?.slice(0, 12) || 'None'}
                </Text>
              </XStack>
              {currentSoul && (
                <XStack justifyContent="space-between" paddingVertical={12}
                  borderBottomWidth={0.5} borderBottomColor={colors.border}>
                  <Text fontSize={13} color={colors.textMuted}>Soul</Text>
                  <Text fontSize={13} color={colors.text} fontWeight="500">{currentSoul.name}</Text>
                </XStack>
              )}
              <XStack justifyContent="space-between" paddingVertical={12}
                borderBottomWidth={0.5} borderBottomColor={colors.border}>
                <Text fontSize={13} color={colors.textMuted}>Status</Text>
                <XStack alignItems="center" gap={6}>
                  <YStack width={7} height={7} borderRadius={4}
                    backgroundColor={isConnected ? colors.success : colors.error} />
                  <Text fontSize={13} fontWeight="500" color={colors.text}>
                    {isConnected ? 'Connected' : 'Offline'}
                  </Text>
                </XStack>
              </XStack>
            </YStack>

            <YStack paddingHorizontal={20} paddingTop={12} paddingBottom={8}>
              <Text fontSize={11} fontWeight="700" letterSpacing={0.6} color={colors.textSecondary} marginBottom={12}>CHAT BACKGROUND</Text>
              <XStack flexWrap="wrap" gap={10}>
                {BG_PRESETS.map(p => {
                  const active = chatBackground === p.value;
                  return (
                    <YStack
                      key={p.value || 'none'}
                      width={36} height={36} borderRadius={12}
                      justifyContent="center" alignItems="center"
                      borderWidth={2}
                      borderColor={active ? colors.primary : 'transparent'}
                      backgroundColor={p.value ? p.value : colors.background}
                      style={!p.value ? {borderColor: colors.border} : {}}
                      onPress={() => updateTheme({chatBackground: p.value})}
                      pressStyle={{scale: 0.88}}>
                      {active && (
                        <Icon name="check" size={12} color="white" />
                      )}
                    </YStack>
                  );
                })}
              </XStack>
            </YStack>

            <YStack paddingHorizontal={20} paddingTop={16} paddingBottom={24}>
              <Text fontSize={11} fontWeight="700" letterSpacing={0.6} color={colors.textSecondary} marginBottom={12}>LABELS</Text>
              {activeSessionId && (sessionLabels[activeSessionId] || []).length > 0 && (
                <XStack flexWrap="wrap" gap={6} marginBottom={12}>
                  {(sessionLabels[activeSessionId] || []).map(label => (
                    <YStack
                      key={label}
                      flexDirection="row" alignItems="center" gap={4}
                      backgroundColor={colors.primaryAlpha(0.08)}
                      paddingHorizontal={10} paddingVertical={5} borderRadius={999}
                      onPress={async () => {
                        if (activeSessionId) {
                          try {
                            await labelsService.removeLabel(activeSessionId, label);
                            const labels = await labelsService.getLabels(activeSessionId);
                            setSessionLabels(prev => ({...prev, [activeSessionId]: labels}));
                            const distinct = await labelsService.getAllDistinctLabels();
                            setAllLabels(distinct);
                          } catch {
                            toast.error('Failed to remove label');
                          }
                        }
                      }}>
                      <Text fontSize={12} fontWeight="500" color={colors.primary}>{label}</Text>
                      <Icon name="x" size={10} color={colors.primary} />
                    </YStack>
                  ))}
                </XStack>
              )}
              <XStack gap={10}>
                <RNTextInput
                  style={{
                    flex: 1, fontSize: 13,
                    color: colors.text,
                    backgroundColor: colors.primaryAlpha(0.04),
                    borderRadius: 12,
                    paddingHorizontal: 12, paddingVertical: 9,
                    borderWidth: 0.5, borderColor: colors.border,
                  }}
                  value={labelInput}
                  onChangeText={setLabelInput}
                  placeholder="Add a label..."
                  placeholderTextColor={colors.textMuted}
                  returnKeyType="done"
                  onSubmitEditing={async () => {
                    if (labelInput.trim() && activeSessionId) {
                      try {
                        await labelsService.addLabel(activeSessionId, labelInput.trim());
                        setLabelInput('');
                        const labels = await labelsService.getLabels(activeSessionId);
                        setSessionLabels(prev => ({...prev, [activeSessionId]: labels}));
                        const distinct = await labelsService.getAllDistinctLabels();
                        setAllLabels(distinct);
                      } catch {
                        toast.error('Failed to add label');
                      }
                    }
                  }}
                />
              </XStack>
            </YStack>
          </YStack>
        </YStack>
      </Modal>

      {/* Soul picker — bottom sheet */}
      <Modal visible={showSoulPicker} animationType="slide" transparent>
        <YStack flex={1} justifyContent="flex-end">
          <YStack
            flex={1}
            backgroundColor={colors.overlay(0.3)}
            onPress={() => setShowSoulPicker(false)}
          />
          <YStack
            backgroundColor={colors.background}
            borderTopLeftRadius={24}
            borderTopRightRadius={24}
            maxHeight="75%"
            overflow="hidden">
            <YStack alignItems="center" paddingTop={12} paddingBottom={2}>
              <YStack width={40} height={5} borderRadius={3} backgroundColor={colors.border} opacity={0.4} />
            </YStack>

            <XStack
              paddingHorizontal={20} paddingVertical={14}
              borderBottomWidth={0.5} borderBottomColor={colors.border}
              alignItems="center" justifyContent="space-between">
              <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color={colors.text}>Personalities</Text>
              <YStack
                width={28} height={28} borderRadius={9}
                alignItems="center" justifyContent="center"
                onPress={() => setShowSoulPicker(false)}
                pressStyle={{opacity: 0.6}}>
                <Icon name="x" size={14} color={colors.textSecondary} />
              </YStack>
            </XStack>

            {currentSoul && (
              <YStack
                marginHorizontal={16} marginTop={12} marginBottom={4}
                paddingHorizontal={16} paddingVertical={14}
                borderRadius={14}
                backgroundColor={colors.primaryAlpha(0.06)}
                borderWidth={0.5} borderColor={colors.primaryAlpha(0.15)}>
                <Text fontSize={10} fontWeight="700" letterSpacing={0.6} color={colors.primary} marginBottom={6}>ACTIVE</Text>
                <XStack alignItems="center" gap={10}>
                  <YStack width={32} height={32} borderRadius={16} backgroundColor={colors.primaryAlpha(0.12)} alignItems="center" justifyContent="center">
                    <Icon name="check" size={14} color={colors.primary} />
                  </YStack>
                  <YStack flex={1}>
                    <Text fontSize={15} fontWeight="600" color={colors.text}>{currentSoul.name}</Text>
                    {currentSoul.description && (
                      <Text fontSize={12} color={colors.textSecondary} marginTop={1}>{currentSoul.description}</Text>
                    )}
                  </YStack>
                </XStack>
              </YStack>
            )}

            <FlatList
              data={souls}
              keyExtractor={item => item.name}
              contentContainerStyle={{paddingVertical: 6, paddingHorizontal: 16}}
              renderItem={({item: soul}) => {
                const isActive = currentSoul?.name === soul.name;
                return (
                  <XStack
                    paddingVertical={12} paddingHorizontal={14}
                    marginVertical={2}
                    borderRadius={12}
                    alignItems="center" gap={12}
                    backgroundColor={isActive ? colors.primaryAlpha(0.08) : 'transparent'}
                    onPress={() => {
                      switchSoul(soul.name);
                      setShowSoulPicker(false);
                    }}
                    pressStyle={{backgroundColor: colors.primaryAlpha(0.06), scale: 0.98}}>
                    <YStack
                      width={36} height={36} borderRadius={18}
                      backgroundColor={isActive ? '$color9' : colors.primaryAlpha(0.08)}
                      alignItems="center" justifyContent="center">
                      <Icon name="user" size={16} color={isActive ? 'white' : colors.primary} />
                    </YStack>
                    <YStack flex={1}>
                      <Text fontSize={14} fontWeight={isActive ? '600' : '400'} color={colors.text}>{soul.name}</Text>
                      {soul.description && (
                        <Text fontSize={11} color={colors.textSecondary} numberOfLines={1} marginTop={1}>
                          {soul.description}
                        </Text>
                      )}
                      {soul.traits && soul.traits.length > 0 && (
                        <XStack flexWrap="wrap" gap={3} marginTop={4}>
                          {soul.traits.map((trait: string) => (
                            <StatusBadge key={trait} label={trait} variant="info" />
                          ))}
                        </XStack>
                      )}
                    </YStack>
                    {isActive && (
                      <YStack width={20} height={20} borderRadius={10} backgroundColor="$color9" alignItems="center" justifyContent="center">
                        <Icon name="check" size={12} color="white" />
                      </YStack>
                    )}
                  </XStack>
                );
              }}
              ListEmptyComponent={
                <YStack padding={40} alignItems="center">
                  <Text fontSize={13} color={colors.textSecondary}>No personalities found</Text>
                </YStack>
              }
            />
            <YStack height={Platform.OS === 'ios' ? 34 : 16} />
          </YStack>
        </YStack>
      </Modal>

      <SearchSessionsModal
        visible={showSearchSessions}
        onClose={() => setShowSearchSessions(false)}
        onSelectSession={handleSelectSearchSession}
      />

      {/* Overflow menu — bottom sheet */}
      <Modal visible={showSettings} animationType="slide" transparent onRequestClose={() => setShowSettings(false)}>
        <YStack flex={1} justifyContent="flex-end">
          <YStack
            flex={1}
            backgroundColor={colors.overlay(0.3)}
            onPress={() => setShowSettings(false)}
          />
          <YStack
            backgroundColor={colors.background}
            borderTopLeftRadius={24}
            borderTopRightRadius={24}
            overflow="hidden">
            <YStack alignItems="center" paddingTop={12} paddingBottom={2}>
              <YStack width={40} height={5} borderRadius={3} backgroundColor={colors.border} opacity={0.4} />
            </YStack>

            <XStack
              paddingHorizontal={20} paddingVertical={16}
              borderBottomWidth={0.5} borderBottomColor={colors.border}
              alignItems="center" justifyContent="space-between">
              <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color={colors.text}>Menu</Text>
              <YStack
                width={28} height={28} borderRadius={9}
                alignItems="center" justifyContent="center"
                onPress={() => setShowSettings(false)}
                pressStyle={{opacity: 0.6}}>
                <Icon name="x" size={14} color={colors.textSecondary} />
              </YStack>
            </XStack>

            <YStack
              paddingVertical={14} paddingHorizontal={20}
              borderBottomWidth={0.5} borderBottomColor={colors.border}
              onPress={() => { createSession(); setShowSettings(false); }}
              pressStyle={{backgroundColor: colors.primaryAlpha(0.04)}}>
              <XStack alignItems="center" gap={14}>
                <YStack width={36} height={36} borderRadius={12} backgroundColor={colors.primaryAlpha(0.08)} alignItems="center" justifyContent="center">
                  <Icon name="plus" size={18} color={colors.primary} />
                </YStack>
                <YStack>
                  <Text fontSize={15} fontWeight="600" color={colors.text}>New Chat</Text>
                  <Text fontSize={12} color={colors.textSecondary}>Start a fresh conversation</Text>
                </YStack>
              </XStack>
            </YStack>

            <YStack
              paddingVertical={14} paddingHorizontal={20}
              borderBottomWidth={0.5} borderBottomColor={colors.border}
              onPress={() => { setShowSettings(false); setShowSearch(true); }}
              pressStyle={{backgroundColor: colors.primaryAlpha(0.04)}}>
              <XStack alignItems="center" gap={14}>
                <YStack width={36} height={36} borderRadius={12} backgroundColor={colors.primaryAlpha(0.08)} alignItems="center" justifyContent="center">
                  <Icon name="search" size={18} color={colors.primary} />
                </YStack>
                <YStack>
                  <Text fontSize={15} fontWeight="600" color={colors.text}>Search</Text>
                  <Text fontSize={12} color={colors.textSecondary}>Find messages in this conversation</Text>
                </YStack>
              </XStack>
            </YStack>

            <YStack
              paddingVertical={14} paddingHorizontal={20}
              borderBottomWidth={0.5} borderBottomColor={colors.border}
              onPress={() => {
                setShowSettings(false);
                updateTheme({theme: themeMode === 'dark' ? 'light' : 'dark'});
              }}
              pressStyle={{backgroundColor: colors.primaryAlpha(0.04)}}>
              <XStack alignItems="center" gap={14}>
                <YStack width={36} height={36} borderRadius={12} backgroundColor={colors.primaryAlpha(0.08)} alignItems="center" justifyContent="center">
                  <Icon name={themeMode === 'dark' ? 'sun' : 'moon'} size={18} color={colors.primary} />
                </YStack>
                <YStack>
                  <Text fontSize={15} fontWeight="600" color={colors.text}>
                    {themeMode === 'dark' ? 'Light Mode' : 'Dark Mode'}
                  </Text>
                  <Text fontSize={12} color={colors.textSecondary}>Currently: {themeMode === 'dark' ? 'Dark' : themeMode === 'light' ? 'Light' : 'System'}</Text>
                </YStack>
              </XStack>
            </YStack>

            <YStack
              paddingVertical={14} paddingHorizontal={20}
              borderBottomWidth={0.5} borderBottomColor={colors.border}
              onPress={() => { setShowSettings(false); setShowInfo(true); }}
              pressStyle={{backgroundColor: colors.primaryAlpha(0.04)}}>
              <XStack alignItems="center" gap={14}>
                <YStack width={36} height={36} borderRadius={12} backgroundColor={colors.primaryAlpha(0.08)} alignItems="center" justifyContent="center">
                  <Icon name="info" size={18} color={colors.primary} />
                </YStack>
                <YStack>
                  <Text fontSize={15} fontWeight="600" color={colors.text}>Details</Text>
                  <Text fontSize={12} color={colors.textSecondary}>Stats, labels, and background</Text>
                </YStack>
              </XStack>
            </YStack>

            <YStack
              paddingVertical={14} paddingHorizontal={20}
              borderBottomWidth={0.5} borderBottomColor={colors.border}
              onPress={() => { setShowSettings(false); setShowChatSettings(true); }}
              pressStyle={{backgroundColor: colors.primaryAlpha(0.04)}}>
              <XStack alignItems="center" gap={14}>
                <YStack width={36} height={36} borderRadius={12} backgroundColor={colors.primaryAlpha(0.08)} alignItems="center" justifyContent="center">
                  <Icon name="settings" size={18} color={colors.primary} />
                </YStack>
                <YStack>
                  <Text fontSize={15} fontWeight="600" color={colors.text}>Generation Settings</Text>
                  <Text fontSize={12} color={colors.textSecondary}>Temperature, tokens, and sampling</Text>
                </YStack>
              </XStack>
            </YStack>

            <YStack
              paddingVertical={14} paddingHorizontal={20}
              borderBottomWidth={0.5} borderBottomColor={colors.border}
              onPress={() => { setShowSettings(false); setShowSystemPrompt(true); }}
              pressStyle={{backgroundColor: colors.primaryAlpha(0.04)}}>
              <XStack alignItems="center" gap={14}>
                <YStack width={36} height={36} borderRadius={12} backgroundColor={colors.primaryAlpha(0.08)} alignItems="center" justifyContent="center">
                  <Icon name="info" size={18} color={colors.primary} />
                </YStack>
                <YStack>
                  <Text fontSize={15} fontWeight="600" color={colors.text}>System Prompt</Text>
                  <Text fontSize={12} color={colors.textSecondary}>View the active personality prompt</Text>
                </YStack>
              </XStack>
            </YStack>

            <YStack
              paddingVertical={14} paddingHorizontal={20}
              onPress={() => { setShowSettings(false); handleExportChat(); }}
              pressStyle={{backgroundColor: colors.primaryAlpha(0.04)}}>
              <XStack alignItems="center" gap={14}>
                <YStack width={36} height={36} borderRadius={12} backgroundColor={colors.primaryAlpha(0.08)} alignItems="center" justifyContent="center">
                  <Icon name="download" size={18} color={colors.primary} />
                </YStack>
                <YStack>
                  <Text fontSize={15} fontWeight="600" color={colors.text}>Export</Text>
                  <Text fontSize={12} color={colors.textSecondary}>Save conversation as markdown</Text>
                </YStack>
              </XStack>
            </YStack>

            <YStack height={Platform.OS === 'ios' ? 34 : 16} />
          </YStack>
        </YStack>
      </Modal>

      {/* Forward to session — bottom sheet */}
      <Modal
        visible={forwardTo !== null}
        transparent
        animationType="slide"
        onRequestClose={() => setForwardTo(null)}>
        <YStack flex={1} justifyContent="flex-end">
          <YStack flex={1} backgroundColor={colors.overlay(0.3)} onPress={() => setForwardTo(null)} />
          <YStack
            backgroundColor={colors.background}
            borderTopLeftRadius={24}
            borderTopRightRadius={24}
            maxHeight="65%"
            overflow="hidden">
            <YStack alignItems="center" paddingTop={12} paddingBottom={2}>
              <YStack width={40} height={5} borderRadius={3} backgroundColor={colors.border} opacity={0.4} />
            </YStack>

            <XStack
              paddingHorizontal={20} paddingVertical={14}
              borderBottomWidth={0.5} borderBottomColor={colors.border}
              alignItems="center" justifyContent="space-between">
              <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color={colors.text}>Forward to...</Text>
              <YStack width={28} height={28} borderRadius={9} alignItems="center" justifyContent="center"
                onPress={() => setForwardTo(null)} pressStyle={{opacity: 0.6}}>
                <Icon name="x" size={14} color={colors.textSecondary} />
              </YStack>
            </XStack>

            {forwardTo && (
              <YStack
                marginHorizontal={16} marginTop={10} marginBottom={6}
                paddingHorizontal={14} paddingVertical={10}
                borderRadius={12}
                backgroundColor={colors.primaryAlpha(0.04)}
                borderWidth={0.5} borderColor={colors.primaryAlpha(0.12)}>
                <Text fontSize={11} fontWeight="600" color={colors.primary} marginBottom={2}>MESSAGE</Text>
                <Text fontSize={13} color={colors.textSecondary} numberOfLines={2}>{forwardTo.content}</Text>
              </YStack>
            )}

            <FlatList
              data={safeSessions}
              keyExtractor={s => s.id}
              contentContainerStyle={{paddingHorizontal: 16, paddingVertical: 4}}
              renderItem={({item: session}) => (
                <XStack
                  paddingVertical={12} paddingHorizontal={14}
                  marginVertical={2}
                  borderRadius={12}
                  alignItems="center" gap={12}
                  onPress={async () => {
                    if (forwardTo) {
                      await forwardMessage(forwardTo.content, session.id);
                    }
                    setForwardTo(null);
                  }}
                  pressStyle={{backgroundColor: colors.primaryAlpha(0.06), scale: 0.98}}>
                  <YStack width={36} height={36} borderRadius={12}
                    backgroundColor={colors.primaryAlpha(0.08)} alignItems="center" justifyContent="center">
                    <Icon name="message-circle" size={16} color={colors.primary} />
                  </YStack>
                  <YStack flex={1}>
                    <Text fontSize={14} fontWeight="500" numberOfLines={1} color={colors.text}>
                      {session.name || 'New conversation'}
                    </Text>
                    <Text fontSize={11} color={colors.textSecondary} marginTop={1}>
                      {session.message_count || 0} messages
                    </Text>
                  </YStack>
                </XStack>
              )}
              ListEmptyComponent={
                <YStack padding={40} alignItems="center">
                  <Text fontSize={13} color={colors.textSecondary}>No conversations</Text>
                </YStack>
              }
            />
            <YStack height={Platform.OS === 'ios' ? 34 : 16} />
          </YStack>
        </YStack>
      </Modal>

      <Modal visible={showChatSettings} animationType="slide" transparent onRequestClose={() => setShowChatSettings(false)}>
        <YStack flex={1} justifyContent="flex-end">
          <YStack
            flex={1}
            backgroundColor={colors.overlay(0.3)}
            onPress={() => setShowChatSettings(false)}
          />
          <YStack
            backgroundColor={colors.background}
            borderTopLeftRadius={24}
            borderTopRightRadius={24}
            overflow="hidden">
            <YStack alignItems="center" paddingTop={12} paddingBottom={2}>
              <YStack width={40} height={5} borderRadius={3} backgroundColor={colors.border} opacity={0.4} />
            </YStack>

            <XStack
              paddingHorizontal={20} paddingVertical={16}
              borderBottomWidth={0.5} borderBottomColor={colors.border}
              alignItems="center" justifyContent="space-between">
              <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color={colors.text}>Generation Settings</Text>
              <YStack
                width={28} height={28} borderRadius={9}
                alignItems="center" justifyContent="center"
                onPress={() => setShowChatSettings(false)}
                pressStyle={{opacity: 0.6}}>
                <Icon name="x" size={14} color={colors.textSecondary} />
              </YStack>
            </XStack>

            <YStack paddingVertical={14} paddingHorizontal={20}>
              <ChatSettingsContent />
            </YStack>

            <YStack height={Platform.OS === 'ios' ? 34 : 16} />
          </YStack>
        </YStack>
      </Modal>

      <Modal visible={showSystemPrompt} animationType="slide" transparent onRequestClose={() => setShowSystemPrompt(false)}>
        <YStack flex={1} justifyContent="flex-end">
          <YStack
            flex={1}
            backgroundColor={colors.overlay(0.3)}
            onPress={() => setShowSystemPrompt(false)}
          />
          <YStack
            backgroundColor={colors.background}
            borderTopLeftRadius={24}
            borderTopRightRadius={24}
            overflow="hidden">
            <YStack alignItems="center" paddingTop={12} paddingBottom={2}>
              <YStack width={40} height={5} borderRadius={3} backgroundColor={colors.border} opacity={0.4} />
            </YStack>

            <XStack
              paddingHorizontal={20} paddingVertical={16}
              borderBottomWidth={0.5} borderBottomColor={colors.border}
              alignItems="center" justifyContent="space-between">
              <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color={colors.text}>System Prompt</Text>
              <YStack
                width={28} height={28} borderRadius={9}
                alignItems="center" justifyContent="center"
                onPress={() => setShowSystemPrompt(false)}
                pressStyle={{opacity: 0.6}}>
                <Icon name="x" size={14} color={colors.textSecondary} />
              </YStack>
            </XStack>

            <YStack paddingVertical={14} paddingHorizontal={20}>
              <SystemPromptContent />
            </YStack>

            <YStack height={Platform.OS === 'ios' ? 34 : 16} />
          </YStack>
        </YStack>
      </Modal>
    </>
  );
}

function ChatSettingsContent() {
  const colors = useColors();
  const settings = useSettingsStore();
  const update = useSettingsStore(s => s.update);

  const sliders = [
    {key: 'temperature' as const, label: 'Temperature', min: 0, max: 2, step: 0.1},
    {key: 'topP' as const, label: 'Top P', min: 0, max: 1, step: 0.05},
    {key: 'topK' as const, label: 'Top K', min: 1, max: 100, step: 1},
    {key: 'maxTokens' as const, label: 'Max Tokens', min: 32, max: 4096, step: 32},
  ];

  return (
    <YStack gap={20}>
      {sliders.map(s => (
        <YStack key={s.key}>
          <XStack justifyContent="space-between" alignItems="center" marginBottom={6}>
            <Text fontSize={13} fontWeight="500" color={colors.text}>{s.label}</Text>
            <Text fontSize={13} fontWeight="600" color={colors.primary}>
              {s.key === 'maxTokens' ? Math.round(settings[s.key]) : settings[s.key].toFixed(2)}
            </Text>
          </XStack>
          <YStack
            height={44}
            backgroundColor={colors.primaryAlpha(0.04)}
            borderRadius={10}
            paddingHorizontal={12}
            alignItems="center"
            justifyContent="center">
            <RNTextInput
              style={{
                width: '100%',
                height: 44,
                fontSize: 14,
                color: colors.text,
                textAlign: 'center',
              }}
              keyboardType="numeric"
              value={String(settings[s.key])}
              onEndEditing={(e) => {
                const val = parseFloat(e.nativeEvent.text);
                if (!isNaN(val)) {
                  const clamped = Math.max(s.min, Math.min(s.max, val));
                  update({[s.key]: clamped});
                }
              }}
            />
          </YStack>
        </YStack>
      ))}

      <YStack
        paddingVertical={12}
        borderRadius={10}
        alignItems="center"
        backgroundColor={colors.primaryAlpha(0.08)}
        onPress={() => update({temperature: 0.8, maxTokens: 256, topP: 0.9, topK: 50})}
        pressStyle={{opacity: 0.7, scale: 0.98}}>
        <Text fontSize={13} fontWeight="600" color={colors.primary}>Reset to Defaults</Text>
      </YStack>
    </YStack>
  );
}

function SystemPromptContent() {
  const colors = useColors();
  const currentSoul = useModelStore(s => s.currentSoul);
  const [prompt, setPrompt] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    const fetchPrompt = async () => {
      try {
        const res = await fetch('http://localhost:8000/souls/current');
        if (res.ok) {
          const data = await res.json();
          setPrompt(data.system_prompt || data.description || 'No system prompt available for this soul.');
        }
      } catch {
        setPrompt('Unable to load system prompt.');
      } finally {
        setLoading(false);
      }
    };
    fetchPrompt();
  }, []);

  if (loading) {
    return <Text fontSize={13} color={colors.textSecondary} padding={20}>Loading...</Text>;
  }

  return (
    <YStack gap={12}>
      <XStack alignItems="center" gap={8}>
        <Text fontSize={13} fontWeight="600" color={colors.primary}>Active Soul:</Text>
        <Text fontSize={13} color={colors.text}>{currentSoul?.name || 'None'}</Text>
      </XStack>
      <YStack
        backgroundColor={colors.primaryAlpha(0.04)}
        borderRadius={10}
        padding={14}>
        <Text fontSize={13} lineHeight={20} color={colors.text} selectable>
          {prompt || 'No system prompt available.'}
        </Text>
      </YStack>
    </YStack>
  );
}
