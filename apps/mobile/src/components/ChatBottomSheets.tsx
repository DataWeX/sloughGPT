import React from 'react';
import {Modal, FlatList, TextInput as RNTextInput, Platform} from 'react-native';
import {YStack, XStack, Text, useTheme} from 'tamagui';
import {useChatStore} from '../stores/chat-store';
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
  const theme = useTheme();

  return (
    <>
      {/* Conversation Info — bottom sheet */}
      <Modal visible={showInfo} animationType="slide" transparent onRequestClose={() => setShowInfo(false)}>
        <YStack flex={1} justifyContent="flex-end">
          <YStack
            flex={1}
            backgroundColor="rgba(0,0,0,0.3)"
            onPress={() => setShowInfo(false)}
          />
          <YStack
            backgroundColor="$background"
            borderTopLeftRadius={24}
            borderTopRightRadius={24}
            maxHeight="85%"
            overflow="hidden">
            <YStack alignItems="center" paddingTop={12} paddingBottom={2}>
              <YStack width={40} height={5} borderRadius={3} backgroundColor="$borderColor" opacity={0.4} />
            </YStack>

            <XStack
              paddingHorizontal={20} paddingVertical={14}
              borderBottomWidth={0.5} borderBottomColor="$borderColor"
              alignItems="center" justifyContent="space-between">
              <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color="$color">Details</Text>
              <YStack
                width={28} height={28} borderRadius={9}
                alignItems="center" justifyContent="center"
                onPress={() => setShowInfo(false)}
                pressStyle={{opacity: 0.6}}>
                <Icon name="x" size={14} color={(theme.color11?.val || '#6B7280')} />
              </YStack>
            </XStack>

            <XStack paddingHorizontal={20} paddingVertical={16} gap={10}>
              {[
                {label: 'Messages', value: String(messages.length)},
                {label: 'Words', value: String(messages.reduce((sum, m) => sum + (m.content?.split(/\s+/).length || 0), 0))},
                {label: 'Characters', value: String(messages.reduce((sum, m) => sum + (m.content?.length || 0), 0))},
              ].map(stat => (
                <YStack key={stat.label} flex={1} paddingVertical={14} paddingHorizontal={10} borderRadius={14}
                  backgroundColor="rgba(124, 82, 196, 0.04)" alignItems="center">
                  <Text fontSize={20} fontWeight="700" color="$color9">{stat.value}</Text>
                  <Text fontSize={10} fontWeight="500" color="$color10" marginTop={4}>{stat.label}</Text>
                </YStack>
              ))}
            </XStack>

            <YStack paddingHorizontal={20} paddingBottom={4}>
              <XStack justifyContent="space-between" paddingVertical={12}
                borderBottomWidth={0.5} borderBottomColor="$borderColor">
                <Text fontSize={13} color="$color11">Session</Text>
                <Text fontSize={13} color="$color" fontWeight="500" numberOfLines={1}>
                  {activeSessionId?.slice(0, 12) || 'None'}
                </Text>
              </XStack>
              {currentSoul && (
                <XStack justifyContent="space-between" paddingVertical={12}
                  borderBottomWidth={0.5} borderBottomColor="$borderColor">
                  <Text fontSize={13} color="$color11">Soul</Text>
                  <Text fontSize={13} color="$color" fontWeight="500">{currentSoul.name}</Text>
                </XStack>
              )}
              <XStack justifyContent="space-between" paddingVertical={12}
                borderBottomWidth={0.5} borderBottomColor="$borderColor">
                <Text fontSize={13} color="$color11">Status</Text>
                <XStack alignItems="center" gap={6}>
                  <YStack width={7} height={7} borderRadius={4}
                    backgroundColor={isConnected ? '#22C55E' : '#EF4444'} />
                  <Text fontSize={13} fontWeight="500" color="$color">
                    {isConnected ? 'Connected' : 'Offline'}
                  </Text>
                </XStack>
              </XStack>
            </YStack>

            <YStack paddingHorizontal={20} paddingTop={12} paddingBottom={8}>
              <Text fontSize={11} fontWeight="700" letterSpacing={0.6} color="$color10" marginBottom={12}>CHAT BACKGROUND</Text>
              <XStack flexWrap="wrap" gap={10}>
                {BG_PRESETS.map(p => {
                  const active = chatBackground === p.value;
                  return (
                    <YStack
                      key={p.value || 'none'}
                      width={36} height={36} borderRadius={12}
                      justifyContent="center" alignItems="center"
                      borderWidth={2}
                      borderColor={active ? '$color9' : 'transparent'}
                      backgroundColor={p.value ? p.value : (theme.background?.val || '#FFFFFF')}
                      style={!p.value ? {borderColor: (theme.borderColor?.val || '#E5E7EB')} : {}}
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
              <Text fontSize={11} fontWeight="700" letterSpacing={0.6} color="$color10" marginBottom={12}>LABELS</Text>
              {activeSessionId && (sessionLabels[activeSessionId] || []).length > 0 && (
                <XStack flexWrap="wrap" gap={6} marginBottom={12}>
                  {(sessionLabels[activeSessionId] || []).map(label => (
                    <YStack
                      key={label}
                      flexDirection="row" alignItems="center" gap={4}
                      backgroundColor="rgba(124, 82, 196, 0.08)"
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
                      <Text fontSize={12} fontWeight="500" color="$color9">{label}</Text>
                      <Icon name="x" size={10} color="$color9" />
                    </YStack>
                  ))}
                </XStack>
              )}
              <XStack gap={10}>
                <RNTextInput
                  style={{
                    flex: 1, fontSize: 13,
                    color: (theme.color?.val || '#111827'),
                    backgroundColor: 'rgba(124, 82, 196, 0.04)',
                    borderRadius: 12,
                    paddingHorizontal: 12, paddingVertical: 9,
                    borderWidth: 0.5, borderColor: (theme.borderColor?.val || '#E5E7EB'),
                  }}
                  value={labelInput}
                  onChangeText={setLabelInput}
                  placeholder="Add a label..."
                  placeholderTextColor={(theme.color10?.val || '#9CA3AF')}
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
            backgroundColor="rgba(0,0,0,0.3)"
            onPress={() => setShowSoulPicker(false)}
          />
          <YStack
            backgroundColor="$background"
            borderTopLeftRadius={24}
            borderTopRightRadius={24}
            maxHeight="75%"
            overflow="hidden">
            <YStack alignItems="center" paddingTop={12} paddingBottom={2}>
              <YStack width={40} height={5} borderRadius={3} backgroundColor="$borderColor" opacity={0.4} />
            </YStack>

            <XStack
              paddingHorizontal={20} paddingVertical={14}
              borderBottomWidth={0.5} borderBottomColor="$borderColor"
              alignItems="center" justifyContent="space-between">
              <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color="$color">Personalities</Text>
              <YStack
                width={28} height={28} borderRadius={9}
                alignItems="center" justifyContent="center"
                onPress={() => setShowSoulPicker(false)}
                pressStyle={{opacity: 0.6}}>
                <Icon name="x" size={14} color={(theme.color11?.val || '#6B7280')} />
              </YStack>
            </XStack>

            {currentSoul && (
              <YStack
                marginHorizontal={16} marginTop={12} marginBottom={4}
                paddingHorizontal={16} paddingVertical={14}
                borderRadius={14}
                backgroundColor="rgba(124, 82, 196, 0.06)"
                borderWidth={0.5} borderColor="rgba(124, 82, 196, 0.15)">
                <Text fontSize={10} fontWeight="700" letterSpacing={0.6} color="$color9" marginBottom={6}>ACTIVE</Text>
                <XStack alignItems="center" gap={10}>
                  <YStack width={32} height={32} borderRadius={16} backgroundColor="rgba(124, 82, 196, 0.12)" alignItems="center" justifyContent="center">
                    <Icon name="check" size={14} color="$color9" />
                  </YStack>
                  <YStack flex={1}>
                    <Text fontSize={15} fontWeight="600" color="$color">{currentSoul.name}</Text>
                    {currentSoul.description && (
                      <Text fontSize={12} color="$color10" marginTop={1}>{currentSoul.description}</Text>
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
                    backgroundColor={isActive ? 'rgba(124, 82, 196, 0.08)' : 'transparent'}
                    onPress={() => {
                      switchSoul(soul.name);
                      setShowSoulPicker(false);
                    }}
                    pressStyle={{backgroundColor: 'rgba(124, 82, 196, 0.06)', scale: 0.98}}>
                    <YStack
                      width={36} height={36} borderRadius={18}
                      backgroundColor={isActive ? '$color9' : 'rgba(124, 82, 196, 0.08)'}
                      alignItems="center" justifyContent="center">
                      <Icon name="user" size={16} color={isActive ? 'white' : (theme.color9?.val || '#7C52C4')} />
                    </YStack>
                    <YStack flex={1}>
                      <Text fontSize={14} fontWeight={isActive ? '600' : '400'} color="$color">{soul.name}</Text>
                      {soul.description && (
                        <Text fontSize={11} color="$color10" numberOfLines={1} marginTop={1}>
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
                  <Text fontSize={13} color="$color10">No personalities found</Text>
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
            backgroundColor="rgba(0,0,0,0.3)"
            onPress={() => setShowSettings(false)}
          />
          <YStack
            backgroundColor="$background"
            borderTopLeftRadius={24}
            borderTopRightRadius={24}
            overflow="hidden">
            <YStack alignItems="center" paddingTop={12} paddingBottom={2}>
              <YStack width={40} height={5} borderRadius={3} backgroundColor="$borderColor" opacity={0.4} />
            </YStack>

            <XStack
              paddingHorizontal={20} paddingVertical={16}
              borderBottomWidth={0.5} borderBottomColor="$borderColor"
              alignItems="center" justifyContent="space-between">
              <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color="$color">Menu</Text>
              <YStack
                width={28} height={28} borderRadius={9}
                alignItems="center" justifyContent="center"
                onPress={() => setShowSettings(false)}
                pressStyle={{opacity: 0.6}}>
                <Icon name="x" size={14} color={(theme.color11?.val || '#6B7280')} />
              </YStack>
            </XStack>

            <YStack
              paddingVertical={14} paddingHorizontal={20}
              borderBottomWidth={0.5} borderBottomColor="$borderColor"
              onPress={() => { createSession(); setShowSettings(false); }}
              pressStyle={{backgroundColor: 'rgba(124, 82, 196, 0.04)'}}>
              <XStack alignItems="center" gap={14}>
                <YStack width={36} height={36} borderRadius={12} backgroundColor="rgba(124, 82, 196, 0.08)" alignItems="center" justifyContent="center">
                  <Icon name="plus" size={18} color="$color9" />
                </YStack>
                <YStack>
                  <Text fontSize={15} fontWeight="600" color="$color">New Chat</Text>
                  <Text fontSize={12} color="$color10">Start a fresh conversation</Text>
                </YStack>
              </XStack>
            </YStack>

            <YStack
              paddingVertical={14} paddingHorizontal={20}
              borderBottomWidth={0.5} borderBottomColor="$borderColor"
              onPress={() => { setShowSettings(false); setShowSearch(true); }}
              pressStyle={{backgroundColor: 'rgba(124, 82, 196, 0.04)'}}>
              <XStack alignItems="center" gap={14}>
                <YStack width={36} height={36} borderRadius={12} backgroundColor="rgba(124, 82, 196, 0.08)" alignItems="center" justifyContent="center">
                  <Icon name="search" size={18} color="$color9" />
                </YStack>
                <YStack>
                  <Text fontSize={15} fontWeight="600" color="$color">Search</Text>
                  <Text fontSize={12} color="$color10">Find messages in this conversation</Text>
                </YStack>
              </XStack>
            </YStack>

            <YStack
              paddingVertical={14} paddingHorizontal={20}
              borderBottomWidth={0.5} borderBottomColor="$borderColor"
              onPress={() => {
                setShowSettings(false);
                updateTheme({theme: themeMode === 'dark' ? 'light' : 'dark'});
              }}
              pressStyle={{backgroundColor: 'rgba(124, 82, 196, 0.04)'}}>
              <XStack alignItems="center" gap={14}>
                <YStack width={36} height={36} borderRadius={12} backgroundColor="rgba(124, 82, 196, 0.08)" alignItems="center" justifyContent="center">
                  <Icon name={themeMode === 'dark' ? 'sun' : 'moon'} size={18} color="$color9" />
                </YStack>
                <YStack>
                  <Text fontSize={15} fontWeight="600" color="$color">
                    {themeMode === 'dark' ? 'Light Mode' : 'Dark Mode'}
                  </Text>
                  <Text fontSize={12} color="$color10">Currently: {themeMode === 'dark' ? 'Dark' : themeMode === 'light' ? 'Light' : 'System'}</Text>
                </YStack>
              </XStack>
            </YStack>

            <YStack
              paddingVertical={14} paddingHorizontal={20}
              borderBottomWidth={0.5} borderBottomColor="$borderColor"
              onPress={() => { setShowSettings(false); setShowInfo(true); }}
              pressStyle={{backgroundColor: 'rgba(124, 82, 196, 0.04)'}}>
              <XStack alignItems="center" gap={14}>
                <YStack width={36} height={36} borderRadius={12} backgroundColor="rgba(124, 82, 196, 0.08)" alignItems="center" justifyContent="center">
                  <Icon name="info" size={18} color="$color9" />
                </YStack>
                <YStack>
                  <Text fontSize={15} fontWeight="600" color="$color">Details</Text>
                  <Text fontSize={12} color="$color10">Stats, labels, and background</Text>
                </YStack>
              </XStack>
            </YStack>

            <YStack
              paddingVertical={14} paddingHorizontal={20}
              onPress={() => { setShowSettings(false); handleExportChat(); }}
              pressStyle={{backgroundColor: 'rgba(124, 82, 196, 0.04)'}}>
              <XStack alignItems="center" gap={14}>
                <YStack width={36} height={36} borderRadius={12} backgroundColor="rgba(124, 82, 196, 0.08)" alignItems="center" justifyContent="center">
                  <Icon name="download" size={18} color="$color9" />
                </YStack>
                <YStack>
                  <Text fontSize={15} fontWeight="600" color="$color">Export</Text>
                  <Text fontSize={12} color="$color10">Save conversation as markdown</Text>
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
          <YStack flex={1} backgroundColor="rgba(0,0,0,0.3)" onPress={() => setForwardTo(null)} />
          <YStack
            backgroundColor="$background"
            borderTopLeftRadius={24}
            borderTopRightRadius={24}
            maxHeight="65%"
            overflow="hidden">
            <YStack alignItems="center" paddingTop={12} paddingBottom={2}>
              <YStack width={40} height={5} borderRadius={3} backgroundColor="$borderColor" opacity={0.4} />
            </YStack>

            <XStack
              paddingHorizontal={20} paddingVertical={14}
              borderBottomWidth={0.5} borderBottomColor="$borderColor"
              alignItems="center" justifyContent="space-between">
              <Text fontSize={17} fontWeight="700" letterSpacing={-0.3} color="$color">Forward to...</Text>
              <YStack width={28} height={28} borderRadius={9} alignItems="center" justifyContent="center"
                onPress={() => setForwardTo(null)} pressStyle={{opacity: 0.6}}>
                <Icon name="x" size={14} color={(theme.color11?.val || '#6B7280')} />
              </YStack>
            </XStack>

            {forwardTo && (
              <YStack
                marginHorizontal={16} marginTop={10} marginBottom={6}
                paddingHorizontal={14} paddingVertical={10}
                borderRadius={12}
                backgroundColor="rgba(124, 82, 196, 0.04)"
                borderWidth={0.5} borderColor="rgba(124, 82, 196, 0.12)">
                <Text fontSize={11} fontWeight="600" color="$color9" marginBottom={2}>MESSAGE</Text>
                <Text fontSize={13} color="$color10" numberOfLines={2}>{forwardTo.content}</Text>
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
                  pressStyle={{backgroundColor: 'rgba(124, 82, 196, 0.06)', scale: 0.98}}>
                  <YStack width={36} height={36} borderRadius={12}
                    backgroundColor="rgba(124, 82, 196, 0.08)" alignItems="center" justifyContent="center">
                    <Icon name="message-circle" size={16} color="$color9" />
                  </YStack>
                  <YStack flex={1}>
                    <Text fontSize={14} fontWeight="500" numberOfLines={1} color="$color">
                      {session.name || 'New conversation'}
                    </Text>
                    <Text fontSize={11} color="$color10" marginTop={1}>
                      {session.message_count || 0} messages
                    </Text>
                  </YStack>
                </XStack>
              )}
              ListEmptyComponent={
                <YStack padding={40} alignItems="center">
                  <Text fontSize={13} color="$color10">No conversations</Text>
                </YStack>
              }
            />
            <YStack height={Platform.OS === 'ios' ? 34 : 16} />
          </YStack>
        </YStack>
      </Modal>
    </>
  );
}
