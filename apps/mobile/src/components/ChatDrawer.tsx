import React, { useCallback, useEffect, useState } from 'react'
import { FlatList, Alert, TextInput as RNTextInput, Modal } from 'react-native'
import { YStack, XStack, Text } from 'tamagui'
import { TamaguiProvider } from '../theme/TamaguiProvider'
import { useColors } from '../theme/colors'
import { S, R, T } from '../theme/scales'
import { useChatStore } from '../stores/chat-store'
import { triggerHaptic } from '../services/haptics'
import { toast } from '../services/toast'
import * as starsService from '../services/stars'
import * as labelsService from '../services/labels'
import * as foldersService from '../services/folders'
import { Icon } from './Icon'
import type { Session } from '../types'
import { Platform } from 'react-native'

interface ChatDrawerProps {
  visible: boolean
  onClose: () => void
  sortedActiveSessions: Session[]
  archivedSessions: Session[]
  sessionLabels: Record<string, string[]>
  allLabels: string[]
  labelFilter: string | null
  setLabelFilter: (f: string | null) => void
  folderFilter: string | null
  setFolderFilter: (f: string | null) => void
  showArchived: boolean
  setShowArchived: (v: boolean) => void
  labelInput: string
  setLabelInput: (v: string) => void
  starredIds: string[]
  setStarredIds: (fn: (prev: string[]) => string[]) => void
  setSessionLabels: (fn: (prev: Record<string, string[]>) => Record<string, string[]>) => void
  setAllLabels: (v: string[]) => void
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
  folderFilter,
  setFolderFilter,
  showArchived,
  setShowArchived,
  labelInput,
  setLabelInput,
  starredIds,
  setStarredIds,
  setSessionLabels,
  setAllLabels,
}: ChatDrawerProps) {
  const colors = useColors()
  const { loadSession, deleteSession, renameSession, archiveSession, cloneSession } = useChatStore()
  const [folders, setFolders] = useState<foldersService.Folder[]>([])
  const [folderOf, setFolderOf] = useState<Record<string, string>>({})

  const refreshFolders = useCallback(async () => {
    const all = await foldersService.getFolders()
    setFolders(all)
    const mapping: Record<string, string> = {}
    for (const s of sortedActiveSessions) {
      const fid = await foldersService.getFolderOf(s.id)
      if (fid) mapping[s.id] = fid
    }
    setFolderOf(mapping)
  }, [sortedActiveSessions])

  useEffect(() => {
    if (visible) {
      refreshFolders()
    }
  }, [visible, refreshFolders])

  const handleNewFolder = useCallback(() => {
    Alert.prompt(
      'New Folder',
      'Enter a folder name:',
      async (name: string) => {
        if (!name || !name.trim()) return
        try {
          await foldersService.createFolder(name.trim())
          await refreshFolders()
          triggerHaptic('light')
        } catch (e: any) {
          toast.error(e?.message || 'Failed to create folder')
        }
      },
      'plain-text',
    )
  }, [refreshFolders])

  const handleMoveToFolder = useCallback(
    (session: Session) => {
      const options: { text: string; style?: 'cancel' | 'destructive'; onPress?: () => void }[] =
        folders.map((f) => ({
          text: f.name,
          onPress: async () => {
            try {
              await foldersService.moveSessionToFolder(session.id, f.id)
              await refreshFolders()
              triggerHaptic('light')
            } catch {
              toast.error('Failed to move session')
            }
          },
        }))
      options.push({
        text: 'Unfile',
        onPress: async () => {
          await foldersService.moveSessionToFolder(session.id, null)
          await refreshFolders()
        },
      })
      options.push({ text: 'Cancel', style: 'cancel' as const })
      Alert.alert('Move to Folder', 'Choose a folder:', options)
    },
    [folders, refreshFolders],
  )

  const handleSessionMenu = useCallback(
    (session: Session) => {
      const currentTitle = session.name || 'New conversation'
      Alert.alert(currentTitle, undefined, [
        {
          text: 'Rename',
          onPress: () => {
            if (Platform.OS === 'android') {
              Alert.prompt(
                'Rename',
                'Enter a new title:',
                (newTitle: string) => {
                  if (newTitle && newTitle.trim() && newTitle.trim() !== currentTitle) {
                    renameSession(session.id, newTitle.trim())
                  }
                },
                'plain-text',
                currentTitle,
              )
            }
          },
        },
        { text: 'Move to Folder', onPress: () => handleMoveToFolder(session) },
        {
          text: 'Clone',
          onPress: async () => {
            const newId = await cloneSession(session.id, `${currentTitle} (Copy)`)
            if (newId) {
              await loadSession(newId)
              onClose()
            }
          },
        },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => {
            Alert.alert('Delete', 'Delete this conversation?', [
              { text: 'Cancel', style: 'cancel' },
              {
                text: 'Delete',
                style: 'destructive',
                onPress: () => deleteSession(session.id),
              },
            ])
          },
        },
        { text: 'Cancel', style: 'cancel' },
      ])
    },
    [renameSession, handleMoveToFolder, cloneSession, loadSession, onClose, deleteSession],
  )

  return (
    <Modal visible={visible} animationType="slide" transparent>
      <TamaguiProvider>
        <YStack flex={1} justifyContent="flex-end">
          <YStack flex={1} backgroundColor={colors.overlay(0.3)} onPress={onClose} />
          <YStack
            backgroundColor="$background"
            borderTopLeftRadius={R.lg}
            borderTopRightRadius={R.lg}
            maxHeight="85%"
            overflow="hidden"
            shadowColor="black"
            shadowOffset={{ width: 0, height: -4 }}
            shadowOpacity={0.12}
            shadowRadius={16}
            elevation={8}
          >
            {/* Handle bar */}
            <YStack alignItems="center" paddingTop={S.md} paddingBottom={S.sm}>
              <YStack width={40} height={5} borderRadius={R.xs} backgroundColor={colors.border} />
            </YStack>

            <XStack
              paddingHorizontal={S.xl}
              paddingVertical={S.lg}
              paddingBottom={S.lg}
              borderBottomWidth={1}
              borderBottomColor={colors.border}
              alignItems="center"
              justifyContent="space-between"
            >
              <Text {...T.page} fontWeight="700" letterSpacing={-0.5} color="$color">
                Conversations
              </Text>
              <YStack
                width={32}
                height={32}
                borderRadius={R.md}
                alignItems="center"
                justifyContent="center"
                backgroundColor={colors.primaryAlpha(0.08)}
                onPress={onClose}
                pressStyle={{ opacity: 0.6, scale: 0.95 }}
              >
                <Icon name="x" size={16} color={colors.primary} />
              </YStack>
            </XStack>

            {/* Folder chips */}
            {(folders.length > 0 || true) && (
              <XStack
                flexWrap="wrap"
                gap={5}
                paddingHorizontal={16}
                paddingTop={10}
                paddingBottom={4}
                alignItems="center"
              >
                {folders.map((f) => (
                  <YStack
                    key={f.id}
                    testID={`folder-chip-${f.id}`}
                    paddingHorizontal={12}
                    paddingVertical={5}
                    borderRadius={10}
                    backgroundColor={
                      folderFilter === f.id ? colors.primary : colors.primaryAlpha(0.06)
                    }
                    borderWidth={1}
                    borderColor={folderFilter === f.id ? colors.primary : colors.border}
                    pressStyle={{ opacity: 0.8 }}
                    onPress={() => setFolderFilter(folderFilter === f.id ? null : f.id)}
                  >
                    <Text
                      fontSize={11}
                      color={folderFilter === f.id ? 'white' : '$color11'}
                      fontWeight={folderFilter === f.id ? '600' : '500'}
                    >
                      {f.name}
                    </Text>
                  </YStack>
                ))}
                <YStack
                  testID="new-folder"
                  width={28}
                  height={28}
                  borderRadius={10}
                  alignItems="center"
                  justifyContent="center"
                  backgroundColor={colors.primaryAlpha(0.08)}
                  onPress={handleNewFolder}
                  pressStyle={{ opacity: 0.6 }}
                >
                  <Icon name="plus" size={14} color={colors.primary} />
                </YStack>
              </XStack>
            )}

            {/* Session list */}
            <FlatList
              data={sortedActiveSessions.filter((s) => {
                if (labelFilter && !(sessionLabels[s.id] || []).includes(labelFilter)) {
                  return false
                }
                if (folderFilter) {
                  return folderOf[s.id] === folderFilter
                }
                return true
              })}
              keyExtractor={(item) => item.id}
              renderItem={({ item: session }) => {
                const isStarred = starredIds.includes(session.id)
                const isActive = session.id === useChatStore.getState().activeSessionId
                const folderName = folders.find((f) => f.id === folderOf[session.id])?.name
                return (
                  <XStack
                    paddingHorizontal={S.lg}
                    paddingVertical={S.md}
                    marginHorizontal={S.md}
                    marginVertical={2}
                    borderRadius={R.lg}
                    alignItems="center"
                    justifyContent="space-between"
                    backgroundColor={isActive ? colors.primaryAlpha(0.1) : 'transparent'}
                    borderWidth={isActive ? 1 : 0}
                    borderColor={isActive ? colors.primaryAlpha(0.2) : 'transparent'}
                    pressStyle={{ opacity: 0.7, scale: 0.98 }}
                    onPress={() => {
                      loadSession(session.id)
                      onClose()
                    }}
                  >
                    <YStack flex={1} marginRight={S.sm}>
                      <YStack
                        onLongPress={() => {
                          triggerHaptic('light')
                          handleSessionMenu(session)
                        }}
                      >
                        <XStack alignItems="center" gap={S.sm}>
                          {isStarred && <Icon name="star" size={12} color={colors.warning} />}
                          <Text
                            {...T.body}
                            fontWeight={isActive ? '600' : '400'}
                            color="$color"
                            numberOfLines={1}
                          >
                            {session.name || 'New conversation'}
                          </Text>
                        </XStack>
                      </YStack>
                      <XStack alignItems="center" gap={S.sm} marginTop={S.xs}>
                        <Text {...T.caption} color="$color10">
                          {session.message_count || 0} messages
                        </Text>
                        {folderName && (
                          <YStack
                            backgroundColor={colors.primaryAlpha(0.1)}
                            paddingHorizontal={6}
                            paddingVertical={2}
                            borderRadius={6}
                          >
                            <Text fontSize={9} fontWeight="600" color={colors.primary}>
                              {folderName}
                            </Text>
                          </YStack>
                        )}
                        {(sessionLabels[session.id] || []).length > 0 && (
                          <XStack flexWrap="wrap" gap={S.xs}>
                            {(sessionLabels[session.id] || []).slice(0, 2).map((label) => (
                              <YStack
                                key={label}
                                backgroundColor={colors.primaryAlpha(0.1)}
                                paddingHorizontal={S.sm}
                                paddingVertical={2}
                                borderRadius={R.sm}
                              >
                                <Text {...T.badge} fontWeight="600" color={colors.primary}>
                                  {label}
                                </Text>
                              </YStack>
                            ))}
                          </XStack>
                        )}
                      </XStack>
                    </YStack>
                    <XStack alignItems="center" gap={2}>
                      <YStack
                        width={28}
                        height={28}
                        borderRadius={R.md}
                        alignItems="center"
                        justifyContent="center"
                        onPress={async () => {
                          try {
                            if (isStarred) {
                              await starsService.unstarSession(session.id)
                              setStarredIds((prev) => prev.filter((id) => id !== session.id))
                            } else {
                              await starsService.starSession(session.id)
                              setStarredIds((prev) => [session.id, ...prev])
                            }
                            triggerHaptic('light')
                          } catch (e) {
                            toast.error('Failed to update star')
                          }
                        }}
                        pressStyle={{ opacity: 0.6 }}
                      >
                        <Icon
                          name={isStarred ? 'star' : 'star-outline'}
                          size={14}
                          color={colors.warning}
                        />
                      </YStack>
                      <YStack
                        width={28}
                        height={28}
                        borderRadius={R.md}
                        alignItems="center"
                        justifyContent="center"
                        onPress={() => {
                          Alert.alert('Delete', 'Delete this conversation?', [
                            { text: 'Cancel', style: 'cancel' },
                            {
                              text: 'Delete',
                              style: 'destructive',
                              onPress: () => deleteSession(session.id),
                            },
                          ])
                        }}
                        pressStyle={{ opacity: 0.6 }}
                      >
                        <Icon name="trash-2" size={14} color={colors.error} />
                      </YStack>
                    </XStack>
                  </XStack>
                )
              }}
              ListEmptyComponent={
                <YStack padding={48} alignItems="center" gap={S.sm}>
                  <YStack
                    width={48}
                    height={48}
                    borderRadius={R.lg}
                    backgroundColor={colors.primaryAlpha(0.08)}
                    alignItems="center"
                    justifyContent="center"
                  >
                    <Icon name="message-circle" size={24} color={colors.primary} />
                  </YStack>
                  <Text {...T.body} fontWeight="600" color="$color" marginTop={S.xs}>
                    No conversations yet
                  </Text>
                  <Text {...T.caption} color="$color10">
                    Start a new chat to begin
                  </Text>
                </YStack>
              }
              removeClippedSubviews
              maxToRenderPerBatch={10}
              windowSize={11}
            />
            {archivedSessions.length > 0 && (
              <XStack
                paddingHorizontal={S.lg}
                paddingVertical={S.md}
                borderTopWidth={1}
                borderTopColor={colors.border}
                alignItems="center"
                justifyContent="space-between"
                pressStyle={{ backgroundColor: colors.primaryAlpha(0.04) }}
                onPress={() => setShowArchived(!showArchived)}
              >
                <XStack alignItems="center" gap={S.md}>
                  <YStack
                    width={28}
                    height={28}
                    borderRadius={R.md}
                    backgroundColor={colors.primaryAlpha(0.08)}
                    alignItems="center"
                    justifyContent="center"
                  >
                    <Icon name="archive" size={14} color={colors.primary} />
                  </YStack>
                  <Text {...T.label} fontWeight="600" color="$color">
                    Archived
                  </Text>
                  <YStack
                    paddingHorizontal={S.sm}
                    paddingVertical={2}
                    borderRadius={R.md}
                    backgroundColor={colors.primaryAlpha(0.1)}
                  >
                    <Text {...T.badge} fontWeight="700" color={colors.primary}>
                      {archivedSessions.length}
                    </Text>
                  </YStack>
                </XStack>
                <Icon name="chevron-down" size={14} color={colors.textMuted} />
              </XStack>
            )}
            {showArchived && archivedSessions.length > 0 && (
              <YStack maxHeight={200}>
                <FlatList
                  data={archivedSessions}
                  keyExtractor={(item) => item.id}
                  renderItem={({ item: session }) => (
                    <XStack
                      paddingHorizontal={S.lg}
                      paddingVertical={S.md}
                      marginHorizontal={S.md}
                      marginVertical={1}
                      borderRadius={R.md}
                      alignItems="center"
                      justifyContent="space-between"
                      backgroundColor={
                        session.id === useChatStore.getState().activeSessionId
                          ? colors.primaryAlpha(0.08)
                          : 'transparent'
                      }
                      pressStyle={{ opacity: 0.7 }}
                      onPress={() => {
                        loadSession(session.id)
                        onClose()
                      }}
                    >
                      <YStack flex={1}>
                        <Text {...T.caption} color="$color" numberOfLines={1}>
                          {session.name || 'New conversation'}
                        </Text>
                        <Text {...T.caption} color="$color10">
                          {session.message_count || 0} messages
                        </Text>
                      </YStack>
                      <YStack
                        paddingHorizontal={S.md}
                        paddingVertical={S.xs}
                        borderRadius={R.md}
                        backgroundColor={colors.primaryAlpha(0.08)}
                        borderWidth={1}
                        borderColor={colors.primaryAlpha(0.15)}
                        pressStyle={{ opacity: 0.7 }}
                        onPress={() => archiveSession(session.id, false)}
                      >
                        <Text {...T.badge} fontWeight="600" color={colors.primary}>
                          Restore
                        </Text>
                      </YStack>
                    </XStack>
                  )}
                />
              </YStack>
            )}
            {/* Label filter chips — below list */}
            {allLabels.length > 0 && (
              <XStack
                flexWrap="wrap"
                gap={S.xs}
                paddingHorizontal={S.lg}
                paddingVertical={S.md}
                paddingBottom={S.lg}
                borderTopWidth={1}
                borderTopColor={colors.border}
              >
                <YStack
                  paddingHorizontal={S.md}
                  paddingVertical={S.xs}
                  borderRadius={R.md}
                  backgroundColor={labelFilter === null ? colors.primary : 'transparent'}
                  borderWidth={1}
                  borderColor={labelFilter === null ? colors.primary : colors.border}
                  pressStyle={{ opacity: 0.8 }}
                  onPress={() => setLabelFilter(null)}
                >
                  <Text
                    {...T.badge}
                    color={labelFilter === null ? 'white' : '$color11'}
                    fontWeight={labelFilter === null ? '600' : '500'}
                  >
                    All
                  </Text>
                </YStack>
                {allLabels.map((label) => (
                  <YStack
                    key={label}
                    paddingHorizontal={S.md}
                    paddingVertical={S.xs}
                    borderRadius={R.md}
                    backgroundColor={labelFilter === label ? colors.primary : 'transparent'}
                    borderWidth={1}
                    borderColor={labelFilter === label ? colors.primary : colors.border}
                    pressStyle={{ opacity: 0.8 }}
                    onPress={() => setLabelFilter(label === labelFilter ? null : label)}
                  >
                    <Text
                      {...T.badge}
                      color={labelFilter === label ? 'white' : '$color11'}
                      fontWeight={labelFilter === label ? '600' : '500'}
                    >
                      {label}
                    </Text>
                  </YStack>
                ))}
              </XStack>
            )}
            <YStack height={S.xl} />
          </YStack>
        </YStack>
      </TamaguiProvider>
    </Modal>
  )
}
