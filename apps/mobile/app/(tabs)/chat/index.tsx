import { useState, useRef, useEffect, useCallback } from 'react'
import {
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
} from 'react-native'
import {
  YStack,
  XStack,
  Text,
  Input,
  Button,
  Card,
  Paragraph,
  Separator,
  useTheme,
  AnimatePresence,
  Spinner,
  Sheet,
  ScrollView,
  Label,
} from 'tamagui'
import {
  Send,
  Plus,
  ChevronLeft,
  ThumbsUp,
  ThumbsDown,
  RefreshCw,
  Copy,
  Sparkles,
  Mic,
  Image as ImageIcon,
  Camera,
  X,
} from '@tamagui/lucide-icons'
import * as Haptics from 'expo-haptics'
import { Image } from 'react-native'
import { useChatStore, Message, Conversation } from '@/stores/chat-store'
import { useModelStore, SoulInfo } from '@/stores/model-store'
import { PerformanceTracker, Analytics } from '@/lib/analytics'
import { useVoiceInput } from '@/hooks/useVoiceInput'
import { useImageUpload } from '@/hooks/useImageUpload'

export default function ChatScreen() {
  const theme = useTheme()
  const flatListRef = useRef<FlatList>(null)
  const [inputText, setInputText] = useState('')
  const [showDrawer, setShowDrawer] = useState(false)
  const [showSoulPicker, setShowSoulPicker] = useState(false)
  const [showImagePicker, setShowImagePicker] = useState(false)
  const [actionMessageId, setActionMessageId] = useState<string | null>(null)

  const voiceInput = useVoiceInput()
  const imageUpload = useImageUpload()

  const {
    messages,
    streaming,
    error,
    activeSessionId,
    sessions,
    sendMessage,
    regenerate,
    cancelStream,
    recordFeedback,
    createSession,
    loadSession,
    deleteSession,
    refreshSessions,
    clearError,
  } = useChatStore()

  const { currentSoul, currentModel, souls, health, refresh: refreshModels } = useModelStore()

  useEffect(() => {
    const screenLoad = PerformanceTracker.trackScreenLoad('ChatScreen')
    refreshSessions()
    refreshModels()
    screenLoad.finish()
  }, [])

  useEffect(() => {
    if (messages.length > 0) {
      setTimeout(() => {
        flatListRef.current?.scrollToEnd({ animated: true })
      }, 100)
    }
  }, [messages.length, messages[messages.length - 1]?.content])

  const handleSend = useCallback(async () => {
    const text = inputText.trim()
    if (!text || streaming) return

    setInputText('')
    const hasImage = !!imageUpload.image
    imageUpload.clearImage()
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)
    
    // Track message sent
    Analytics.trackChatMessageSent({
      messageLength: text.length,
      hasImages: hasImage,
    })
    
    await PerformanceTracker.trackApiRequest('/chat/stream', 'POST', async () => {
      await sendMessage(text)
    })
  }, [inputText, streaming, sendMessage, currentModel, currentSoul, imageUpload])

  const handleVoiceInput = useCallback(async () => {
    if (voiceInput.isRecording) {
      const transcribedText = await voiceInput.stopRecording()
      if (transcribedText) {
        setInputText(prev => prev + (prev ? ' ' : '') + transcribedText)
      }
    } else {
      await voiceInput.startRecording()
    }
  }, [voiceInput])

  const handleImageSelect = useCallback(async () => {
    setShowImagePicker(false)
    await imageUpload.pickImage()
  }, [imageUpload])

  const handleCameraCapture = useCallback(async () => {
    setShowImagePicker(false)
    await imageUpload.takePhoto()
  }, [imageUpload])

  const handleNewChat = useCallback(async () => {
    Analytics.trackSessionCreated()
    await createSession()
    setShowDrawer(false)
  }, [createSession])

  const handleSelectConversation = useCallback(
    async (id: string) => {
      await loadSession(id)
      setShowDrawer(false)
    },
    [loadSession]
  )

  const handleFeedback = useCallback(
    async (messageId: string, positive: boolean) => {
      Haptics.notificationAsync(
        positive ? Haptics.NotificationFeedbackType.Success : Haptics.NotificationFeedbackType.Warning
      )
      
      // Track feedback
      Analytics.trackEvent('message_feedback', {
        messageId,
        positive,
        model: currentModel || undefined,
        soul: currentSoul || undefined,
      })
      
      await recordFeedback(messageId, positive)
      setActionMessageId(null)
    },
    [recordFeedback, currentModel, currentSoul]
  )

  const handleRegenerate = useCallback(
    async (messageId: string) => {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium)
      Analytics.trackEvent('message_regenerated', { messageId })
      setActionMessageId(null)
      await regenerate(messageId)
    },
    [regenerate]
  )

  const handleCopy = useCallback((content: string) => {
    // Clipboard.setString(content)
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success)
    setActionMessageId(null)
  }, [])

  function renderMessage({ item }: { item: Message }) {
    const isUser = item.role === 'user'
    const isStreaming = streaming && item === messages[messages.length - 1] && !isUser
    const showActions = actionMessageId === item.id && !isUser && !streaming

    return (
      <XStack
        paddingHorizontal="$3"
        paddingVertical="$1"
        justifyContent={isUser ? 'flex-end' : 'flex-start'}
      >
        <Pressable
          onLongPress={() => !isUser && !streaming && setActionMessageId(item.id)}
          style={{ maxWidth: '85%' }}
        >
          <Card
            backgroundColor={isUser ? '$primary' : '$backgroundStrong'}
            borderRadius="$5"
            padding="$3"
            elevation={!isUser ? 2 : 0}
          >
            <Text
              color={isUser ? '$background' : '$color'}
              fontSize="$3"
              lineHeight={20}
            >
              {item.content || (isStreaming ? '' : '...')}
            </Text>
            {isStreaming && (
              <XStack alignItems="center" gap="$1" marginTop="$1">
                <Spinner size="small" color="$primary" />
                <Text color="$placeholderColor" fontSize="$1">
                  thinking...
                </Text>
              </XStack>
            )}
          </Card>

          <AnimatePresence>
            {showActions && (
              <XStack
                gap="$1"
                marginTop="$1"
                justifyContent="flex-start"
                enterStyle={{ opacity: 0, scale: 0.9 }}
                exitStyle={{ opacity: 0, scale: 0.9 }}
               
              >
                <Button
                  size="$2"
                  chromeless
                  icon={<ThumbsUp size={14} />}
                  onPress={() => handleFeedback(item.id, true)}
                />
                <Button
                  size="$2"
                  chromeless
                  icon={<ThumbsDown size={14} />}
                  onPress={() => handleFeedback(item.id, false)}
                />
                <Button
                  size="$2"
                  chromeless
                  icon={<RefreshCw size={14} />}
                  onPress={() => handleRegenerate(item.id)}
                />
                <Button
                  size="$2"
                  chromeless
                  icon={<Copy size={14} />}
                  onPress={() => handleCopy(item.content)}
                />
              </XStack>
            )}
          </AnimatePresence>
        </Pressable>
      </XStack>
    )
  }

  function renderEmptyState() {
    return (
      <YStack flex={1} justifyContent="center" alignItems="center" paddingHorizontal="$6">
        <YStack
          width={64}
          height={64}
          borderRadius={20}
          backgroundColor="$primary"
          alignItems="center"
          justifyContent="center"
          marginBottom="$4"
          opacity={0.2}
        >
          <Sparkles size={32} color="$background" />
        </YStack>
        <Text fontSize="$5" fontWeight="600" color="$color" textAlign="center">
          Start a conversation
        </Text>
        <Paragraph
          color="$placeholderColor"
          textAlign="center"
          marginTop="$2"
          size="$3"
        >
          Ask anything — I'm here to help
        </Paragraph>

        <YStack gap="$2" marginTop="$5" width="100%">
          {['Tell me something interesting', 'Help me brainstorm', 'Explain a concept'].map(
            (suggestion) => (
              <Button
                key={suggestion}
                size="$3"
                chromeless
                backgroundColor="$backgroundStrong"
                borderRadius="$4"
                onPress={() => {
                  setInputText(suggestion)
                }}
              >
                <Text color="$color" fontSize="$3">
                  {suggestion}
                </Text>
              </Button>
            )
          )}
        </YStack>
      </YStack>
    )
  }

  return (
    <YStack flex={1} backgroundColor="$background">
      {/* Header */}
      <XStack
        paddingHorizontal="$3"
        paddingVertical="$2"
        alignItems="center"
        justifyContent="space-between"
        borderBottomWidth={1}
        borderBottomColor="$borderColor"
        backgroundColor="$background"
        paddingTop={Platform.OS === 'ios' ? 56 : 12}
      >
        <XStack alignItems="center" gap="$2">
          <Button
            size="$3"
            chromeless
            icon={<ChevronLeft size={20} />}
            onPress={() => setShowDrawer(true)}
          />
          <YStack>
            <Text fontSize="$4" fontWeight="600" color="$color">
              Chat
            </Text>
            {currentSoul && (
              <Pressable onPress={() => setShowSoulPicker(true)}>
                <XStack
                  backgroundColor="$primary"
                  borderRadius="$6"
                  paddingHorizontal="$2"
                  paddingVertical={2}
                  opacity={0.85}
                >
                  <Text color="$background" fontSize="$1" fontWeight="500">
                    {currentSoul}
                  </Text>
                </XStack>
              </Pressable>
            )}
          </YStack>
        </XStack>

        <XStack alignItems="center" gap="$2">
          <XStack
            width={8}
            height={8}
            borderRadius={4}
            backgroundColor={health?.status === 'healthy' ? '$success' : '$destructive'}
          />
          <Button size="$3" chromeless icon={<Plus size={20} />} onPress={handleNewChat} />
        </XStack>
      </XStack>

      {/* Error banner */}
      {error && (
        <XStack
          backgroundColor="$destructive"
          paddingHorizontal="$3"
          paddingVertical="$2"
          alignItems="center"
          justifyContent="space-between"
        >
          <Text color="$background" fontSize="$2" flex={1}>
            {error}
          </Text>
          <Button size="$1" chromeless onPress={clearError}>
            <Text color="$background" fontSize="$2">
              Dismiss
            </Text>
          </Button>
        </XStack>
      )}

      {/* Messages */}
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderMessage}
        keyExtractor={(item) => item.id}
        ListEmptyComponent={renderEmptyState}
        contentContainerStyle={{
          flexGrow: 1,
          paddingVertical: 12,
        }}
        onContentSizeChange={() => {
          if (messages.length > 0) {
            flatListRef.current?.scrollToEnd({ animated: false })
          }
        }}
      />

      {/* Image Preview */}
      {imageUpload.image && (
        <XStack
          paddingHorizontal="$3"
          paddingTop="$2"
          backgroundColor="$background"
          borderTopWidth={1}
          borderTopColor="$borderColor"
        >
          <YStack position="relative">
            <Image
              source={{ uri: imageUpload.image }}
              style={{ width: 80, height: 80, borderRadius: 8 }}
            />
            <Pressable
              onPress={imageUpload.clearImage}
              style={{
                position: 'absolute',
                top: -8,
                right: -8,
                backgroundColor: theme.destructive?.val || '#DC505A',
                borderRadius: 12,
                width: 24,
                height: 24,
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <X size={14} color="$background" />
            </Pressable>
          </YStack>
        </XStack>
      )}

      {/* Input bar */}
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
      >
        <XStack
          paddingHorizontal="$3"
          paddingVertical="$2"
          alignItems="flex-end"
          gap="$2"
          borderTopWidth={1}
          borderTopColor="$borderColor"
          backgroundColor="$background"
          paddingBottom={Platform.OS === 'ios' ? 34 : 8}
        >
          {/* Voice Input Button */}
          <Button
            size="$4"
            circular
            chromeless
            onPress={handleVoiceInput}
            backgroundColor={voiceInput.isRecording ? '$destructive' : 'transparent'}
            icon={
              voiceInput.isProcessing ? (
                <Spinner size="small" color="$primary" />
              ) : (
                <Mic
                  size={20}
                  color={voiceInput.isRecording ? '$background' : '$color'}
                />
              )
            }
          />

          {/* Image Upload Button */}
          <Button
            size="$4"
            circular
            chromeless
            onPress={() => setShowImagePicker(true)}
            icon={<ImageIcon size={20} color="$color" />}
          />

          <Input
            flex={1}
            size="$4"
            placeholder="Message..."
            value={inputText}
            onChangeText={setInputText}
            multiline
            maxLength={4000}
            onSubmitEditing={handleSend}
            blurOnSubmit={false}
          />
          {streaming ? (
            <Button
              size="$4"
              circular
              backgroundColor="$destructive"
              onPress={cancelStream}
              icon={<Text color="$background">■</Text>}
            />
          ) : (
            <Button
              size="$4"
              circular
              theme="active"
              disabled={!inputText.trim() && !imageUpload.image}
              onPress={handleSend}
              icon={<Send size={18} color="$background" />}
              opacity={inputText.trim() || imageUpload.image ? 1 : 0.4}
            />
          )}
        </XStack>
      </KeyboardAvoidingView>

      {/* Conversation Drawer */}
      <Sheet
        modal
        open={showDrawer}
        onOpenChange={setShowDrawer}
        snapPoints={[85]}
        snapPointsMode="percent"
        dismissOnSnapToBottom
      >
        <Sheet.Overlay backgroundColor="rgba(0,0,0,0.4)" />
        <Sheet.Frame backgroundColor="$background" padding="$4">
          <YStack flex={1}>
            <XStack justifyContent="space-between" alignItems="center" marginBottom="$3">
              <Text fontSize="$6" fontWeight="700" color="$color">
                Conversations
              </Text>
              <Button size="$3" theme="active" icon={<Plus size={16} />} onPress={handleNewChat}>
                New
              </Button>
            </XStack>

            <ScrollView flex={1}>
              {sessions.length === 0 ? (
                <YStack alignItems="center" paddingVertical="$6">
                  <Paragraph color="$placeholderColor">No conversations yet</Paragraph>
                </YStack>
              ) : (
                <YStack gap="$2">
                  {sessions.map((session) => {
                    const lastMsg =
                      session.messages?.[session.messages.length - 1]?.content || ''
                    const isActive = session.id === activeSessionId

                    return (
                      <Card
                        key={session.id}
                        backgroundColor={isActive ? '$primary' : '$backgroundStrong'}
                        borderRadius="$4"
                        padding="$3"
                        pressStyle={{ opacity: 0.8 }}
                        onPress={() => handleSelectConversation(session.id)}
                      >
                        <Text
                          color={isActive ? '$background' : '$color'}
                          fontSize="$3"
                          fontWeight="500"
                          numberOfLines={1}
                        >
                          {session.title || 'New Chat'}
                        </Text>
                        <Text
                          color={isActive ? '$background' : '$placeholderColor'}
                          fontSize="$2"
                          numberOfLines={1}
                          marginTop="$1"
                        >
                          {lastMsg.slice(0, 80) || 'No messages'}
                        </Text>
                      </Card>
                    )
                  })}
                </YStack>
              )}
            </ScrollView>
          </YStack>
        </Sheet.Frame>
      </Sheet>

      {/* Soul Picker */}
      <Sheet
        modal
        open={showSoulPicker}
        onOpenChange={setShowSoulPicker}
        snapPoints={[60]}
        snapPointsMode="percent"
        dismissOnSnapToBottom
      >
        <Sheet.Overlay backgroundColor="rgba(0,0,0,0.4)" />
        <Sheet.Frame backgroundColor="$background" padding="$4">
          <YStack flex={1}>
            <Text fontSize="$6" fontWeight="700" color="$color" marginBottom="$3">
              Switch Personality
            </Text>
            <ScrollView flex={1}>
              <YStack gap="$2">
                {souls.map((soul) => {
                  const isActive = soul.name === currentSoul
                  return (
                    <Card
                      key={soul.name}
                      backgroundColor={isActive ? '$primary' : '$backgroundStrong'}
                      borderRadius="$4"
                      padding="$3"
                      pressStyle={{ opacity: 0.8 }}
                      onPress={async () => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)
                        await useModelStore.getState().switchSoul(soul.name)
                        setShowSoulPicker(false)
                      }}
                    >
                      <Text
                        color={isActive ? '$background' : '$color'}
                        fontSize="$4"
                        fontWeight="600"
                      >
                        {soul.name}
                      </Text>
                      <Text
                        color={isActive ? '$background' : '$placeholderColor'}
                        fontSize="$2"
                        marginTop="$1"
                      >
                        {soul.description}
                      </Text>
                      {soul.traits?.length > 0 && (
                        <XStack gap="$1" marginTop="$2" flexWrap="wrap">
                          {soul.traits.map((trait) => (
                            <XStack
                              key={trait}
                              backgroundColor={isActive ? '$background' : '$primary'}
                              borderRadius="$6"
                              paddingHorizontal="$2"
                              paddingVertical={1}
                              opacity={0.8}
                            >
                              <Text
                                color={isActive ? '$color' : '$background'}
                                fontSize="$1"
                              >
                                {trait}
                              </Text>
                            </XStack>
                          ))}
                        </XStack>
                      )}
                    </Card>
                  )
                })}
              </YStack>
            </ScrollView>
          </YStack>
        </Sheet.Frame>
      </Sheet>

      {/* Image Picker */}
      <Sheet
        modal
        open={showImagePicker}
        onOpenChange={setShowImagePicker}
        snapPoints={[30]}
        snapPointsMode="percent"
        dismissOnSnapToBottom
      >
        <Sheet.Overlay backgroundColor="rgba(0,0,0,0.4)" />
        <Sheet.Frame backgroundColor="$background" padding="$4">
          <YStack flex={1} gap="$3">
            <Text fontSize="$5" fontWeight="600" color="$color">
              Add Image
            </Text>
            <YStack gap="$2">
              <Button
                size="$5"
                icon={<ImageIcon size={20} />}
                onPress={handleImageSelect}
              >
                Choose from Library
              </Button>
              <Button
                size="$5"
                icon={<Camera size={20} />}
                onPress={handleCameraCapture}
              >
                Take Photo
              </Button>
            </YStack>
          </YStack>
        </Sheet.Frame>
      </Sheet>
    </YStack>
  )
}
