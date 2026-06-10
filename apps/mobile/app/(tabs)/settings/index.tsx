import { useState, useEffect } from 'react'
import { ScrollView, Linking } from 'react-native'
import { useRouter } from 'expo-router'
import {
  YStack,
  XStack,
  Text,
  Card,
  Button,
  Paragraph,
  Separator,
  Input,
  Slider,
  Switch,
  Label,
} from 'tamagui'
import {
  Server,
  Moon,
  Sun,
  Smartphone,
  Trash2,
  Activity,
  ExternalLink,
  Shield,
} from '@tamagui/lucide-icons'
import * as Haptics from 'expo-haptics'
import { useSettingsStore } from '@/stores/settings-store'
import { useModelStore } from '@/stores/model-store'
import { useAuthStore } from '@/stores/auth-store'
import { apiGet } from '@/lib/api-client'

interface HealthData {
  status: string
  model_loaded: boolean
  model_type?: string
  uptime_seconds?: number
  inference_count?: number
}

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

export default function SettingsScreen() {
  const router = useRouter()
  const { theme, temperature, maxTokens, memoryContext, setTheme, update, reset } =
    useSettingsStore()
  const { health, refresh: refreshModels } = useModelStore()
  const { isAuthenticated, user, logout } = useAuthStore()

  const [healthData, setHealthData] = useState<HealthData | null>(null)
  const [showClearConfirm, setShowClearConfirm] = useState(false)
  const [showResetConfirm, setShowResetConfirm] = useState(false)

  useEffect(() => {
    fetchHealth()
  }, [])

  async function fetchHealth() {
    try {
      const data = await apiGet<HealthData>('/health')
      setHealthData(data)
    } catch {
      // silent
    }
  }

  const handleLogout = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium)
    logout()
    router.replace('/(auth)/login')
  }

  return (
    <YStack flex={1} backgroundColor="$background">
      <XStack
        paddingHorizontal="$3"
        paddingVertical="$2"
        alignItems="center"
        borderBottomWidth={1}
        borderBottomColor="$borderColor"
        paddingTop={56}
      >
        <Text fontSize="$6" fontWeight="700" color="$color">
          Settings
        </Text>
      </XStack>

      <ScrollView contentContainerStyle={{ padding: 12 }}>
        <YStack gap="$3">
          {/* Server Status */}
          <Card elevation={2} borderRadius="$5" backgroundColor="$backgroundStrong" padding="$4">
            <XStack alignItems="center" gap="$2" marginBottom="$3">
              <Server size={18} color="$primary" />
              <Text fontSize="$4" fontWeight="600" color="$color">
                Server Status
              </Text>
            </XStack>

            <YStack gap="$2">
              <XStack justifyContent="space-between" alignItems="center">
                <Text color="$placeholderColor" fontSize="$2">
                  Connection
                </Text>
                <XStack alignItems="center" gap="$1">
                  <XStack
                    width={8}
                    height={8}
                    borderRadius={4}
                    backgroundColor={
                      healthData?.status === 'healthy' ? '$success' : '$destructive'
                    }
                  />
                  <Text color="$color" fontSize="$2">
                    {healthData?.status || 'checking...'}
                  </Text>
                </XStack>
              </XStack>

              {healthData?.model_type && (
                <XStack justifyContent="space-between" alignItems="center">
                  <Text color="$placeholderColor" fontSize="$2">
                    Model
                  </Text>
                  <Text color="$color" fontSize="$2">
                    {healthData.model_type}
                  </Text>
                </XStack>
              )}

              {healthData?.uptime_seconds != null && (
                <XStack justifyContent="space-between" alignItems="center">
                  <Text color="$placeholderColor" fontSize="$2">
                    Uptime
                  </Text>
                  <Text color="$color" fontSize="$2">
                    {formatUptime(healthData.uptime_seconds)}
                  </Text>
                </XStack>
              )}

              {healthData?.inference_count != null && (
                <XStack justifyContent="space-between" alignItems="center">
                  <Text color="$placeholderColor" fontSize="$2">
                    Inferences
                  </Text>
                  <Text color="$color" fontSize="$2">
                    {healthData.inference_count}
                  </Text>
                </XStack>
              )}
            </YStack>

            <Button
              size="$3"
              chromeless
              marginTop="$3"
              icon={<Activity size={14} />}
              onPress={() => {
                fetchHealth()
                refreshModels()
              }}
            >
              <Text color="$primary" fontSize="$2">
                Refresh
              </Text>
            </Button>
          </Card>

          {/* Appearance */}
          <Card elevation={2} borderRadius="$5" backgroundColor="$backgroundStrong" padding="$4">
            <XStack alignItems="center" gap="$2" marginBottom="$3">
              <Moon size={18} color="$primary" />
              <Text fontSize="$4" fontWeight="600" color="$color">
                Appearance
              </Text>
            </XStack>

            <XStack gap="$2">
              {(['light', 'dark', 'system'] as const).map((t) => {
                const isActive = theme === t
                const icons = {
                  light: <Sun size={16} />,
                  dark: <Moon size={16} />,
                  system: <Smartphone size={16} />,
                }
                return (
                  <Button
                    key={t}
                    flex={1}
                    size="$3"
                    borderRadius="$4"
                    backgroundColor={isActive ? '$primary' : '$background'}
                    pressStyle={{ opacity: 0.8 }}
                    onPress={() => {
                      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)
                      setTheme(t)
                    }}
                  >
                    <XStack alignItems="center" gap="$1">
                      <Text color={isActive ? '$background' : '$color'}>
                        {icons[t]}
                      </Text>
                      <Text
                        color={isActive ? '$background' : '$color'}
                        fontSize="$2"
                        fontWeight="500"
                        textTransform="capitalize"
                      >
                        {t}
                      </Text>
                    </XStack>
                  </Button>
                )
              })}
            </XStack>
          </Card>

          {/* Chat Defaults */}
          <Card elevation={2} borderRadius="$5" backgroundColor="$backgroundStrong" padding="$4">
            <Text fontSize="$4" fontWeight="600" color="$color" marginBottom="$3">
              Chat Defaults
            </Text>

            <YStack gap="$3">
              <YStack gap="$1">
                <XStack justifyContent="space-between">
                  <Label>Temperature</Label>
                  <Text color="$primary" fontSize="$2" fontWeight="600">
                    {temperature.toFixed(1)}
                  </Text>
                </XStack>
                <Slider
                  value={[temperature]}
                  onValueChange={([v]) => update({ temperature: v })}
                  min={0}
                  max={2}
                  step={0.1}
                />
              </YStack>

              <YStack gap="$1">
                <XStack justifyContent="space-between">
                  <Label>Max Tokens</Label>
                  <Text color="$primary" fontSize="$2" fontWeight="600">
                    {maxTokens}
                  </Text>
                </XStack>
                <Slider
                  value={[maxTokens]}
                  onValueChange={([v]) => update({ maxTokens: Math.round(v) })}
                  min={64}
                  max={2048}
                  step={64}
                />
              </YStack>
            </YStack>
          </Card>

          {/* Memory Context */}
          <Card elevation={2} borderRadius="$5" backgroundColor="$backgroundStrong" padding="$4">
            <Text fontSize="$4" fontWeight="600" color="$color" marginBottom="$2">
              Memory Context
            </Text>
            <Paragraph color="$placeholderColor" fontSize="$2" marginBottom="$2">
              Custom context the AI will always remember
            </Paragraph>
            <Input
              size="$4"
              placeholder="e.g. I prefer concise answers..."
              value={memoryContext}
              onChangeText={(text) => update({ memoryContext: text })}
              multiline
              numberOfLines={3}
              minHeight={80}
            />
          </Card>

          {/* Account */}
          {isAuthenticated && (
            <Card elevation={2} borderRadius="$5" backgroundColor="$backgroundStrong" padding="$4">
              <XStack alignItems="center" gap="$2" marginBottom="$3">
                <Shield size={18} color="$primary" />
                <Text fontSize="$4" fontWeight="600" color="$color">
                  Account
                </Text>
              </XStack>
              <XStack justifyContent="space-between" alignItems="center">
                <Text color="$color" fontSize="$3">
                  {user?.username || 'Unknown'}
                </Text>
                <Button size="$3" chromeless onPress={handleLogout}>
                  <Text color="$destructive" fontSize="$2">
                    Sign Out
                  </Text>
                </Button>
              </XStack>
            </Card>
          )}

          {/* Danger Zone */}
          <Card borderRadius="$5" backgroundColor="$backgroundStrong" padding="$4" borderWidth={1} borderColor="$destructive" opacity={0.8}>
            <XStack alignItems="center" gap="$2" marginBottom="$3">
              <Trash2 size={18} color="$destructive" />
              <Text fontSize="$4" fontWeight="600" color="$destructive">
                Danger Zone
              </Text>
            </XStack>

            <YStack gap="$2">
              {showClearConfirm ? (
                <YStack gap="$2">
                  <Paragraph color="$destructive" fontSize="$2">
                    This will clear all local chat history. Continue?
                  </Paragraph>
                  <XStack gap="$2">
                    <Button
                      size="$3"
                      backgroundColor="$destructive"
                      flex={1}
                      onPress={() => {
                        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning)
                        setShowClearConfirm(false)
                      }}
                    >
                      <Text color="$background" fontSize="$2">
                        Yes, clear
                      </Text>
                    </Button>
                    <Button
                      size="$3"
                      chromeless
                      flex={1}
                      onPress={() => setShowClearConfirm(false)}
                    >
                      Cancel
                    </Button>
                  </XStack>
                </YStack>
              ) : (
                <Button
                  size="$3"
                  chromeless
                  onPress={() => setShowClearConfirm(true)}
                >
                  <Text color="$destructive" fontSize="$2">
                    Clear chat history
                  </Text>
                </Button>
              )}

              {showResetConfirm ? (
                <YStack gap="$2">
                  <Paragraph color="$destructive" fontSize="$2">
                    Reset all settings to defaults?
                  </Paragraph>
                  <XStack gap="$2">
                    <Button
                      size="$3"
                      backgroundColor="$destructive"
                      flex={1}
                      onPress={() => {
                        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning)
                        reset()
                        setShowResetConfirm(false)
                      }}
                    >
                      <Text color="$background" fontSize="$2">
                        Yes, reset
                      </Text>
                    </Button>
                    <Button
                      size="$3"
                      chromeless
                      flex={1}
                      onPress={() => setShowResetConfirm(false)}
                    >
                      Cancel
                    </Button>
                  </XStack>
                </YStack>
              ) : (
                <Button
                  size="$3"
                  chromeless
                  onPress={() => setShowResetConfirm(true)}
                >
                  <Text color="$destructive" fontSize="$2">
                    Reset all settings
                  </Text>
                </Button>
              )}
            </YStack>
          </Card>
        </YStack>
      </ScrollView>
    </YStack>
  )
}
