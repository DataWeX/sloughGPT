import { useState } from 'react'
import { KeyboardAvoidingView, Platform } from 'react-native'
import { useRouter } from 'expo-router'
import {
  YStack,
  XStack,
  Text,
  Input,
  Button,
  Card,
  H2,
  Paragraph,
  Separator,
} from 'tamagui'
import { useAuthStore } from '@/stores/auth-store'
import { apiPost } from '@/lib/api-client'

export default function LoginScreen() {
  const router = useRouter()
  const login = useAuthStore((s) => s.login)
  const [isRegister, setIsRegister] = useState(false)
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleSubmit() {
    if (!username.trim() || !password.trim()) {
      setError('Username and password are required')
      return
    }
    if (isRegister && !email.trim()) {
      setError('Email is required for registration')
      return
    }

    setLoading(true)
    setError('')

    try {
      const endpoint = isRegister ? '/auth/register' : '/auth/login'
      const body = isRegister
        ? { username: username.trim(), email: email.trim(), password }
        : { username: username.trim(), password }

      const data = await apiPost<{ token: string; user: { id: string; username: string; email: string } }>(
        endpoint,
        body
      )

      login(data.user, data.token)
      router.replace('/(tabs)/chat')
    } catch (err) {
      setError((err as Error).message || 'Authentication failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={{ flex: 1 }}
    >
      <YStack
        flex={1}
        justifyContent="center"
        paddingHorizontal="$4"
        backgroundColor="$background"
      >
        <YStack alignItems="center" marginBottom="$6">
          <YStack
            width={80}
            height={80}
            borderRadius={20}
            backgroundColor="$primary"
            alignItems="center"
            justifyContent="center"
            marginBottom="$4"
          >
            <Text fontSize={32} fontWeight="700" color="$background">
              SG
            </Text>
          </YStack>
          <H2 color="$color">SloughGPT</H2>
          <Paragraph color="$placeholderColor" marginTop="$1">
            Your personal AI companion
          </Paragraph>
        </YStack>

        <Card elevation={2} padding="$5" borderRadius="$6" backgroundColor="$background">
          <Card.Header>
            <YStack gap="$3">
              <Input
                size="$4"
                placeholder="Username"
                value={username}
                onChangeText={setUsername}
                autoCapitalize="none"
                autoCorrect={false}
              />

              {isRegister && (
                <Input
                  size="$4"
                  placeholder="Email"
                  value={email}
                  onChangeText={setEmail}
                  autoCapitalize="none"
                  keyboardType="email-address"
                />
              )}

              <Input
                size="$4"
                placeholder="Password"
                value={password}
                onChangeText={setPassword}
                secureTextEntry
              />

              {error ? (
                <Paragraph color="$destructive" size="$2">
                  {error}
                </Paragraph>
              ) : null}

              <Button
                size="$5"
                theme="active"
                onPress={handleSubmit}
                disabled={loading}
              >
                {loading ? 'Please wait...' : isRegister ? 'Create Account' : 'Sign In'}
              </Button>
            </YStack>
          </Card.Header>

          <Separator marginVertical="$3" />

          <Card.Footer>
            <XStack justifyContent="center" gap="$2">
              <Paragraph color="$placeholderColor" size="$2">
                {isRegister ? 'Already have an account?' : "Don't have an account?"}
              </Paragraph>
              <Button
                chromeless
                size="$2"
                onPress={() => {
                  setIsRegister(!isRegister)
                  setError('')
                }}
              >
                {isRegister ? 'Sign In' : 'Register'}
              </Button>
            </XStack>
          </Card.Footer>
        </Card>

        <Paragraph
          textAlign="center"
          color="$placeholderColor"
          size="$1"
          marginTop="$4"
        >
          Auth not required in dev mode
        </Paragraph>
      </YStack>
    </KeyboardAvoidingView>
  )
}
