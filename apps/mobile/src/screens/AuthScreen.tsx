import React, {useEffect, useState, useCallback} from 'react';
import {Pressable, TextInput, ActivityIndicator, ScrollView} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {toast} from '../services/toast';

type Mode = 'login' | 'register';

interface UserInfo {
  id: string;
  username: string;
  email: string;
}

const AUTH_TOKEN_KEY = '@sloughgpt/auth_token';

function Card({children, style}: {children: React.ReactNode; style?: any}) {
  const colors = useColors();
  return (
    <YStack
      padding={14}
      borderRadius={12}
      backgroundColor={colors.white}
      borderWidth={0.5}
      borderColor={colors.border}
      gap={8}
      {...style}>
      {children}
    </YStack>
  );
}

function InputField({
  value,
  onChangeText,
  placeholder,
  secureTextEntry,
  keyboardType,
}: {
  value: string;
  onChangeText: (t: string) => void;
  placeholder: string;
  secureTextEntry?: boolean;
  keyboardType?: 'default' | 'email-address';
}) {
  const colors = useColors();
  return (
    <TextInput
      value={value}
      onChangeText={onChangeText}
      placeholder={placeholder}
      placeholderTextColor={colors.textMuted}
      secureTextEntry={secureTextEntry}
      keyboardType={keyboardType}
      autoCapitalize="none"
      autoCorrect={false}
      style={{
        backgroundColor: colors.muted,
        borderRadius: 8,
        paddingHorizontal: 12,
        paddingVertical: 10,
        fontSize: 14,
        color: colors.text,
        borderWidth: 0.5,
        borderColor: colors.border,
      }}
    />
  );
}

export function AuthScreen() {
  const colors = useColors();
  const [mode, setMode] = useState<Mode>('login');
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentUser, setCurrentUser] = useState<UserInfo | null>(null);
  const [checking, setChecking] = useState(true);
  const [token, setToken] = useState<string | null>(null);

  useEffect(() => {
    AsyncStorage.getItem(AUTH_TOKEN_KEY).then(saved => {
      if (saved) {
        setToken(saved);
        api.get<UserInfo>('/auth/me')
          .then(d => setCurrentUser(d))
          .catch(() => {
            AsyncStorage.removeItem(AUTH_TOKEN_KEY);
            setToken(null);
          })
          .finally(() => setChecking(false));
      } else {
        setChecking(false);
      }
    });
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!username || !password) {
      setError('Username and password required');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = mode === 'login'
        ? await api.post<{token: string; user: UserInfo}>('/auth/login', {username, password})
        : await api.post<{token: string; user: UserInfo}>('/auth/register', {username, email, password});
      setToken(data.token);
      setCurrentUser(data.user);
      await AsyncStorage.setItem(AUTH_TOKEN_KEY, data.token);
      toast.success(`Logged in as ${data.user.username}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Connection failed');
    } finally {
      setLoading(false);
    }
  }, [mode, username, email, password]);

  const handleLogout = useCallback(() => {
    setToken(null);
    setCurrentUser(null);
    AsyncStorage.removeItem(AUTH_TOKEN_KEY);
    toast.info('Logged out');
  }, []);

  if (checking) {
    return (
      <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
        <YStack flex={1} alignItems="center" justifyContent="center">
          <ActivityIndicator size="large" color={colors.primary} />
        </YStack>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <YStack paddingHorizontal={16} paddingVertical={12}>
        <Text fontSize={20} fontWeight="600" color={colors.text}>Auth</Text>
        <Text fontSize={12} color={colors.textMuted}>
          {currentUser ? `Logged in as ${currentUser.username}` : 'Authentication'}
        </Text>
      </YStack>
      <ScrollView contentContainerStyle={{padding: 16, gap: 12}}>
        {/* Status */}
        <Card>
          <XStack gap={16} flexWrap="wrap">
            <YStack flex={1} alignItems="center" gap={4}>
              <Text fontSize={10} color={colors.textMuted}>STATUS</Text>
              <StatusBadge label={currentUser ? 'Logged In' : 'Guest'} variant={currentUser ? 'success' : 'default'} />
            </YStack>
            <YStack flex={1} alignItems="center" gap={4}>
              <Text fontSize={10} color={colors.textMuted}>USER</Text>
              <Text fontSize={13} fontWeight="500" color={colors.text}>{currentUser?.username ?? '—'}</Text>
            </YStack>
            <YStack flex={1} alignItems="center" gap={4}>
              <Text fontSize={10} color={colors.textMuted}>TOKEN</Text>
              <StatusBadge label={token ? 'Active' : 'None'} variant={token ? 'info' : 'default'} />
            </YStack>
          </XStack>
        </Card>

        {currentUser ? (
          <>
            {/* Current User */}
            <Card>
              <Text fontSize={13} fontWeight="600" color={colors.text}>Current User</Text>
              <YStack gap={8}>
                <XStack gap={16} flexWrap="wrap">
                  <YStack flex={1} backgroundColor={colors.muted} padding={10} borderRadius={8} alignItems="center">
                    <Text fontSize={10} color={colors.textMuted}>Username</Text>
                    <Text fontSize={13} fontWeight="500" color={colors.text}>{currentUser.username}</Text>
                  </YStack>
                  <YStack flex={1} backgroundColor={colors.muted} padding={10} borderRadius={8} alignItems="center">
                    <Text fontSize={10} color={colors.textMuted}>Email</Text>
                    <Text fontSize={13} fontWeight="500" color={colors.text}>{currentUser.email}</Text>
                  </YStack>
                </XStack>
                <YStack backgroundColor={colors.muted} padding={10} borderRadius={8} alignItems="center">
                  <Text fontSize={10} color={colors.textMuted}>User ID</Text>
                  <Text fontSize={11} fontWeight="500" color={colors.text} numberOfLines={1}>{currentUser.id}</Text>
                </YStack>
                <Pressable onPress={handleLogout}>
                  <YStack
                    backgroundColor={colors.errorAlpha(0.1)}
                    padding={10}
                    borderRadius={8}
                    alignItems="center"
                    pressStyle={{opacity: 0.6}}>
                    <Text fontSize={13} fontWeight="500" color={colors.error}>Logout</Text>
                  </YStack>
                </Pressable>
              </YStack>
            </Card>

            {/* Token Info */}
            <Card>
              <Text fontSize={13} fontWeight="600" color={colors.text}>Token Info</Text>
              <YStack backgroundColor={colors.muted} padding={10} borderRadius={8}>
                <Text fontSize={10} color={colors.textMuted} marginBottom={4}>JWT Token</Text>
                <Text fontSize={10} color={colors.textMuted} numberOfLines={2}>{token?.slice(0, 60)}...</Text>
              </YStack>
              <Pressable
                onPress={async () => {
                  try {
                    const data = await api.post<{valid: boolean}>('/auth/verify', undefined);
                    toast.info(data?.valid ? 'Token valid' : 'Token invalid');
                  } catch {
                    toast.error('Verification failed');
                  }
                }}>
                <YStack
                  backgroundColor={colors.muted}
                  padding={10}
                  borderRadius={8}
                  alignItems="center"
                  pressStyle={{opacity: 0.6}}>
                  <Text fontSize={12} fontWeight="500" color={colors.primary}>Verify Token</Text>
                </YStack>
              </Pressable>
            </Card>
          </>
        ) : (
          /* Login / Register Form */
          <Card>
            <Text fontSize={13} fontWeight="600" color={colors.text}>
              {mode === 'login' ? 'Login' : 'Register'}
            </Text>
            <YStack gap={8}>
              <InputField
                value={username}
                onChangeText={setUsername}
                placeholder="Username"
              />
              {mode === 'register' && (
                <InputField
                  value={email}
                  onChangeText={setEmail}
                  placeholder="Email"
                  keyboardType="email-address"
                />
              )}
              <InputField
                value={password}
                onChangeText={setPassword}
                placeholder="Password"
                secureTextEntry
              />
              {error && (
                <Text fontSize={12} color={colors.error}>{error}</Text>
              )}
              <XStack gap={12} alignItems="center">
                <Pressable
                  onPress={handleSubmit}
                  disabled={loading}
                  style={{flex: 1}}>
                  <YStack
                    backgroundColor={colors.primary}
                    padding={12}
                    borderRadius={8}
                    alignItems="center"
                    opacity={loading ? 0.6 : 1}
                    pressStyle={{opacity: 0.7}}>
                    {loading ? (
                      <ActivityIndicator size="small" color={colors.white} />
                    ) : (
                      <Text fontSize={14} fontWeight="600" color={colors.white}>
                        {mode === 'login' ? 'Login' : 'Register'}
                      </Text>
                    )}
                  </YStack>
                </Pressable>
                <Pressable
                  onPress={() => { setMode(mode === 'login' ? 'register' : 'login'); setError(null); }}>
                  <Text fontSize={12} fontWeight="500" color={colors.primary}>
                    {mode === 'login' ? 'Create account' : 'Already have an account?'}
                  </Text>
                </Pressable>
              </XStack>
            </YStack>
          </Card>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}
