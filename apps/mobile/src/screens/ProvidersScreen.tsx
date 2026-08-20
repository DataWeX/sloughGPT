import React, {useState} from 'react';
import {
  ScrollView,
  TextInput as RNTextInput,
  Alert,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text, Input, Button} from 'tamagui';
import {useColors} from '../theme/colors';
import {useProvidersStore} from '../stores/providers-store';
import {useHybridStore} from '../stores/hybrid-inference-store';
import {PROVIDER_REGISTRY, PROVIDER_MODELS} from '../types/providers';
import type {ProviderId, ProviderConfig} from '../types/providers';
import {triggerHaptic} from '../services/haptics';
import {sounds} from '../services/sounds';
import {toast} from '../services/toast';

const PROVIDER_ORDER: ProviderId[] = [
  'openai', 'anthropic', 'google', 'mistral', 'groq', 'together', 'deepseek', 'openrouter', 'custom',
];

function _maskKey(key: string): string {
  if (!key) return '';
  if (key.length <= 8) return '••••••••';
  return key.slice(0, 4) + '••••' + key.slice(-4);
}

export function ProvidersScreen() {
  const colors = useColors();
  const providers = useProvidersStore(s => s.providers);
  const activeProviderId = useProvidersStore(s => s.activeProviderId);
  const setActiveProvider = useProvidersStore(s => s.setActiveProvider);
  const setApiKey = useProvidersStore(s => s.setApiKey);
  const setBaseUrl = useProvidersStore(s => s.setBaseUrl);
  const setDefaultModel = useProvidersStore(s => s.setDefaultModel);
  const hybrid = useHybridStore();

  const [expandedId, setExpandedId] = useState<ProviderId | null>(null);
  const [editingKey, setEditingKey] = useState<ProviderId | null>(null);
  const [keyInput, setKeyInput] = useState('');
  const [customModelInput, setCustomModelInput] = useState('');

  const _handleSelect = async (id: ProviderId) => {
    const config = providers[id];
    if (!config.apiKey) {
      toast.warn('Add an API key first');
      return;
    }
    triggerHaptic('selection');
    sounds.send();
    await setActiveProvider(id);
    await hybrid.setActiveEngine(id);
    toast.success(`Switched to ${config.name}`);
  };

  const _handleSaveKey = async (id: ProviderId) => {
    if (!keyInput.trim()) {
      toast.warn('API key cannot be empty');
      return;
    }
    triggerHaptic('success');
    sounds.receive();
    await setApiKey(id, keyInput.trim());
    setEditingKey(null);
    setKeyInput('');
    toast.success('API key saved');
  };

  const _handleTestKey = async (id: ProviderId) => {
    const config = providers[id];
    if (!config.apiKey) {
      toast.warn('No API key configured');
      return;
    }
    triggerHaptic('light');
    if (id === 'anthropic') {
      if (config.apiKey.startsWith('sk-ant-')) {
        toast.success('Key format looks valid');
      } else {
        toast.warn('Anthropic keys usually start with sk-ant-');
      }
    } else if (id === 'google') {
      if (config.apiKey.length > 20) {
        toast.success('Key format looks valid');
      } else {
        toast.warn('Google API keys are typically longer');
      }
    } else {
      if (config.apiKey.startsWith('sk-') || config.apiKey.length > 10) {
        toast.success('Key format looks valid');
      } else {
        toast.warn('Key format may be invalid');
      }
    }
  };

  const _handleAddCustomModel = async (id: ProviderId) => {
    const model = customModelInput.trim();
    if (!model) return;
    const config = providers[id];
    const customModels = config.headers?.custom_models
      ? JSON.parse(config.headers.custom_models)
      : [];
    if (!customModels.includes(model)) {
      customModels.push(model);
      await useProvidersStore.getState().updateProvider(id, {
        headers: {...config.headers, custom_models: JSON.stringify(customModels)},
      });
    }
    await setDefaultModel(id, model);
    setCustomModelInput('');
    toast.success(`Using ${model}`);
  };

  return (
    <SafeAreaView style={{flex: 1}} edges={['top']}>
      <KeyboardAvoidingView
        style={{flex: 1}}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <ScrollView contentContainerStyle={{padding: 16}}>
          {/* Header */}
          <YStack marginBottom={16}>
            <Text fontSize={22} fontWeight="600" color={colors.text}>
              Providers
            </Text>
            <Text fontSize={13} color={colors.textMuted} marginTop={4}>
              Connect third-party AI providers for remote inference
            </Text>
          </YStack>

          {/* Active provider indicator */}
          {activeProviderId && (
            <YStack
              backgroundColor="$color4"
              borderRadius={12}
              padding={12}
              marginBottom={16}
              borderWidth={1}
              borderColor="$color6"
            >
              <XStack alignItems="center" gap={8}>
                <YStack
                  width={8}
                  height={8}
                  borderRadius={4}
                  backgroundColor="$green10"
                />
                <Text fontSize={13} fontWeight="500" color={colors.text}>
                  Active: {providers[activeProviderId]?.name || activeProviderId}
                </Text>
              </XStack>
            </YStack>
          )}

          {/* Provider cards */}
          {PROVIDER_ORDER.map(id => {
            const config = providers[id];
            const isExpanded = expandedId === id;
            const isActive = activeProviderId === id;
            const hasKey = !!config.apiKey;
            const registry = PROVIDER_REGISTRY[id];
            const models = PROVIDER_MODELS[id] || [];

            return (
              <YStack
                key={id}
                backgroundColor="$color2"
                borderRadius={12}
                borderWidth={1}
                borderColor={isActive ? '$color8' : '$color4'}
                marginBottom={12}
                overflow="hidden"
              >
                {/* Provider header */}
                <XStack
                  padding={12}
                  alignItems="center"
                  justifyContent="space-between"
                  onPress={() => {
                    triggerHaptic('light');
                    setExpandedId(isExpanded ? null : id);
                  }}
                >
                  <YStack flex={1}>
                    <XStack alignItems="center" gap={8}>
                      <Text fontSize={15} fontWeight="600" color={colors.text}>
                        {config.name}
                      </Text>
                      {isActive && (
                        <YStack
                          backgroundColor="$green10"
                          paddingHorizontal={6}
                          paddingVertical={2}
                          borderRadius={4}
                        >
                           <Text fontSize={10} fontWeight="600" color="$color1">
                            ACTIVE
                          </Text>
                        </YStack>
                      )}
                      {hasKey && !isActive && (
                        <YStack
                          backgroundColor="$color5"
                          paddingHorizontal={6}
                          paddingVertical={2}
                          borderRadius={4}
                        >
                          <Text fontSize={10} fontWeight="500" color="$color10">
                            CONFIGURED
                          </Text>
                        </YStack>
                      )}
                    </XStack>
                    <Text fontSize={12} color={colors.textMuted} marginTop={2}>
                      {hasKey ? _maskKey(config.apiKey) : 'No API key'}
                      {' · '}
                      {config.defaultModel}
                    </Text>
                  </YStack>
                  <Text fontSize={18} color={colors.textMuted}>
                    {isExpanded ? '▾' : '▸'}
                  </Text>
                </XStack>

                {/* Expanded details */}
                {isExpanded && (
                  <YStack padding={12} paddingTop={0} gap={12}>
                    {/* Base URL */}
                    <YStack gap={4}>
                      <Text fontSize={11} fontWeight="500" color={colors.textMuted} textTransform="uppercase" letterSpacing={0.5}>
                        Base URL
                      </Text>
                      <Input
                        value={config.baseUrl}
                        onChangeText={v => setBaseUrl(id, v)}
                        placeholder={registry.baseUrl}
                        fontSize={13}
                        backgroundColor="$color3"
                        borderColor="$color5"
                        color={colors.text}
                        autoCapitalize="none"
                        autoCorrect={false}
                      />
                    </YStack>

                    {/* API Key */}
                    <YStack gap={4}>
                      <Text fontSize={11} fontWeight="500" color={colors.textMuted} textTransform="uppercase" letterSpacing={0.5}>
                        API Key
                      </Text>
                      {editingKey === id ? (
                        <XStack gap={8}>
                          <Input
                            value={keyInput}
                            onChangeText={setKeyInput}
                            placeholder="Enter API key"
                            fontSize={13}
                            backgroundColor="$color3"
                            borderColor="$color8"
                            color={colors.text}
                            secureTextEntry
                            autoCapitalize="none"
                            autoCorrect={false}
                            flex={1}
                          />
                          <Button
                            size="$3"
                            backgroundColor="$color8"
                            color="white"
                            fontWeight="600"
                            onPress={() => _handleSaveKey(id)}
                          >
                            Save
                          </Button>
                          <Button
                            size="$3"
                            backgroundColor="$color4"
                            color="$color11"
                            fontWeight="600"
                            onPress={() => {
                              setEditingKey(null);
                              setKeyInput('');
                            }}
                          >
                            Cancel
                          </Button>
                        </XStack>
                      ) : (
                        <XStack gap={8}>
                          <Input
                            value={hasKey ? _maskKey(config.apiKey) : ''}
                            placeholder="Not configured"
                            fontSize={13}
                            backgroundColor="$color3"
                            borderColor="$color5"
                            color={colors.textMuted}
                            flex={1}
                            readOnly
                          />
                          <Button
                            size="$3"
                            backgroundColor="$color5"
                            color="$color11"
                            fontWeight="600"
                            onPress={() => {
                              setEditingKey(id);
                              setKeyInput(config.apiKey);
                            }}
                          >
                            {hasKey ? 'Edit' : 'Add'}
                          </Button>
                          {hasKey && (
                            <Button
                              size="$3"
                              backgroundColor="$color4"
                              color="$color11"
                              fontWeight="600"
                              onPress={() => _handleTestKey(id)}
                            >
                              Test
                            </Button>
                          )}
                        </XStack>
                      )}
                    </YStack>

                    {/* Default model */}
                    <YStack gap={4}>
                      <Text fontSize={11} fontWeight="500" color={colors.textMuted} textTransform="uppercase" letterSpacing={0.5}>
                        Default Model
                      </Text>
                      <XStack gap={8} flexWrap="wrap">
                        {models.map(m => (
                          <Button
                            key={m.id}
                            size="$2"
                            backgroundColor={
                              config.defaultModel === m.id ? '$color8' : '$color4'
                            }
                            color={config.defaultModel === m.id ? 'white' : '$color11'}
                            fontWeight="500"
                            borderRadius={999}
                            onPress={() => {
                              triggerHaptic('selection');
                              setDefaultModel(id, m.id);
                            }}
                          >
                            {m.name}
                          </Button>
                        ))}
                      </XStack>
                      {/* Custom model input */}
                      <XStack gap={8}>
                        <Input
                          value={customModelInput}
                          onChangeText={setCustomModelInput}
                          placeholder="Custom model ID"
                          fontSize={12}
                          backgroundColor="$color3"
                          borderColor="$color5"
                          color={colors.text}
                          flex={1}
                          autoCapitalize="none"
                          autoCorrect={false}
                        />
                        <Button
                          size="$2"
                          backgroundColor="$color5"
                          color="$color11"
                          fontWeight="500"
                          onPress={() => _handleAddCustomModel(id)}
                          disabled={!customModelInput.trim()}
                          opacity={customModelInput.trim() ? 1 : 0.5}
                        >
                          Use
                        </Button>
                      </XStack>
                    </YStack>

                    {/* Action buttons */}
                    <XStack gap={8} marginTop={4}>
                      {!isActive && hasKey && (
                        <Button
                          size="$3"
                          backgroundColor="$color8"
                          color="white"
                          fontWeight="600"
                          flex={1}
                          onPress={() => _handleSelect(id)}
                        >
                          Set as Active
                        </Button>
                      )}
                      {isActive && (
                        <Button
                          size="$3"
                          backgroundColor="$color4"
                          color="$color11"
                          fontWeight="600"
                          flex={1}
                          onPress={async () => {
                            triggerHaptic('light');
                            await setActiveProvider(null);
                            await hybrid.setActiveEngine('remote');
                            toast.info('Switched to self-hosted remote');
                          }}
                        >
                          Deactivate
                        </Button>
                      )}
                    </XStack>
                  </YStack>
                )}
              </YStack>
            );
          })}

          {/* Info note */}
          <YStack
            backgroundColor="$color3"
            borderRadius={8}
            padding={12}
            marginTop={4}
            marginBottom={32}
          >
            <Text fontSize={12} color={colors.textMuted} lineHeight={18}>
              API keys are stored locally on your device and never sent to our servers.
              Each provider requires its own account and billing.
            </Text>
          </YStack>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}
