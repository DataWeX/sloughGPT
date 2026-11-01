import React, {useState} from 'react';
import {Modal, TextInput, ScrollView, ActivityIndicator, Pressable} from 'react-native';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../../theme/colors';
import {Icon} from '../Icon';
import {useHapticPress} from '../../hooks/useHapticPress';
import {api} from '../../services/api-client';
import {toast} from '../../services/toast';

interface Props {
  visible: boolean;
  onClose: () => void;
  checkpoint?: string | null;
}

export function TestModelModal({visible, onClose, checkpoint}: Props) {
  const colors = useColors();
  const hapticPress = useHapticPress();
  const [testPrompt, setTestPrompt] = useState('');
  const [testResult, setTestResult] = useState('');
  const [testLoading, setTestLoading] = useState(false);

  const handleTestModel = async () => {
    if (!testPrompt.trim() || testLoading) return;
    setTestLoading(true);
    setTestResult('');
    try {
      const result = await api.post<{response: string}>('/training/test', {
        prompt: testPrompt,
        checkpoint: checkpoint,
      });
      setTestResult(result.response || 'No response');
    } catch (err: any) {
      setTestResult(`Error: ${err.message || 'Test failed'}`);
    } finally {
      setTestLoading(false);
    }
  };

  const handleClose = () => {
    onClose();
    setTestResult('');
    setTestPrompt('');
  };

  return (
    <Modal visible={visible} animationType="slide" transparent>
      <YStack
        flex={1}
        backgroundColor={colors.overlay(0.4)}
        justifyContent="flex-end">
        <YStack
          backgroundColor={colors.background}
          borderTopLeftRadius={24}
          borderTopRightRadius={24}
          maxHeight="80%">
          <XStack
            alignItems="center"
            justifyContent="space-between"
            paddingHorizontal={20}
            paddingVertical={16}
            borderBottomWidth={1}
            borderBottomColor="$borderColor">
            <Text fontSize={16} fontWeight="600" color={colors.text}>
              Test Model
            </Text>
            <Pressable
              onPress={hapticPress('light', handleClose)}
              accessibilityLabel="Close test">
              <YStack
                width={28}
                height={28}
                borderRadius={9}
                alignItems="center"
                justifyContent="center">
                <Icon name="x" size={16} color={colors.textSecondary} />
              </YStack>
            </Pressable>
          </XStack>
          <YStack padding={20} gap={12}>
            <YStack gap={4}>
              <Text fontSize={13} color={colors.textSecondary}>
                Prompt
              </Text>
              <TextInput
                value={testPrompt}
                onChangeText={setTestPrompt}
                placeholder="Type a prompt to test the trained model..."
                placeholderTextColor={colors.textMuted}
                autoCapitalize="none"
                autoCorrect={false}
                style={{
                  fontSize: 15,
                  color: colors.text,
                  backgroundColor: colors.primaryAlpha(0.04),
                  borderRadius: 8,
                  paddingHorizontal: 12,
                  paddingVertical: 10,
                  minHeight: 60,
                }}
              />
            </YStack>
            <YStack
              paddingVertical={12}
              borderRadius={10}
              alignItems="center"
              backgroundColor={
                testLoading || !testPrompt.trim()
                  ? colors.primaryAlpha(0.3)
                  : colors.primary
              }
              onPress={hapticPress('light', handleTestModel)}
              disabled={testLoading || !testPrompt.trim()}
              pressStyle={{opacity: 0.7}}>
              {testLoading ? (
                <ActivityIndicator color={colors.white} />
              ) : (
                <Text fontSize={14} fontWeight="600" color={colors.white}>
                  Generate
                </Text>
              )}
            </YStack>
            {testResult ? (
              <YStack gap={4}>
                <Text fontSize={13} color={colors.textSecondary}>
                  Response
                </Text>
                <ScrollView style={{maxHeight: 200}}>
                  <YStack
                    backgroundColor={colors.primaryAlpha(0.04)}
                    borderRadius={8}
                    padding={12}>
                    <Text
                      fontSize={14}
                      color={colors.text}
                      lineHeight={20}>
                      {testResult}
                    </Text>
                  </YStack>
                </ScrollView>
              </YStack>
            ) : null}
          </YStack>
        </YStack>
      </YStack>
    </Modal>
  );
}
