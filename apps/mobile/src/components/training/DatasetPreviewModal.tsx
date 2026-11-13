import React from 'react';
import {Modal, ScrollView, Pressable} from 'react-native';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../../theme/colors';
import {Icon} from '../Icon';

interface Props {
  visible: boolean;
  onClose: () => void;
  data: string[];
}

export function DatasetPreviewModal({visible, onClose, data}: Props) {
  const colors = useColors();

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
          maxHeight="70%">
          <XStack
            alignItems="center"
            justifyContent="space-between"
            paddingHorizontal={20}
            paddingVertical={16}
            borderBottomWidth={1}
            borderBottomColor="$borderColor">
            <Text fontSize={16} fontWeight="600" color={colors.text}>
              Dataset Preview
            </Text>
            <Pressable onPress={onClose} accessibilityLabel="Close preview">
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
          <ScrollView style={{paddingHorizontal: 20, paddingVertical: 12}}>
            {data.map((line, i) => (
              <XStack
                key={i}
                gap={8}
                paddingVertical={4}
                borderBottomWidth={1}
                borderBottomColor="$borderColor">
                <Text
                  fontSize={11}
                  color={colors.textSecondary}
                  letterSpacing={0.2}
                  width={24}>
                  {i + 1}
                </Text>
                <Text
                  fontSize={13}
                  color={colors.text}
                  lineHeight={18}
                  flex={1}
                  numberOfLines={3}>
                  {line}
                </Text>
              </XStack>
            ))}
            {data.length === 0 && (
              <Text
                fontSize={13}
                color={colors.textSecondary}
                lineHeight={18}
                textAlign="center"
                padding={24}>
                No preview available
              </Text>
            )}
          </ScrollView>
        </YStack>
      </YStack>
    </Modal>
  );
}
