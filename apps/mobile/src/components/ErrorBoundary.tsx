import React from 'react';
import {Pressable} from 'react-native';
import {YStack, Text} from 'tamagui';

interface Props {
  children: React.ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {hasError: false, error: null};
  }

  static getDerivedStateFromError(error: Error): State {
    return {hasError: true, error};
  }

  render() {
    if (this.state.hasError) {
      return (
        <YStack flex={1} backgroundColor="#110F18" alignItems="center" justifyContent="center" padding={32}>
          <Text fontSize={48} marginBottom={16} color="#E8A83C" fontWeight="700">!</Text>
          <Text fontSize={20} fontWeight="600" color="#F0ECF5" marginBottom={8}>
            Something went wrong
          </Text>
          <Text fontSize={13} color="#827A96" textAlign="center" lineHeight={18} marginBottom={24}>
            {this.state.error?.message}
          </Text>
          <Pressable onPress={() => this.setState({hasError: false, error: null})}>
            <YStack backgroundColor="#7C52C4" paddingHorizontal={20} paddingVertical={12} borderRadius={12}>
              <Text fontSize={15} fontWeight="600" color="#FFFFFF">
                Try Again
              </Text>
            </YStack>
          </Pressable>
        </YStack>
      );
    }
    return this.props.children;
  }
}
