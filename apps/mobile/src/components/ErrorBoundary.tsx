import React from 'react';
import {YStack, Text, Button} from 'tamagui';
import {Icon} from './Icon';

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
        <YStack flex={1} backgroundColor="$background" alignItems="center" justifyContent="center" padding={32}>
          <YStack marginBottom={16} alignItems="center">
            <Icon name="triangle-alert" size={48} color="#E8A83C" />
          </YStack>
          <Text fontSize={20} fontWeight="600" color="$color" marginBottom={8}>
            Something went wrong
          </Text>
          <Text fontSize={13} fontWeight="400" color="$color10" textAlign="center" lineHeight={18} marginBottom={24}>
            {this.state.error?.message}
          </Text>
          <Button
            onPress={() => this.setState({hasError: false, error: null})}
            backgroundColor="$color9"
            paddingHorizontal={20}
            paddingVertical={12}
            borderRadius={8}>
            <Text fontSize={15} fontWeight="600" color="white">
              Try Again
            </Text>
          </Button>
        </YStack>
      );
    }
    return this.props.children;
  }
}
