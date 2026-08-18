import React from 'react';
import {Pressable} from 'react-native';
import {YStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';

interface ErrorFallbackProps {
  error: Error | null;
  onRetry: () => void;
}

function ErrorFallback({error, onRetry}: ErrorFallbackProps) {
  const c = useColors();
  return (
    <YStack flex={1} backgroundColor={c.background} alignItems="center" justifyContent="center" padding={32}>
      <Text fontSize={48} marginBottom={16} color={c.warningDark} fontWeight="700">!</Text>
      <Text fontSize={20} fontWeight="600" color={c.text} marginBottom={8}>
        Something went wrong
      </Text>
      <Text fontSize={13} color={c.textMuted} textAlign="center" lineHeight={18} marginBottom={24}>
        {error?.message}
      </Text>
      <Pressable onPress={onRetry}>
        <YStack backgroundColor={c.primary} paddingHorizontal={20} paddingVertical={12} borderRadius={12}>
          <Text fontSize={15} fontWeight="600" color={c.textOnPrimary}>
            Try Again
          </Text>
        </YStack>
      </Pressable>
    </YStack>
  );
}

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
        <ErrorFallback
          error={this.state.error}
          onRetry={() => this.setState({hasError: false, error: null})}
        />
      );
    }
    return this.props.children;
  }
}
