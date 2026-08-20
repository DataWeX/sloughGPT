import React from 'react';
import {Pressable, View, Text as RNText, StyleSheet} from 'react-native';
import {useColors} from '../theme/colors';

interface ErrorFallbackProps {
  error: Error | null;
  onRetry: () => void;
}

function ErrorFallback({error, onRetry}: ErrorFallbackProps) {
  const colors = useColors();
  const s = makeStyles(colors);
  return (
    <View style={s.container}>
      <RNText style={s.icon}>!</RNText>
      <RNText style={s.title}>Something went wrong</RNText>
      <RNText style={s.message}>
        {error?.message || 'An unexpected error occurred'}
      </RNText>
      <Pressable onPress={onRetry} style={s.button} testID="error-retry-button">
        <RNText style={s.buttonText}>Try Again</RNText>
      </Pressable>
    </View>
  );
}

function makeStyles(colors: ReturnType<typeof useColors>) {
  return StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: colors.background,
      alignItems: 'center',
      justifyContent: 'center',
      padding: 32,
    },
    icon: {
      fontSize: 48,
      marginBottom: 16,
      color: colors.error,
      fontWeight: '700',
    },
    title: {
      fontSize: 20,
      fontWeight: '600',
      color: colors.text,
      marginBottom: 8,
    },
    message: {
      fontSize: 13,
      color: colors.textMuted,
      textAlign: 'center',
      lineHeight: 18,
      marginBottom: 24,
    },
    button: {
      backgroundColor: colors.primary,
      paddingHorizontal: 20,
      paddingVertical: 12,
      borderRadius: 12,
    },
    buttonText: {
      fontSize: 15,
      fontWeight: '600',
      color: '#FFFFFF',
    },
  });
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

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    if (__DEV__) console.error('[ErrorBoundary]', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <ErrorBoundaryFallback
          error={this.state.error}
          onRetry={() => this.setState({hasError: false, error: null})}
        />
      );
    }
    return this.props.children;
  }
}

function ErrorBoundaryFallback(props: ErrorFallbackProps) {
  return <ErrorFallback {...props} />;
}
