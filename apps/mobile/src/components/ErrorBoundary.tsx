import React from 'react';
import {Pressable, View, Text as RNText, StyleSheet} from 'react-native';

interface ErrorFallbackProps {
  error: Error | null;
  onRetry: () => void;
}

function ErrorFallback({error, onRetry}: ErrorFallbackProps) {
  const s = makeStyles();
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

function makeStyles() {
  return StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: '#110F18',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 32,
    },
    icon: {
      fontSize: 48,
      marginBottom: 16,
      color: '#EF4444',
      fontWeight: '700',
    },
    title: {
      fontSize: 20,
      fontWeight: '600',
      color: '#F0ECF5',
      marginBottom: 8,
    },
    message: {
      fontSize: 13,
      color: '#9B95A8',
      textAlign: 'center',
      lineHeight: 18,
      marginBottom: 24,
    },
    button: {
      backgroundColor: '#7C52C4',
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
