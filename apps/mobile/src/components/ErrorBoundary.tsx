import React from 'react';
import {Pressable, View, Text as RNText, StyleSheet, ScrollView} from 'react-native';
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
      <ScrollView contentContainerStyle={s.scrollContent}>
        <RNText style={s.icon}>!</RNText>
        <RNText style={s.title}>Something went wrong</RNText>
        <RNText style={s.message}>
          {error?.message || 'An unexpected error occurred'}
        </RNText>
        {error?.stack && (
          <View style={s.stackContainer}>
            <ScrollView horizontal={false} style={s.stackScroll}>
              <RNText style={s.stackTrace} selectable>
                {error.stack}
              </RNText>
            </ScrollView>
          </View>
        )}
        <Pressable onPress={onRetry} style={s.button} testID="error-retry-button">
          <RNText style={s.buttonText}>Try Again</RNText>
        </Pressable>
      </ScrollView>
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
    scrollContent: {
      alignItems: 'center',
      justifyContent: 'center',
    },
    icon: {
      fontSize: 48,
      marginBottom: 16,
      color: colors.error || '#EF4444',
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
      color: colors.textSecondary,
      textAlign: 'center',
      lineHeight: 18,
      marginBottom: 16,
    },
    stackContainer: {
      backgroundColor: colors.surface || colors.background,
      borderWidth: 1,
      borderColor: colors.border || 'rgba(255,255,255,0.1)',
      borderRadius: 8,
      padding: 12,
      marginBottom: 24,
      maxHeight: 200,
      width: '100%',
    },
    stackScroll: {
      maxHeight: 180,
    },
    stackTrace: {
      fontSize: 10,
      fontFamily: 'monospace',
      color: colors.textSecondary,
      lineHeight: 14,
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
    console.error('[ErrorBoundary]', error, errorInfo);
    try {
      const {api} = require('../services/api-client');
      api.post('/mobile/errors', {
        level: 'error',
        message: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
      }).catch(() => {});
    } catch {}
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
