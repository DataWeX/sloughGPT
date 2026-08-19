import React from 'react';
import {Pressable, View, Text as RNText, StyleSheet} from 'react-native';

interface ErrorFallbackProps {
  error: Error | null;
  onRetry: () => void;
}

function ErrorFallback({error, onRetry}: ErrorFallbackProps) {
  return (
    <View style={styles.container}>
      <RNText style={styles.icon}>!</RNText>
      <RNText style={styles.title}>Something went wrong</RNText>
      <RNText style={styles.message}>
        {error?.message}
      </RNText>
      <Pressable onPress={onRetry} style={styles.button}>
        <RNText style={styles.buttonText}>Try Again</RNText>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
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
    color: '#E85D04',
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
    color: '#968CAC',
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
