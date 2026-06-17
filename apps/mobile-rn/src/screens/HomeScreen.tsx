import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, SafeAreaView } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../App';

type Nav = NativeStackNavigationProp<RootStackParamList, 'Home'>;

export function HomeScreen() {
  const navigation = useNavigation<Nav>();

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.title}>SloughGPT</Text>
        <Text style={styles.subtitle}>AI Platform</Text>

        <View style={styles.buttons}>
          <TouchableOpacity style={styles.button} onPress={() => navigation.navigate('Chat')}>
            <Text style={styles.buttonText}>Start Chatting</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.buttonSecondary} onPress={() => navigation.navigate('Models')}>
            <Text style={styles.buttonSecondaryText}>Models</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.buttonSecondary} onPress={() => navigation.navigate('Settings')}>
            <Text style={styles.buttonSecondaryText}>Settings</Text>
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  content: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24 },
  title: { fontSize: 32, fontWeight: 'bold', color: '#1a1a1a' },
  subtitle: { fontSize: 16, color: '#666', marginTop: 4, marginBottom: 48 },
  buttons: { width: '100%', maxWidth: 320, gap: 12 },
  button: { backgroundColor: '#6366f1', paddingVertical: 14, borderRadius: 12, alignItems: 'center' },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  buttonSecondary: { borderWidth: 1, borderColor: '#e5e5e5', paddingVertical: 14, borderRadius: 12, alignItems: 'center' },
  buttonSecondaryText: { color: '#333', fontSize: 16 },
});
