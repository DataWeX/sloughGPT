import React, { useEffect, useState } from 'react';
import { View, Text, FlatList, TouchableOpacity, StyleSheet, SafeAreaView } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../App';

type Nav = NativeStackNavigationProp<RootStackParamList, 'Models'>;

interface Model {
  id: string;
  name: string;
  loaded: boolean;
  size_gb?: number;
}

export function ModelsScreen() {
  const navigation = useNavigation<Nav>();
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const API_URL = process.env.REACT_NATIVE_API_URL || 'http://localhost:8000';
    fetch(`${API_URL}/models`)
      .then(r => r.json())
      .then(data => setModels(data.models || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
          <Text style={styles.backText}>←</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Models</Text>
        <View style={{ width: 40 }} />
      </View>

      {loading ? (
        <View style={styles.center}><Text style={styles.muted}>Loading models...</Text></View>
      ) : models.length === 0 ? (
        <View style={styles.center}><Text style={styles.muted}>No models available</Text></View>
      ) : (
        <FlatList
          data={models}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.list}
          renderItem={({ item }) => (
            <View style={styles.modelRow}>
              <View style={styles.modelInfo}>
                <Text style={styles.modelName}>{item.name}</Text>
                {item.size_gb && <Text style={styles.modelSize}>{item.size_gb.toFixed(2)} GB</Text>}
              </View>
              <View style={[styles.badge, item.loaded ? styles.badgeLoaded : styles.badgeAvailable]}>
                <Text style={[styles.badgeText, item.loaded ? styles.badgeTextLoaded : styles.badgeTextAvailable]}>
                  {item.loaded ? 'Loaded' : 'Available'}
                </Text>
              </View>
            </View>
          )}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 12, paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: '#eee' },
  backButton: { width: 40, height: 40, justifyContent: 'center', alignItems: 'center' },
  backText: { fontSize: 20, color: '#333' },
  headerTitle: { fontSize: 16, fontWeight: '600', color: '#1a1a1a' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  muted: { color: '#999', fontSize: 14 },
  list: { padding: 16 },
  modelRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: '#f0f0f0' },
  modelInfo: { flex: 1 },
  modelName: { fontSize: 15, fontWeight: '500', color: '#1a1a1a', fontFamily: 'monospace' },
  modelSize: { fontSize: 12, color: '#999', marginTop: 2 },
  badge: { paddingHorizontal: 8, paddingVertical: 4, borderRadius: 12 },
  badgeLoaded: { backgroundColor: '#dcfce7' },
  badgeAvailable: { backgroundColor: '#f3f4f6' },
  badgeText: { fontSize: 11, fontWeight: '500' },
  badgeTextLoaded: { color: '#16a34a' },
  badgeTextAvailable: { color: '#666' },
});
