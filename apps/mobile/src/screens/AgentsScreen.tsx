import React, {useEffect, useState, useCallback} from 'react';
import {FlatList, Pressable, RefreshControl, TextInput as RNTextInput} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text} from 'tamagui';
import {useColors} from '../theme/colors';
import {api} from '../services/api-client';
import {Icon} from '../components/Icon';
import {StatusBadge} from '../components/StatusBadge';
import {triggerHaptic} from '../services/haptics';
import {toast} from '../services/toast';

interface Agent {
  id: string;
  name: string;
  description: string;
  instructions: string;
  tools: string[];
  avatar: string;
}

const AVAILABLE_TOOLS = ['search', 'browse', 'execute_code', 'read_file', 'write_file', 'api_call', 'image_analyze'];

export function AgentsScreen() {
  const colors = useColors();
  const [agents, setAgents] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [showCreate, setShowCreate] = useState(false);
  const [executing, setExecuting] = useState<string | null>(null);
  const [execResult, setExecResult] = useState<{agent: string; output: string} | null>(null);

  // Create form
  const [newName, setNewName] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [newInstructions, setNewInstructions] = useState('');
  const [newTools, setNewTools] = useState<string[]>([]);
  const [creating, setCreating] = useState(false);

  // Execute form
  const [execInput, setExecInput] = useState('');
  const [execTarget, setExecTarget] = useState<string | null>(null);

  const fetchAgents = useCallback(async () => {
    try {
      const data = await api.get<Agent[]>('/agents');
      setAgents(Array.isArray(data) ? data : []);
    } catch {
      setAgents([]);
    }
  }, []);

  useEffect(() => {
    fetchAgents().finally(() => setLoading(false));
  }, [fetchAgents]);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchAgents();
    setRefreshing(false);
  };

  const handleCreate = async () => {
    if (!newName.trim()) return;
    try {
      setCreating(true);
      triggerHaptic('light');
      await api.post('/agents', {
        name: newName.trim(),
        description: newDesc.trim(),
        instructions: newInstructions.trim(),
        tools: newTools,
      });
      triggerHaptic('success');
      toast.success(`Agent "${newName.trim()}" created`);
      setNewName('');
      setNewDesc('');
      setNewInstructions('');
      setNewTools([]);
      setShowCreate(false);
      await fetchAgents();
    } catch {
      toast.error('Failed to create agent');
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      triggerHaptic('light');
      await api.delete(`/agents/${id}`);
      triggerHaptic('success');
      toast.success('Agent deleted');
      await fetchAgents();
    } catch {
      toast.error('Failed to delete agent');
    }
  };

  const handleExecute = async () => {
    if (!execTarget || !execInput.trim()) return;
    try {
      setExecuting(execTarget);
      triggerHaptic('light');
      const result = await api.post<{output: string}>(`/agents/${execTarget}/execute`, {
        request: execInput.trim(),
      });
      setExecResult({agent: execTarget, output: result.output || JSON.stringify(result)});
      setExecInput('');
      setExecTarget(null);
      toast.success('Agent executed');
    } catch {
      toast.error('Execution failed');
    } finally {
      setExecuting(null);
    }
  };

  const toggleTool = (tool: string) => {
    setNewTools(prev => prev.includes(tool) ? prev.filter(t => t !== tool) : [...prev, tool]);
  };

  const renderAgent = ({item}: {item: Agent}) => (
    <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={0.5} borderColor={colors.border} gap={8}>
      <XStack justifyContent="space-between" alignItems="center">
        <YStack flex={1} gap={2}>
          <Text fontSize={15} fontWeight="600" color={colors.text}>{item.name}</Text>
          {item.description ? <Text fontSize={12} color={colors.textMuted} numberOfLines={2}>{item.description}</Text> : null}
        </YStack>
        <Pressable onPress={() => handleDelete(item.id)}>
          <Icon name="trash-2" size={16} color={colors.error} />
        </Pressable>
      </XStack>

      {item.tools.length > 0 && (
        <XStack gap={4} flexWrap="wrap">
          {item.tools.slice(0, 4).map(tool => (
            <StatusBadge key={tool} label={tool} variant="info" />
          ))}
          {item.tools.length > 4 && <StatusBadge label={`+${item.tools.length - 4}`} variant="default" />}
        </XStack>
      )}

      {/* Execute */}
      {execTarget === item.id ? (
        <YStack gap={6}>
          <RNTextInput
            value={execInput}
            onChangeText={setExecInput}
            placeholder="Enter task for this agent..."
            placeholderTextColor={colors.textMuted}
            style={{
              borderWidth: 1,
              borderColor: colors.border,
              borderRadius: 8,
              padding: 8,
              fontSize: 13,
              color: colors.text,
              backgroundColor: colors.background,
            }}
          />
          <XStack gap={6}>
            <Pressable onPress={handleExecute} disabled={!execInput.trim() || executing !== null} style={{flex: 1}}>
              <XStack padding={8} borderRadius={6} backgroundColor={execInput.trim() && executing === null ? colors.primary : colors.border} alignItems="center" justifyContent="center" gap={4}>
                <Icon name="zap" size={14} color="white" />
                <Text fontSize={12} fontWeight="600" color="white">{executing === item.id ? 'Running...' : 'Run'}</Text>
              </XStack>
            </Pressable>
            <Pressable onPress={() => {setExecTarget(null); setExecInput('');}}>
              <XStack padding={8} borderRadius={6} backgroundColor={colors.background} alignItems="center" justifyContent="center">
                <Icon name="x" size={14} color={colors.textMuted} />
              </XStack>
            </Pressable>
          </XStack>
        </YStack>
      ) : (
        <Pressable onPress={() => setExecTarget(item.id)}>
          <XStack padding={8} borderRadius={6} backgroundColor={colors.primaryAlpha(0.1)} alignItems="center" justifyContent="center" gap={4}>
            <Icon name="zap" size={14} color={colors.primary} />
            <Text fontSize={12} fontWeight="500" color={colors.primary}>Execute</Text>
          </XStack>
        </Pressable>
      )}
    </YStack>
  );

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: colors.background}} edges={['top']}>
      <XStack paddingHorizontal={16} paddingVertical={12} alignItems="center" justifyContent="space-between">
        <Text fontSize={20} fontWeight="600" color={colors.text}>Agents</Text>
        <XStack gap={12}>
          <Pressable onPress={() => setShowCreate(!showCreate)}>
            <Icon name={showCreate ? 'x' : 'plus'} size={20} color={colors.primary} />
          </Pressable>
          <Pressable onPress={onRefresh}>
            <Icon name="refresh-cw" size={18} color={colors.primary} />
          </Pressable>
        </XStack>
      </XStack>

      {loading ? (
        <YStack flex={1} alignItems="center" justifyContent="center">
          <StatusBadge label="Loading..." variant="info" />
        </YStack>
      ) : (
        <FlatList
          data={agents}
          keyExtractor={item => item.id}
          renderItem={renderAgent}
          contentContainerStyle={{paddingHorizontal: 16, paddingBottom: 32, gap: 10}}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
          ListHeaderComponent={
            showCreate ? (
              <YStack padding={14} borderRadius={10} backgroundColor={colors.white} borderWidth={1} borderColor={colors.primary} gap={8}>
                <Text fontSize={15} fontWeight="600" color={colors.text}>New Agent</Text>
                <RNTextInput
                  value={newName}
                  onChangeText={setNewName}
                  placeholder="Agent name"
                  placeholderTextColor={colors.textMuted}
                  style={{borderWidth: 1, borderColor: colors.border, borderRadius: 8, padding: 8, fontSize: 13, color: colors.text, backgroundColor: colors.background}}
                />
                <RNTextInput
                  value={newDesc}
                  onChangeText={setNewDesc}
                  placeholder="Description"
                  placeholderTextColor={colors.textMuted}
                  style={{borderWidth: 1, borderColor: colors.border, borderRadius: 8, padding: 8, fontSize: 13, color: colors.text, backgroundColor: colors.background}}
                />
                <RNTextInput
                  value={newInstructions}
                  onChangeText={setNewInstructions}
                  placeholder="Instructions for the agent"
                  placeholderTextColor={colors.textMuted}
                  multiline
                  numberOfLines={3}
                  style={{borderWidth: 1, borderColor: colors.border, borderRadius: 8, padding: 8, fontSize: 13, color: colors.text, backgroundColor: colors.background, minHeight: 72, textAlignVertical: 'top'}}
                />
                <YStack gap={4}>
                  <Text fontSize={12} fontWeight="500" color={colors.textMuted}>Tools</Text>
                  <XStack gap={4} flexWrap="wrap">
                    {AVAILABLE_TOOLS.map(tool => (
                      <Pressable key={tool} onPress={() => toggleTool(tool)}>
                        <XStack paddingHorizontal={8} paddingVertical={4} borderRadius={4} backgroundColor={newTools.includes(tool) ? colors.primary : colors.background} borderWidth={0.5} borderColor={colors.border} gap={4} alignItems="center">
                          <Text fontSize={11} color={newTools.includes(tool) ? 'white' : colors.text}>{tool}</Text>
                        </XStack>
                      </Pressable>
                    ))}
                  </XStack>
                </YStack>
                <Pressable onPress={handleCreate} disabled={!newName.trim() || creating}>
                  <XStack padding={10} borderRadius={8} backgroundColor={newName.trim() && !creating ? colors.primary : colors.border} alignItems="center" justifyContent="center" gap={6}>
                    <Icon name="plus" size={16} color="white" />
                    <Text fontSize={13} fontWeight="600" color="white">{creating ? 'Creating...' : 'Create Agent'}</Text>
                  </XStack>
                </Pressable>
              </YStack>
            ) : agents.length === 0 ? (
              <YStack padding={20} alignItems="center" gap={8}>
                <Icon name="brain" size={24} color={colors.textMuted} />
                <Text fontSize={13} color={colors.textMuted}>No agents yet. Tap + to create one.</Text>
              </YStack>
            ) : (
              <Text fontSize={12} color={colors.textMuted} paddingHorizontal={4}>{agents.length} agent{agents.length !== 1 ? 's' : ''}</Text>
            )
          }
          ListEmptyComponent={null}
        />
      )}

      {/* Execution Result */}
      {execResult && (
        <YStack position="absolute" bottom={0} left={0} right={0} padding={16} backgroundColor={colors.white} borderTopWidth={1} borderTopColor={colors.border} gap={6}>
          <XStack justifyContent="space-between" alignItems="center">
            <Text fontSize={13} fontWeight="600" color={colors.text}>{execResult.agent} result</Text>
            <Pressable onPress={() => setExecResult(null)}>
              <Icon name="x" size={16} color={colors.textMuted} />
            </Pressable>
          </XStack>
          <YStack padding={10} borderRadius={6} backgroundColor={colors.background} maxHeight={150}>
            <Text fontSize={12} fontFamily="monospace" color={colors.text} numberOfLines={8}>{execResult.output}</Text>
          </YStack>
        </YStack>
      )}
    </SafeAreaView>
  );
}
