import { useState, useEffect } from 'react'
import { ScrollView, RefreshControl } from 'react-native'
import {
  YStack,
  XStack,
  Text,
  Card,
  Button,
  Paragraph,
  Progress,
} from 'tamagui'
import {
  Activity,
  Cpu,
  HardDrive,
  Clock,
  Zap,
  ArrowLeft,
} from '@tamagui/lucide-icons'
import { useRouter } from 'expo-router'
import { apiGet } from '@/lib/api-client'

interface DetailedHealth {
  status: string
  model_loaded: boolean
  model_type?: string
  uptime_seconds?: number
  system?: {
    cpu_percent: number
    memory_percent: number
    memory_available_mb: number
  }
  inference?: {
    inference_count: number
    is_inferencing: boolean
  }
  gpu?: {
    backend: string
    device_type: string
    vram_gb: number
    tier: string
  }
}

interface SystemMetrics {
  cpu_percent: number
  memory_percent: number
  disk_used_bytes: number
  disk_free_bytes: number
  disk_total_bytes: number
}

function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400)
  const h = Math.floor((seconds % 86400) / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (d > 0) return `${d}d ${h}h`
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

function formatBytes(bytes: number): string {
  const gb = bytes / (1024 ** 3)
  return `${gb.toFixed(1)} GB`
}

export default function HealthScreen() {
  const router = useRouter()
  const [health, setHealth] = useState<DetailedHealth | null>(null)
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null)
  const [refreshing, setRefreshing] = useState(false)

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 5000)
    return () => clearInterval(interval)
  }, [])

  async function fetchData() {
    try {
      const [h, m] = await Promise.all([
        apiGet<DetailedHealth>('/health/detailed').catch(() => null),
        apiGet<SystemMetrics>('/system/metrics').catch(() => null),
      ])
      if (h) setHealth(h)
      if (m) setMetrics(m)
    } catch {
      // silent
    } finally {
      setRefreshing(false)
    }
  }

  const cpuPercent = health?.system?.cpu_percent ?? metrics?.cpu_percent ?? 0
  const memPercent = health?.system?.memory_percent ?? metrics?.memory_percent ?? 0
  const memAvailable = health?.system?.memory_available_mb
    ? (health.system.memory_available_mb / 1024).toFixed(1)
    : '—'

  return (
    <YStack flex={1} backgroundColor="$background">
      <XStack
        paddingHorizontal="$3"
        paddingVertical="$2"
        alignItems="center"
        gap="$2"
        borderBottomWidth={1}
        borderBottomColor="$borderColor"
        paddingTop={56}
      >
        <Button
          size="$3"
          chromeless
          icon={<ArrowLeft size={20} />}
          onPress={() => router.back()}
        />
        <Text fontSize="$6" fontWeight="700" color="$color">
          System Health
        </Text>
      </XStack>

      <ScrollView
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={() => {
              setRefreshing(true)
              fetchData()
            }}
          />
        }
        contentContainerStyle={{ padding: 12 }}
      >
        <YStack gap="$3">
          {/* Status KPIs */}
          <XStack gap="$2" flexWrap="wrap">
            {[
              {
                label: 'API',
                value: health?.status || '...',
                color: health?.status === 'healthy' ? '$success' : '$destructive',
                icon: <Activity size={16} />,
              },
              {
                label: 'Model',
                value: health?.model_loaded ? 'Loaded' : 'None',
                color: health?.model_loaded ? '$success' : '$placeholderColor',
                icon: <Cpu size={16} />,
              },
              {
                label: 'Uptime',
                value: health?.uptime_seconds
                  ? formatUptime(health.uptime_seconds)
                  : '...',
                color: '$primary',
                icon: <Clock size={16} />,
              },
              {
                label: 'Inferences',
                value: String(health?.inference?.inference_count ?? 0),
                color: '$accent',
                icon: <Zap size={16} />,
              },
            ].map((kpi) => (
              <Card
                key={kpi.label}
                flex={1}
                minWidth="45%"
                backgroundColor="$backgroundStrong"
                borderRadius="$4"
                padding="$3"
              >
                <XStack alignItems="center" gap="$1" marginBottom="$1">
                  <Text color={kpi.color}>{kpi.icon}</Text>
                  <Text color="$placeholderColor" fontSize="$1">
                    {kpi.label}
                  </Text>
                </XStack>
                <Text color={kpi.color} fontSize="$5" fontWeight="700">
                  {kpi.value}
                </Text>
              </Card>
            ))}
          </XStack>

          {/* CPU */}
          <Card elevation={2} borderRadius="$5" backgroundColor="$backgroundStrong" padding="$4">
            <XStack justifyContent="space-between" alignItems="center" marginBottom="$2">
              <XStack alignItems="center" gap="$2">
                <Cpu size={18} color="$primary" />
                <Text fontSize="$4" fontWeight="600" color="$color">
                  CPU
                </Text>
              </XStack>
              <Text color="$primary" fontSize="$5" fontWeight="700">
                {cpuPercent.toFixed(0)}%
              </Text>
            </XStack>
            <Progress value={cpuPercent} backgroundColor="$background">
              <Progress.Indicator backgroundColor="$primary" />
            </Progress>
          </Card>

          {/* Memory */}
          <Card elevation={2} borderRadius="$5" backgroundColor="$backgroundStrong" padding="$4">
            <XStack justifyContent="space-between" alignItems="center" marginBottom="$2">
              <XStack alignItems="center" gap="$2">
                <HardDrive size={18} color="$accent" />
                <Text fontSize="$4" fontWeight="600" color="$color">
                  Memory
                </Text>
              </XStack>
              <Text color="$accent" fontSize="$5" fontWeight="700">
                {memPercent.toFixed(0)}%
              </Text>
            </XStack>
            <Progress value={memPercent} backgroundColor="$background">
              <Progress.Indicator backgroundColor="$accent" />
            </Progress>
            <Paragraph color="$placeholderColor" fontSize="$2" marginTop="$2">
              {memAvailable} GB available
            </Paragraph>
          </Card>

          {/* Disk */}
          {metrics && (
            <Card elevation={2} borderRadius="$5" backgroundColor="$backgroundStrong" padding="$4">
              <XStack alignItems="center" gap="$2" marginBottom="$2">
                <HardDrive size={18} color="$success" />
                <Text fontSize="$4" fontWeight="600" color="$color">
                  Disk
                </Text>
              </XStack>
              <YStack gap="$1">
                <XStack justifyContent="space-between">
                  <Text color="$placeholderColor" fontSize="$2">Used</Text>
                  <Text color="$color" fontSize="$2">{formatBytes(metrics.disk_used_bytes)}</Text>
                </XStack>
                <XStack justifyContent="space-between">
                  <Text color="$placeholderColor" fontSize="$2">Free</Text>
                  <Text color="$color" fontSize="$2">{formatBytes(metrics.disk_free_bytes)}</Text>
                </XStack>
                <XStack justifyContent="space-between">
                  <Text color="$placeholderColor" fontSize="$2">Total</Text>
                  <Text color="$color" fontSize="$2">{formatBytes(metrics.disk_total_bytes)}</Text>
                </XStack>
              </YStack>
              <Progress
                value={metrics.disk_total_bytes > 0 ? (metrics.disk_used_bytes / metrics.disk_total_bytes) * 100 : 0}
                backgroundColor="$background"
                marginTop="$2"
              >
                <Progress.Indicator backgroundColor="$success" />
              </Progress>
            </Card>
          )}

          {/* GPU */}
          {health?.gpu && (
            <Card elevation={2} borderRadius="$5" backgroundColor="$backgroundStrong" padding="$4">
              <XStack alignItems="center" gap="$2" marginBottom="$2">
                <Zap size={18} color="$warning" />
                <Text fontSize="$4" fontWeight="600" color="$color">
                  GPU
                </Text>
              </XStack>
              <YStack gap="$1">
                <XStack justifyContent="space-between">
                  <Text color="$placeholderColor" fontSize="$2">Backend</Text>
                  <Text color="$color" fontSize="$2">{health.gpu.backend}</Text>
                </XStack>
                <XStack justifyContent="space-between">
                  <Text color="$placeholderColor" fontSize="$2">Device</Text>
                  <Text color="$color" fontSize="$2">{health.gpu.device_type}</Text>
                </XStack>
                <XStack justifyContent="space-between">
                  <Text color="$placeholderColor" fontSize="$2">VRAM</Text>
                  <Text color="$color" fontSize="$2">{health.gpu.vram_gb} GB</Text>
                </XStack>
                <XStack justifyContent="space-between">
                  <Text color="$placeholderColor" fontSize="$2">Tier</Text>
                  <Text color="$color" fontSize="$2">{health.gpu.tier}</Text>
                </XStack>
              </YStack>
            </Card>
          )}

          {/* Model Info */}
          {health?.model_type && (
            <Card elevation={2} borderRadius="$5" backgroundColor="$backgroundStrong" padding="$4">
              <Text fontSize="$4" fontWeight="600" color="$color" marginBottom="$2">
                Active Model
              </Text>
              <Text color="$primary" fontSize="$3" fontWeight="500">
                {health.model_type}
              </Text>
            </Card>
          )}
        </YStack>
      </ScrollView>
    </YStack>
  )
}
