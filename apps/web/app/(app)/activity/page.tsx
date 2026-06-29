'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button, Input, ProgressBar, Spinner, Chip } from '@/components/ui'
import { StatCard, KpiGrid } from '@/components/ui/display'
import { useToastStore } from '@/lib/toast-store'
import { activityController, type ActivityStatus, type DatasetRecord, type PredictResponse } from '@/lib/activity-controller'

const ACTIVITIES = ['stationary', 'walking', 'running', 'shaking', 'driving', 'cycling']
const WINDOW_SIZE = 128
const ACTIVITY_COLORS = ['bg-gray-400', 'bg-green-400', 'bg-red-400', 'bg-yellow-400', 'bg-blue-400', 'bg-purple-400']

function generateSimulatedSensorData(label: number): number[][] {
  const t = Array.from({ length: WINDOW_SIZE }, (_, i) => i / 64)
  const data: number[][] = []
  for (let i = 0; i < WINDOW_SIZE; i++) {
    const ti = t[i]
    let ax = 0, ay = 0, az = 9.81, gx = 0, gy = 0, gz = 0
    const n = () => (Math.random() - 0.5) * 2
    switch (label) {
      case 0: // stationary
        az = 9.81 + n() * 0.05
        ax = n() * 0.05; ay = n() * 0.05
        gx = n() * 0.02; gy = n() * 0.02; gz = n() * 0.02
        break
      case 1: // walking ~2hz
        az = 9.81 + 2 * Math.sin(2 * Math.PI * 2 * ti) + n() * 0.1
        ax = 0.5 * Math.sin(2 * Math.PI * 1 * ti + 0.5) + n() * 0.1
        gx = 0.3 * Math.sin(2 * Math.PI * 2 * ti) + n() * 0.1
        break
      case 2: // running ~3hz
        az = 9.81 + 5 * Math.sin(2 * Math.PI * 3 * ti) + n() * 0.2
        ax = 1.5 * Math.sin(2 * Math.PI * 1.5 * ti + 0.8) + n() * 0.2
        ay = 0.8 * Math.sin(2 * Math.PI * 3 * ti + 0.3) + n() * 0.2
        gx = 0.8 * Math.sin(2 * Math.PI * 3 * ti) + n() * 0.2
        gy = 0.5 * Math.sin(2 * Math.PI * 1.5 * ti) + n() * 0.2
        break
      case 3: // shaking
        ax = n() * 4; ay = n() * 4; az = 9.81 + n() * 4
        gx = n() * 3; gy = n() * 3; gz = n() * 3
        break
      case 4: // driving
        az = 9.81 + 0.3 * Math.sin(2 * Math.PI * 0.5 * ti) + n() * 0.08
        ax = 0.2 * Math.sin(2 * Math.PI * 0.8 * ti) + n() * 0.08
        ay = 0.1 * Math.sin(2 * Math.PI * 0.6 * ti + 0.4) + n() * 0.08
        gx = n() * 0.05; gy = n() * 0.05; gz = n() * 0.05
        break
      case 5: // cycling
        az = 9.81 + 1 * Math.sin(2 * Math.PI * 1.5 * ti) + n() * 0.15
        ax = 2 * Math.sin(2 * Math.PI * 0.75 * ti + 0.2) + n() * 0.15
        gx = 1 * Math.sin(2 * Math.PI * 1.5 * ti) + n() * 0.15
        gy = 0.4 * Math.sin(2 * Math.PI * 0.75 * ti) + n() * 0.15
        break
    }
    data.push([ax, ay, az, gx, gy, gz])
  }
  return data
}

export default function ActivityPage() {
  const addToast = useToastStore(s => s.addToast)

  const [status, setStatus] = useState<ActivityStatus | null>(null)
  const [dataset, setDataset] = useState<DatasetRecord[]>([])
  const [loading, setLoading] = useState(true)

  // Recording
  const [recording, setRecording] = useState(false)
  const [selectedLabel, setSelectedLabel] = useState<number>(1)
  const [recordBuffer, setRecordBuffer] = useState<number[][]>([])

  // Training
  const [training, setTraining] = useState(false)
  const [trainEpochs, setTrainEpochs] = useState(30)
  const [trainResult, setTrainResult] = useState<{ acc: number; samples: number; message: string } | null>(null)

  // Prediction
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null)
  const [predictLoading, setPredictLoading] = useState(false)

  // Real sensor
  const [sensorSupported, setSensorSupported] = useState(false)
  const [sensorPermitted, setSensorPermitted] = useState(false)
  const [liveAxes, setLiveAxes] = useState<number[]>([0, 0, 0, 0, 0, 0])
  const sensorRef = useRef<number[][]>([])
  const animRef = useRef<number>(0)

  const fetchStatus = useCallback(async () => {
    try {
      const [s, d] = await Promise.all([
        activityController.status(),
        activityController.dataset(),
      ])
      setStatus(s)
      setDataset(d.recordings)
    } catch { /* server may be down */ }
  }, [])

  useEffect(() => {
    fetchStatus().finally(() => setLoading(false))
  }, [fetchStatus])

  // Check DeviceMotion API support
  useEffect(() => {
    if (typeof window !== 'undefined' && 'DeviceMotionEvent' in window) {
      setSensorSupported(true)
    }
  }, [])

  const requestSensorPermission = async () => {
    if (typeof (DeviceMotionEvent as any).requestPermission === 'function') {
      try {
        const perm = await (DeviceMotionEvent as any).requestPermission()
        if (perm !== 'granted') {
          addToast('Sensor permission denied — use simulation instead', 'error')
          return
        }
      } catch { /* iOS fallback */ }
    }
    setSensorPermitted(true)
    recording
  }

  const startSensorRecording = () => {
    sensorRef.current = []
    setRecordBuffer([])
    const handler = (e: DeviceMotionEvent) => {
      const ax = e.accelerationIncludingGravity?.x ?? 0
      const ay = e.accelerationIncludingGravity?.y ?? 0
      const az = e.accelerationIncludingGravity?.z ?? 0
      const gx = e.rotationRate?.alpha ?? 0
      const gy = e.rotationRate?.beta ?? 0
      const gz = e.rotationRate?.gamma ?? 0
      sensorRef.current.push([ax, ay, az, gx, gy, gz])
      setLiveAxes([ax, ay, az, gx, gy, gz])

      if (sensorRef.current.length >= WINDOW_SIZE) {
        window.removeEventListener('devicemotion', handler)
        cancelAnimationFrame(animRef.current)
        const buf = sensorRef.current.slice(0, WINDOW_SIZE)
        setRecordBuffer(buf)
        setRecording(false)
        uploadRecording(buf, selectedLabel)
      }
    }

    window.addEventListener('devicemotion', handler)
    recording
  }

  const startSimulatedRecording = (label: number) => {
    const buf = generateSimulatedSensorData(label)
    setRecordBuffer(buf)
    setRecording(false)
    uploadRecording(buf, label)
  }

  const uploadRecording = async (buf: number[][], label: number) => {
    try {
      const res = await activityController.recordData({ data: buf, label })
      addToast(`Saved recording #${res.id} as "${ACTIVITIES[label]}"`, 'success')
      await fetchStatus()
    } catch (e: any) {
      addToast(`Failed to save: ${e.message}`, 'error')
    }
  }

  const handleTrain = async () => {
    setTraining(true)
    setTrainResult(null)
    try {
      const res = await activityController.train({ epochs: trainEpochs, lr: 0.001, batch_size: 16 })
      setTrainResult({
        acc: res.val_accuracy ?? 0,
        samples: res.num_samples,
        message: res.message,
      })
      addToast(`Training done — ${(res.val_accuracy! * 100).toFixed(0)}% accuracy`, 'success')
      await fetchStatus()
    } catch (e: any) {
      addToast(`Training failed: ${e.message}`, 'error')
    }
    setTraining(false)
  }

  const handlePredict = async () => {
    if (recordBuffer.length === 0) {
      addToast('Record or simulate data first', 'error')
      return
    }
    setPredictLoading(true)
    try {
      const res = await activityController.predict({ data: recordBuffer })
      setPredictResult(res)
    } catch (e: any) {
      addToast(`Prediction failed: ${e.message}`, 'error')
    }
    setPredictLoading(false)
  }

  const handleDeleteAll = async () => {
    try {
      const res = await activityController.deleteAll()
      addToast(`Deleted ${res.deleted} recordings`, 'success')
      setPredictResult(null)
      setTrainResult(null)
      setRecordBuffer([])
      await fetchStatus()
    } catch (e: any) {
      addToast(`Delete failed: ${e.message}`, 'error')
    }
  }

  const sensorAvailable = sensorSupported && sensorPermitted

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Activity Recognition" />} />

      <div className="space-y-4">
        {/* Status */}
        <Card>
          <CardHeader><CardTitle className="text-base">System Status</CardTitle></CardHeader>
          <CardContent>
            {loading ? (
              <Spinner className="h-4 w-4" />
            ) : status ? (
              <KpiGrid columns={4}>
                <StatCard label="Recordings" value={status.num_recordings} />
                <StatCard label="Labeled" value={status.num_labels} />
                <StatCard label="Model" value={status.model_loaded ? 'Trained' : 'Not trained'} />
                <StatCard label="Activities" value={`${status.activities.length} classes`} />
              </KpiGrid>
            ) : <p className="text-sm text-muted-foreground">Server offline</p>}
          </CardContent>
        </Card>

        {/* Data Collection Card */}
        <Card>
          <CardHeader><CardTitle className="text-base">Collect Sensor Data</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-wrap gap-1.5">
              {ACTIVITIES.map((name, i) => (
                <Chip
                  key={name}
                  label={name}
                  selected={selectedLabel === i}
                  onClick={() => setSelectedLabel(i)}
                />
              ))}
            </div>

            <div className="flex flex-wrap gap-2">
              {sensorSupported && (
                <Button
                  size="sm"
                  onClick={sensorPermitted ? startSensorRecording : requestSensorPermission}
                  disabled={recording}
                >
                  {sensorPermitted ? 'Record from device' : 'Allow sensors'}
                </Button>
              )}

              <Button
                size="sm"
                variant="secondary"
                onClick={() => {
                  setRecording(true)
                  startSimulatedRecording(selectedLabel)
                }}
                disabled={recording}
              >
                Simulate sample
              </Button>

              {recordBuffer.length > 0 && (
                <span className="text-xs text-muted-foreground self-center">
                  Buffer: {recordBuffer.length} samples
                </span>
              )}
            </div>

            {/* Live sensor preview */}
            {sensorAvailable && (
              <div className="grid grid-cols-6 gap-1 text-[10px] font-mono text-muted-foreground">
                {['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'].map((name, i) => (
                  <div key={name} className="truncate">{name}: {liveAxes[i].toFixed(2)}</div>
                ))}
              </div>
            )}

            {/* Quick-simulate one of each */}
            <div>
              <p className="text-[10px] text-muted-foreground mb-1.5">Quick-fill dataset (one per class):</p>
              <div className="flex flex-wrap gap-1.5">
                {ACTIVITIES.map((name, i) => (
                  <Button
                    key={name}
                    size="sm"
                    variant="ghost"
                    disabled={recording}
                    onClick={() => {
                      setRecording(true)
                      startSimulatedRecording(i)
                    }}
                  >
                    +{name}
                  </Button>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Training Card */}
        <Card>
          <CardHeader><CardTitle className="text-base">Train Classifier</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center gap-3">
              <label className="text-xs text-muted-foreground">Epochs:</label>
              <Input
                type="number"
                value={trainEpochs}
                onChange={e => setTrainEpochs(Math.max(1, parseInt(e.target.value) || 30))}
                className="w-20 text-sm"
              />
              <Button size="sm" onClick={handleTrain} disabled={training || (status?.num_labels ?? 0) < 3}>
                {training ? 'Training...' : 'Train'}
              </Button>
              <Button size="sm" variant="destructive" onClick={handleDeleteAll}>
                Delete all data
              </Button>
            </div>
            {training && <ProgressBar value={0} />}
            {trainResult && (
              <div className="text-sm text-muted-foreground space-y-0.5">
                <p>Validation accuracy: <strong>{(trainResult.acc * 100).toFixed(1)}%</strong></p>
                <p>Samples: {trainResult.samples}</p>
                <p className="text-xs">{trainResult.message}</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Prediction Card */}
        <Card>
          <CardHeader><CardTitle className="text-base">Test Prediction</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <Button
              size="sm"
              onClick={handlePredict}
              disabled={predictLoading || !status?.model_loaded || recordBuffer.length === 0}
            >
              {predictLoading ? 'Predicting...' : 'Predict from buffered data'}
            </Button>

            {predictResult && (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium">Prediction:</span>
                  <span className={`px-2 py-0.5 rounded-full text-xs text-white ${ACTIVITY_COLORS[predictResult.class_id]}`}>
                    {predictResult.activity}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {(predictResult.confidence * 100).toFixed(0)}% confidence
                  </span>
                </div>
                <div className="space-y-1">
                  {predictResult.probabilities.map((p, i) => (
                    <div key={i} className="flex items-center gap-2 text-[11px]">
                      <span className="w-20 truncate text-muted-foreground">{ACTIVITIES[i]}</span>
                      <div className="flex-1 h-3 rounded-full bg-muted overflow-hidden">
                        <div
                          className={`h-full ${ACTIVITY_COLORS[i]} transition-all`}
                          style={{ width: `${p * 100}%` }}
                        />
                      </div>
                      <span className="w-10 text-right font-mono text-muted-foreground">
                        {(p * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Dataset Card */}
        <Card>
          <CardHeader><CardTitle className="text-base">Recorded Dataset ({dataset.length})</CardTitle></CardHeader>
          <CardContent>
            {dataset.length === 0 ? (
              <p className="text-sm text-muted-foreground">No recordings yet. Collect some sensor data above.</p>
            ) : (
              <div className="max-h-48 overflow-y-auto space-y-0.5">
                {dataset.map(r => (
                  <div key={r.id} className="flex items-center gap-2 text-xs">
                    <span className="text-muted-foreground w-8">#{r.id}</span>
                    <span className={`px-1.5 py-0.5 rounded text-[10px] text-white ${ACTIVITY_COLORS[r.label < 0 ? 0 : r.label]}`}>
                      {r.activity}
                    </span>
                    <span className="text-muted-foreground">{r.samples} samples</span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
