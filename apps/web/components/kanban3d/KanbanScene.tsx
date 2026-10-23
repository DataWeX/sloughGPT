/* KanbanScene — immersive 3D board projection (three + react-three-fiber).
   Same board model as the DOM view: columns become lanes, cards become
   instanced tiles. Selection flows back to the page's shared action bar,
   so engine-validated transitions (incl. force) work identically here.
   Lazy-loaded — DOM mode never pays for three.js. */

import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { Board, Card } from '@/components/planner/types'

const MAX_INSTANCES = 2048
const MAX_PER_LANE = 120
const GRID_COLS = 6
const CARD_W = 0.66
const CARD_D = 0.44
const CARD_H = 0.05
const DX = 0.78
const DZ = 0.58
const LANE_W = 5.6
const LANE_GAP = 1.4

// Lane palette cycles by column order — works for any vocabulary.
const LANE_TOKENS = ['--info', '--primary', '--warning', '--success', '--accent', '--chart-1']

function tokenRGB(name: string): [number, number, number] {
  const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  const parts = raw.split(/\s+/).map(Number)
  if (parts.length >= 3 && parts.every((n) => !Number.isNaN(n))) {
    return [parts[0] / 255, parts[1] / 255, parts[2] / 255]
  }
  return [0.5, 0.5, 0.5]
}

interface Placement {
  x: number
  y: number
  z: number
  cardId: string
  color: THREE.Color
}

function makeLabel(text: string, sub: string, rgb: [number, number, number]): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = 512
  canvas.height = 144
  const ctx = canvas.getContext('2d')!
  const toCss = (c: [number, number, number], a = 1) =>
    `rgba(${Math.round(c[0] * 255)}, ${Math.round(c[1] * 255)}, ${Math.round(c[2] * 255)}, ${a})`
  ctx.clearRect(0, 0, 512, 144)
  ctx.textAlign = 'center'
  ctx.fillStyle = toCss(rgb)
  ctx.font = '600 56px system-ui, sans-serif'
  ctx.fillText(text, 256, 62)
  ctx.font = '400 40px system-ui, sans-serif'
  ctx.fillStyle = toCss(rgb, 0.7)
  ctx.fillText(sub, 256, 120)
  const tex = new THREE.CanvasTexture(canvas)
  tex.anisotropy = 4
  return tex
}

function LaneLabels({
  lanes,
}: {
  lanes: { name: string; sub: string; x: number; rgb: [number, number, number] }[]
}) {
  const textures = useMemo(
    () => lanes.map((l) => makeLabel(l.name.toUpperCase(), l.sub, l.rgb)),
    [lanes],
  )
  useEffect(() => () => textures.forEach((t) => t.dispose()), [textures])
  return (
    <group>
      {lanes.map((l, i) => (
        <mesh key={l.name} position={[l.x, 0.02, -1.6]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[4.4, 1.24]} />
          <meshBasicMaterial map={textures[i]} transparent depthWrite={false} />
        </mesh>
      ))}
    </group>
  )
}

function Floor({ lanes }: { lanes: { x: number; rgb: [number, number, number] }[] }) {
  return (
    <group>
      {lanes.map((l, i) => (
        <mesh key={i} position={[l.x, -0.03, 4]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
          <planeGeometry args={[LANE_W, 15]} />
          <meshStandardMaterial
            color={
              new THREE.Color(
                l.rgb[0] * 0.25 + 0.06,
                l.rgb[1] * 0.25 + 0.06,
                l.rgb[2] * 0.25 + 0.06,
              )
            }
            roughness={0.95}
          />
        </mesh>
      ))}
    </group>
  )
}

function Cards({
  placements,
  selectedId,
  onSelect,
}: {
  placements: Placement[]
  selectedId: string | null
  onSelect: (id: string) => void
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])

  useLayoutEffect(() => {
    const mesh = meshRef.current
    if (!mesh) return
    const selected = selectedId
    placements.forEach((p, i) => {
      const isSelected = p.cardId === selected
      dummy.position.set(p.x, isSelected ? p.y + 0.22 : p.y, p.z)
      dummy.rotation.set(0, 0, 0)
      dummy.scale.setScalar(isSelected ? 1.12 : 1)
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)
      const c = isSelected ? p.color.clone().lerp(new THREE.Color(1, 1, 1), 0.55) : p.color
      mesh.setColorAt(i, c)
    })
    mesh.count = placements.length
    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
    mesh.computeBoundingSphere()
  }, [placements, selectedId, dummy])

  return (
    <instancedMesh
      ref={meshRef}
      args={[
        undefined as unknown as THREE.BufferGeometry,
        undefined as unknown as THREE.Material,
        MAX_INSTANCES,
      ]}
      onClick={(e) => {
        e.stopPropagation()
        if (e.instanceId != null && placements[e.instanceId]) {
          onSelect(placements[e.instanceId].cardId)
        }
      }}
      onPointerOver={() => {
        document.body.style.cursor = 'pointer'
      }}
      onPointerOut={() => {
        document.body.style.cursor = ''
      }}
    >
      <boxGeometry args={[CARD_W, CARD_H, CARD_D]} />
      <meshStandardMaterial roughness={0.6} metalness={0.05} />
    </instancedMesh>
  )
}

function Controls() {
  const { camera, gl } = useThree()
  useEffect(() => {
    const controls = new OrbitControls(camera, gl.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.maxPolarAngle = Math.PI / 2.05
    controls.target.set(0, 0, 3)
    return () => controls.dispose()
  }, [camera, gl])
  return null
}

export interface KanbanSceneProps {
  board: Board
  selectedCardId: string | null
  onSelect: (id: string | null) => void
}

export default function KanbanScene({ board, selectedCardId, onSelect }: KanbanSceneProps) {
  const sorted = useMemo(
    () => board.columns.slice().sort((a, b) => a.order - b.order),
    [board.columns],
  )

  const laneMeta = useMemo(() => {
    const byCol = new Map<string, Card[]>()
    for (const col of sorted) byCol.set(col.name, [])
    for (const card of board.cards) byCol.get(card.column)?.push(card)
    return sorted.map((col, i) => {
      const all = byCol.get(col.name) ?? []
      const visible = all.slice(0, MAX_PER_LANE)
      const rgb = tokenRGB(LANE_TOKENS[i % LANE_TOKENS.length])
      const wip = col.wip_limit > 0 ? `${all.length}/${col.wip_limit}` : `${all.length}`
      const hidden = all.length - visible.length
      const sub = hidden > 0 ? `${wip}  (+${hidden} hidden)` : wip
      return { col, all, visible, rgb, sub, x: (i - (sorted.length - 1) / 2) * (LANE_W + LANE_GAP) }
    })
  }, [sorted, board.cards])

  const placements = useMemo(() => {
    const out: Placement[] = []
    for (const lane of laneMeta) {
      const color = new THREE.Color(lane.rgb[0], lane.rgb[1], lane.rgb[2])
      lane.visible.forEach((card, idx) => {
        const row = Math.floor(idx / GRID_COLS)
        const colIdx = idx % GRID_COLS
        out.push({
          x: lane.x + (colIdx - (GRID_COLS - 1) / 2) * DX,
          y: CARD_H / 2 + 0.01,
          z: 0.4 + row * DZ,
          cardId: card.id,
          color,
        })
      })
    }
    return out
  }, [laneMeta])

  return (
    <div style={{ width: '100%', height: 'calc(100vh - 260px)', minHeight: 420 }}>
      <Canvas
        camera={{ position: [0, 13, 15], fov: 48 }}
        gl={{ antialias: true, alpha: true }}
        onPointerMissed={() => onSelect(null)}
        dpr={[1, 2]}
      >
        <ambientLight intensity={0.85} />
        <directionalLight position={[6, 14, 8]} intensity={0.7} />
        <directionalLight position={[-8, 6, -6]} intensity={0.25} />
        <Controls />
        <Floor lanes={laneMeta.map((l) => ({ x: l.x, rgb: l.rgb }))} />
        <LaneLabels
          lanes={laneMeta.map((l) => ({
            name: l.col.name,
            sub: l.sub,
            x: l.x,
            rgb: l.rgb,
          }))}
        />
        <Cards
          placements={placements}
          selectedId={selectedCardId}
          onSelect={(id) => onSelect(id)}
        />
      </Canvas>
    </div>
  )
}
