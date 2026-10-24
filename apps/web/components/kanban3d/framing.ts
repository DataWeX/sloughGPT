/* Pure camera/floor framing math for KanbanScene — no three.js import so
   it stays unit-testable in plain vitest. Board footprint derives from the
   lane constants; camera fit covers the whole footprint at any aspect ratio
   and lane/row count, so lanes are never cut at the viewport edge and cards
   never sit past the lane floor. */

export const LANE_W = 5.6
export const LANE_GAP = 1.4
export const CARD_D = 0.44
export const DZ = 0.58
export const FIRST_ROW_Z = 0.4
export const LABEL_Z = -1.6
export const FOV_DEG = 48

/** Label strip sits in front of the first card row; keep it on the floor. */
export const Z_FRONT = LABEL_Z - 0.8

export function boardWidth(laneCount: number): number {
  return Math.max(1, laneCount) * (LANE_W + LANE_GAP) - LANE_GAP
}

export function rowCount(visibleCount: number, gridCols: number): number {
  return Math.max(1, Math.ceil(Math.max(0, visibleCount) / Math.max(1, gridCols)))
}

export function lastCardZ(rows: number): number {
  return FIRST_ROW_Z + (Math.max(1, rows) - 1) * DZ + CARD_D / 2
}

export function computeFloor(rows: number): { centerZ: number; depth: number } {
  const zBack = lastCardZ(rows) + 1.2
  return { centerZ: (Z_FRONT + zBack) / 2, depth: zBack - Z_FRONT }
}

export interface Fit {
  camera: [number, number, number]
  target: [number, number, number]
}

export function computeFit({
  aspect,
  laneCount,
  rows,
}: {
  aspect: number
  laneCount: number
  rows: number
}): Fit {
  const margin = 1.12
  const width = boardWidth(laneCount) * margin
  const { centerZ, depth } = computeFloor(rows)
  const vfov = (FOV_DEG * Math.PI) / 180
  const hfov = 2 * Math.atan(Math.tan(vfov / 2) * Math.max(0.2, aspect))
  const distV = depth / 2 / Math.tan(vfov / 2)
  const distH = width / 2 / Math.tan(hfov / 2)
  const dist = Math.max(distV, distH)
  const pitch = Math.PI / 4.2
  const cy = dist * Math.sin(pitch)
  const cz = dist * Math.cos(pitch)
  return { camera: [0, cy, centerZ + cz], target: [0, 0, centerZ] }
}
