import { describe, it, expect } from 'vitest'
import {
  boardWidth,
  computeFit,
  computeFloor,
  lastCardZ,
  rowCount,
  LANE_W,
  LANE_GAP,
  CARD_D,
  DZ,
  FIRST_ROW_Z,
} from './framing'

const GRID_COLS = 6

describe('kanban3d framing math', () => {
  it('board width spans all lanes edge to edge', () => {
    expect(boardWidth(1)).toBe(LANE_W)
    expect(boardWidth(4)).toBe(4 * (LANE_W + LANE_GAP) - LANE_GAP)
  })

  it('rowCount ceilings and floors at sane bounds', () => {
    expect(rowCount(0, GRID_COLS)).toBe(1)
    expect(rowCount(1, GRID_COLS)).toBe(1)
    expect(rowCount(6, GRID_COLS)).toBe(1)
    expect(rowCount(7, GRID_COLS)).toBe(2)
    expect(rowCount(120, GRID_COLS)).toBe(20)
  })

  it('last card row never sits past the lane floor', () => {
    for (const rows of [1, 2, 5, 20]) {
      const { centerZ, depth } = computeFloor(rows)
      const floorBack = centerZ + depth / 2
      expect(lastCardZ(rows)).toBeLessThanOrEqual(floorBack)
      // floor also covers the label strip in front of row 1
      expect(centerZ - depth / 2).toBeLessThanOrEqual(FIRST_ROW_Z)
    }
  })

  it('card pitch math matches the scene grid', () => {
    expect(lastCardZ(1)).toBeCloseTo(FIRST_ROW_Z + CARD_D / 2)
    expect(lastCardZ(3)).toBeCloseTo(FIRST_ROW_Z + 2 * DZ + CARD_D / 2)
  })

  it('fit distance grows with lane count (wide boards pull the camera back)', () => {
    const two = computeFit({ aspect: 16 / 9, laneCount: 2, rows: 5 })
    const six = computeFit({ aspect: 16 / 9, laneCount: 6, rows: 5 })
    const dist = (f: { camera: [number, number, number]; target: [number, number, number] }) =>
      Math.hypot(f.camera[0] - f.target[0], f.camera[1] - f.target[1], f.camera[2] - f.target[2])
    expect(dist(six)).toBeGreaterThan(dist(two))
  })

  it('fit distance grows with rows (deep boards pull the camera back)', () => {
    const shallow = computeFit({ aspect: 16 / 9, laneCount: 4, rows: 2 })
    const deep = computeFit({ aspect: 16 / 9, laneCount: 4, rows: 20 })
    const dist = (f: { camera: [number, number, number] }) =>
      Math.hypot(f.camera[0], f.camera[1], f.camera[2])
    expect(dist(deep)).toBeGreaterThan(dist(shallow))
  })

  it('portrait aspect pulls the camera back further than landscape', () => {
    const wide = computeFit({ aspect: 16 / 9, laneCount: 4, rows: 10 })
    const narrow = computeFit({ aspect: 3 / 4, laneCount: 4, rows: 10 })
    const dist = (f: { camera: [number, number, number] }) =>
      Math.hypot(f.camera[0], f.camera[1], f.camera[2])
    expect(dist(narrow)).toBeGreaterThan(dist(wide))
  })

  it('fit keeps the whole floor in front of the camera (target inside bounds)', () => {
    const fit = computeFit({ aspect: 16 / 9, laneCount: 4, rows: 20 })
    const { centerZ, depth } = computeFloor(20)
    expect(fit.target[2]).toBeGreaterThanOrEqual(centerZ - depth / 2)
    expect(fit.target[2]).toBeLessThanOrEqual(centerZ + depth / 2)
    expect(fit.camera[1]).toBeGreaterThan(0)
    expect(fit.camera[2]).toBeGreaterThan(fit.target[2]) // behind, looking down the board
  })
})
