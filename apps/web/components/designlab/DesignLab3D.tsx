/**
 * Design Lab — 3D dock concept, rendered with Three.js.
 *
 * Dev-only showcase mounted at `/design-lab` in the Vite shell (see
 * vite-entry.tsx): the redesign of the OverlayDock corner per review —
 * one slim vertical rail hugging the right edge (game-HUD style) that
 * absorbs every overlay tenant (mole heart, consciousness M, debug,
 * output) so the bottom-right corner holds exactly ONE element.
 *
 * Renders all four approved material treatments (frosted glass, glossy
 * glass, soft neumorphic plastic, obsidian metal) and the full depth
 * stack: perspective tilt that straightens/lights up in the rail zone,
 * beveled edges (rounded-box geometry), soft real shadow onto the page
 * behind, extruded keys that push forward on hover, pulsing status LED
 * and count badge.
 *
 * Not a product page: never in the Next app tree, never in prod nav,
 * and lazy so three.js never reaches a production bundle.
 */
import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'

export const MATERIAL_KINDS = ['frosted', 'glossy', 'neumorph', 'obsidian'] as const
export type MaterialKind = (typeof MATERIAL_KINDS)[number]

const MATERIAL_META: Record<MaterialKind, { label: string; note: string }> = {
  frosted: {
    label: 'Frosted glass',
    note: 'Translucent, blurred depth — the page glows through',
  },
  glossy: {
    label: 'Glossy glass',
    note: 'Clear crystal, specular highlights, violet tint',
  },
  neumorph: {
    label: 'Soft 3D plastic',
    note: 'Matte neumorphic surface, soft extruded shadows',
  },
  obsidian: {
    label: 'Obsidian metal',
    note: 'Dark metal bevel, neon status glow',
  },
}

/** Tenants the rail absorbs — the de-cluttered corner in one column. */
const TENANT_GLYPHS = ['♥', 'M', '>_', '≡', '＋'] as const

function createRailMaterial(kind: MaterialKind): THREE.MeshPhysicalMaterial {
  const base = {
    metalness: 0,
    roughness: 0.5,
    envMapIntensity: 1,
  }
  switch (kind) {
    case 'frosted':
      return new THREE.MeshPhysicalMaterial({
        ...base,
        color: 0xf3eeff,
        transmission: 0.92,
        roughness: 0.52,
        thickness: 1.4,
        ior: 1.42,
        clearcoat: 0.5,
        clearcoatRoughness: 0.25,
        attenuationColor: new THREE.Color(0x8b5cf6),
        attenuationDistance: 8,
        envMapIntensity: 1,
      })
    case 'glossy':
      return new THREE.MeshPhysicalMaterial({
        ...base,
        color: 0xf6f2ff,
        transmission: 0.97,
        roughness: 0.07,
        thickness: 0.9,
        ior: 1.5,
        clearcoat: 1,
        clearcoatRoughness: 0.06,
        attenuationColor: new THREE.Color(0x7c3aed),
        attenuationDistance: 5,
        envMapIntensity: 1.3,
      })
    case 'neumorph':
      return new THREE.MeshPhysicalMaterial({
        ...base,
        color: 0xe6e2f2,
        transmission: 0,
        roughness: 0.92,
        metalness: 0,
        clearcoat: 0,
        envMapIntensity: 0.45,
      })
    case 'obsidian':
      return new THREE.MeshPhysicalMaterial({
        ...base,
        color: 0x0e0b18,
        transmission: 0,
        metalness: 0.92,
        roughness: 0.24,
        clearcoat: 0.5,
        clearcoatRoughness: 0.15,
        envMapIntensity: 1.15,
      })
  }
}

const ENV_INTENSITY: Record<MaterialKind, number> = {
  frosted: 1,
  glossy: 1.3,
  neumorph: 0.45,
  obsidian: 1.15,
}

/** White glyph with a dark outline so it reads on every material. */
function makeGlyphTexture(glyph: string): THREE.CanvasTexture {
  const size = 256
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  ctx.clearRect(0, 0, size, size)
  ctx.font = '600 132px "Segoe UI Symbol", "DejaVu Sans", sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.lineJoin = 'round'
  ctx.lineWidth = 14
  ctx.strokeStyle = 'rgba(8, 4, 20, 0.85)'
  ctx.strokeText(glyph, size / 2, size / 2 + 6)
  ctx.fillStyle = '#ffffff'
  ctx.fillText(glyph, size / 2, size / 2 + 6)
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.anisotropy = 4
  return tex
}

/** Red count badge glyph ("2" today). */
function makeBadgeTexture(count: string): THREE.CanvasTexture {
  const size = 128
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  ctx.clearRect(0, 0, size, size)
  ctx.font = '700 84px "Segoe UI", "DejaVu Sans", sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillStyle = '#ffffff'
  ctx.fillText(count, size / 2, size / 2 + 4)
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  return tex
}

interface DesignLabApi {
  setMaterial: (kind: MaterialKind) => void
}

/** Mock page content behind the rail so translucency/shadow have something to read against. */
function MockPageBackdrop() {
  return (
    <div
      aria-hidden
      className="pointer-events-none absolute inset-0 select-none space-y-4 p-6 pr-32"
    >
      <div className="flex items-center gap-2">
        <span className="h-3 w-3 rounded-full bg-primary/70" />
        <span className="h-3 w-28 rounded-full bg-muted" />
        <span className="ml-auto h-3 w-16 rounded-full bg-muted/70" />
      </div>
      <div className="grid grid-cols-3 gap-3">
        {[0, 1, 2].map((i) => (
          <div key={i} className="rounded-lg border border-border/70 bg-card/60 p-3">
            <div className="mb-2 h-3 w-2/3 rounded-full bg-muted" />
            <div className="space-y-1.5">
              <div className="h-2 w-full rounded-full bg-muted/70" />
              <div className="h-2 w-5/6 rounded-full bg-muted/70" />
              <div className="h-2 w-1/2 rounded-full bg-muted/70" />
            </div>
          </div>
        ))}
      </div>
      <div className="space-y-2.5 pt-2">
        <div className="max-w-[55%] rounded-2xl rounded-tl-sm border border-border/60 bg-card/70 px-4 py-2.5">
          <div className="h-2.5 w-40 rounded-full bg-muted" />
          <div className="mt-1.5 h-2.5 w-24 rounded-full bg-muted/70" />
        </div>
        <div className="ml-auto max-w-[55%] rounded-2xl rounded-tr-sm bg-primary/20 px-4 py-2.5">
          <div className="ml-auto h-2.5 w-32 rounded-full bg-primary/50" />
        </div>
        <div className="max-w-[55%] rounded-2xl rounded-tl-sm border border-border/60 bg-card/70 px-4 py-2.5">
          <div className="h-2.5 w-48 rounded-full bg-muted" />
        </div>
      </div>
      <div className="pt-2 text-[10px] tracking-wide text-muted-foreground/70">
        ↑ mock page behind — the rail floats over real content
      </div>
    </div>
  )
}

export default function DesignLab3D() {
  const [kind, setKind] = useState<MaterialKind>('frosted')
  const [webglFailed, setWebglFailed] = useState(false)
  const wrapRef = useRef<HTMLDivElement>(null)
  const apiRef = useRef<DesignLabApi | null>(null)
  // The frame loop reads the active material without re-running setup.
  const kindRef = useRef<MaterialKind>('frosted')

  // One-time scene setup — mutable per-frame state lives in the closure,
  // the React-facing API in apiRef (no re-render churn while animating).
  useEffect(() => {
    const container = wrapRef.current
    if (!container) return

    let renderer: THREE.WebGLRenderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    } catch {
      setWebglFailed(true)
      return
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearAlpha(0)
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.05
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFShadowMap
    renderer.domElement.style.cssText =
      'position:absolute;inset:0;width:100%;height:100%;pointer-events:none'
    container.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    const pmrem = new THREE.PMREMGenerator(renderer)
    const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture
    scene.environment = envTex

    const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 200)
    camera.position.set(0, 0, 42)

    const key = new THREE.DirectionalLight(0xffffff, 2)
    key.position.set(-6, 8, 14)
    key.castShadow = true
    key.shadow.mapSize.set(1024, 1024)
    key.shadow.radius = 7
    key.shadow.camera.left = -16
    key.shadow.camera.right = 16
    key.shadow.camera.top = 16
    key.shadow.camera.bottom = -16
    scene.add(key)
    const rim = new THREE.DirectionalLight(0x8b5cf6, 1.4)
    rim.position.set(9, -5, -8)
    scene.add(rim)
    scene.add(new THREE.AmbientLight(0xffffff, 0.4))

    // Soft real shadow cast onto the (DOM) page behind — ShadowMaterial
    // renders only the shadow, alpha-composited over the mock backdrop.
    const shadowPlane = new THREE.Mesh(
      new THREE.PlaneGeometry(90, 90),
      new THREE.ShadowMaterial({ opacity: 0.3 }),
    )
    shadowPlane.position.z = -1.6
    shadowPlane.receiveShadow = true
    scene.add(shadowPlane)

    /* ---------------- rail group ---------------- */
    const group = new THREE.Group()
    scene.add(group)

    const disposables: { dispose(): void }[] = [shadowPlane.geometry, shadowPlane.material]
    const track = <T extends { dispose(): void }>(res: T): T => {
      disposables.push(res)
      return res
    }

    const railW = 3.2
    const railH = 22
    const railD = 1.6
    const railGeo = track(new RoundedBoxGeometry(railW, railH, railD, 6, 0.55))
    let railMat = createRailMaterial('frosted')
    const rail = new THREE.Mesh(railGeo, railMat)
    rail.castShadow = true
    group.add(rail)

    const buttonGeo = track(new RoundedBoxGeometry(2.1, 2.1, 1, 4, 0.32))
    const glyphGeo = track(new THREE.PlaneGeometry(1.5, 1.5))
    const buttons: THREE.Mesh[] = []
    const glyphMats: THREE.MeshBasicMaterial[] = []
    const buttonBaseZ = railD / 2 + 0.5 - 0.18
    const spacing = 3.4
    const topY = ((TENANT_GLYPHS.length - 1) * spacing) / 2

    TENANT_GLYPHS.forEach((glyph, i) => {
      const btn = new THREE.Mesh(buttonGeo, railMat)
      btn.position.set(0, topY - i * spacing, buttonBaseZ)
      btn.castShadow = true
      group.add(btn)
      buttons.push(btn)

      const glyphMat = track(
        new THREE.MeshBasicMaterial({
          map: makeGlyphTexture(glyph),
          transparent: true,
          toneMapped: false,
          depthWrite: false,
        }),
      )
      glyphMats.push(glyphMat)
      const plate = new THREE.Mesh(glyphGeo, glyphMat)
      plate.position.z = 0.52
      plate.renderOrder = 2
      btn.add(plate)
    })

    // Count badge — worst actionable severity (critical: 2).
    const badgeDiscGeo = track(new THREE.CircleGeometry(0.6, 40))
    const badgeDiscMat = track(
      new THREE.MeshStandardMaterial({
        color: 0xef4444,
        roughness: 0.35,
        metalness: 0.1,
        emissive: new THREE.Color(0xef4444),
        emissiveIntensity: 0.35,
      }),
    )
    const badge = new THREE.Mesh(badgeDiscGeo, badgeDiscMat)
    badge.position.set(railW / 2 - 0.25, topY + 1.35, railD / 2 + 0.03)
    group.add(badge)
    const badgeTextGeo = track(new THREE.PlaneGeometry(0.86, 0.86))
    const badgeTextMat = track(
      new THREE.MeshBasicMaterial({
        map: makeBadgeTexture('2'),
        transparent: true,
        toneMapped: false,
        depthWrite: false,
      }),
    )
    const badgeText = new THREE.Mesh(badgeTextGeo, badgeTextMat)
    badgeText.position.copy(badge.position)
    badgeText.position.z += 0.02
    group.add(badgeText)

    // Status LED — always pulsing, brighter when the rail is engaged.
    const ledGeo = track(new THREE.SphereGeometry(0.26, 20, 16))
    const ledMat = track(
      new THREE.MeshStandardMaterial({
        color: 0xff5c5c,
        emissive: new THREE.Color(0xff3b3b),
        emissiveIntensity: 1.4,
        roughness: 0.3,
      }),
    )
    const led = new THREE.Mesh(ledGeo, ledMat)
    led.position.set(0, railH / 2 - 1, railD / 2)
    group.add(led)

    /* ---------------- interaction state ---------------- */
    let hover = 0
    let hoverTarget = 0
    let pointerX = 0
    let pointerY = 0
    const BASE_TILT = 0.36

    const onPointerMove = (e: PointerEvent) => {
      const rect = container.getBoundingClientRect()
      const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1
      const ny = ((e.clientY - rect.top) / rect.height) * 2 - 1
      pointerX = nx
      pointerY = ny
      // The rail zone: right ~38% of the frame (where the rail lives).
      hoverTarget = nx > 0.24 ? 1 : 0
    }
    const onPointerLeave = () => {
      hoverTarget = 0
      pointerX = 0
      pointerY = 0
    }
    container.addEventListener('pointermove', onPointerMove)
    container.addEventListener('pointerleave', onPointerLeave)

    /* ---------------- layout ---------------- */
    const layout = () => {
      const w = container.clientWidth || 1
      const h = container.clientHeight || 1
      renderer.setSize(w, h, false)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      const halfH = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * camera.position.z
      const halfW = halfH * camera.aspect
      group.position.x = halfW - 2.4
    }
    layout()
    const ro = new ResizeObserver(layout)
    ro.observe(container)

    /* ---------------- material switching ---------------- */
    apiRef.current = {
      setMaterial: (next: MaterialKind) => {
        railMat.dispose()
        railMat = createRailMaterial(next)
        rail.material = railMat
        for (const b of buttons) b.material = railMat
      },
    }

    /* ---------------- frame loop ---------------- */
    const t0 = performance.now()
    renderer.setAnimationLoop(() => {
      const t = (performance.now() - t0) / 1000
      hover += (hoverTarget - hover) * 0.06

      // Perspective tilt: angled HUD pose at rest → straightens + lifts
      // toward the camera while engaged; subtle pointer-following.
      const targetY = BASE_TILT * (1 - hover * 0.9) + pointerX * 0.1
      const targetX = -pointerY * 0.1 * hover
      group.rotation.y += (targetY + Math.sin(t * 0.5) * 0.025 - group.rotation.y) * 0.08
      group.rotation.x += (targetX - group.rotation.x) * 0.08
      group.position.y = Math.sin(t * 0.8) * 0.12

      // Light-up: env reflections bloom, keys extrude further, LED hot.
      railMat.envMapIntensity = ENV_INTENSITY[kindRef.current] * (1 + hover * 0.7)
      const z = buttonBaseZ + hover * 0.3
      for (const b of buttons) b.position.z = z
      const pulse = 0.9 + Math.sin(t * 3) * 0.7
      ledMat.emissiveIntensity = pulse + hover * 1.4
      badgeDiscMat.emissiveIntensity = 0.35 + hover * 0.65
      const glyphOpacity = 0.82 + hover * 0.18
      for (const g of glyphMats) g.opacity = glyphOpacity

      renderer.render(scene, camera)
    })

    return () => {
      renderer.setAnimationLoop(null)
      ro.disconnect()
      container.removeEventListener('pointermove', onPointerMove)
      container.removeEventListener('pointerleave', onPointerLeave)
      apiRef.current = null
      for (const d of disposables) d.dispose()
      railMat.dispose()
      envTex.dispose()
      pmrem.dispose()
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [])

  // React-facing material switch (disposes the previous material inside).
  useEffect(() => {
    kindRef.current = kind
    apiRef.current?.setMaterial(kind)
  }, [kind])

  return (
    <div className="mx-auto max-w-6xl space-y-5 p-6" data-testid="designlab-3d">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs font-medium uppercase tracking-[0.18em] text-primary">
            Design lab · concept v2
          </p>
          <h1 className="text-2xl font-semibold">3D dock — one rail, one corner</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            The OverlayDock redesign: a single beveled rail hugs the right edge and absorbs every
            overlay tenant — the heart (site doctor), the consciousness&nbsp;M, debug and output —
            so the bottom-right corner holds exactly one element. Tilt, glow and extrusion respond
            to the pointer; the LED never stops pulsing.
          </p>
        </div>
        <p className="rounded-full border border-border px-3 py-1 text-[11px] text-muted-foreground">
          dev-only · /design-lab · not a product page
        </p>
      </header>

      <div className="grid gap-4 lg:grid-cols-[1fr_16rem]">
        <div
          ref={wrapRef}
          className="relative h-[540px] overflow-hidden rounded-xl border border-border bg-background shadow-inner"
        >
          <MockPageBackdrop />
          {webglFailed ? (
            <div
              data-testid="designlab-webgl-fallback"
              className="absolute inset-0 flex items-center justify-center text-sm text-muted-foreground"
            >
              WebGL is not available in this browser — 3D preview disabled.
            </div>
          ) : null}
        </div>

        <aside className="space-y-3">
          <div>
            <h2 className="mb-2 text-xs font-medium uppercase tracking-widest text-muted-foreground">
              Material
            </h2>
            <div className="space-y-1.5" role="radiogroup" aria-label="Rail material">
              {MATERIAL_KINDS.map((m) => {
                const active = m === kind
                return (
                  <button
                    key={m}
                    type="button"
                    role="radio"
                    aria-checked={active}
                    data-testid={`material-${m}`}
                    onClick={() => setKind(m)}
                    className={
                      active
                        ? 'w-full cursor-pointer rounded-lg border border-primary bg-primary/10 px-3 py-2 text-left text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'
                        : 'w-full cursor-pointer rounded-lg border border-border px-3 py-2 text-left text-sm text-muted-foreground transition-colors hover:border-primary/50 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'
                    }
                  >
                    {MATERIAL_META[m].label}
                  </button>
                )
              })}
            </div>
            <p className="mt-2 text-xs leading-relaxed text-muted-foreground/80">
              {MATERIAL_META[kind].note}
            </p>
          </div>

          <div className="rounded-lg border border-border/70 bg-card/60 p-3">
            <h3 className="mb-1.5 text-xs font-medium uppercase tracking-widest text-muted-foreground">
              Depth stack
            </h3>
            <ul className="space-y-1 text-xs leading-relaxed text-muted-foreground">
              <li>▸ Perspective tilt — straightens in the rail zone</li>
              <li>▸ Beveled rounded edges (light-catching)</li>
              <li>▸ Soft shadow onto the page behind</li>
              <li>▸ Extruded keys push forward on hover</li>
              <li>▸ Pulsing LED + worst-count badge</li>
            </ul>
          </div>

          <div className="rounded-lg border border-border/70 bg-card/60 p-3 text-xs leading-relaxed text-muted-foreground">
            <span className="font-medium text-foreground">Corner plan:</span> rail replaces the
            stacked pill + FAB + status trio — status bar stays, everything else folds into this one
            column.
          </div>
        </aside>
      </div>
    </div>
  )
}
