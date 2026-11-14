'use client'

/**
 * Magazine — the Noir Violet brand guide.
 *
 * Purely presentational: nine numbered sections, each a self-contained
 * component in its own file (the interactive gallery and state demos own
 * their demo state locally). Only the page shell — title, print styles,
 * section order and footer — belongs here.
 */
import { PageContainer } from '@/components/PageContainer'
import { Separator, Badge } from '@sloughgpt/strui'
import { MagazineHero } from './magazine-hero'
import { MagazineColorPalette } from './magazine-colors'
import { MagazineTypography } from './magazine-typography'
import { MagazineComponentGallery } from './magazine-components'
import { MagazineSpacingRadius } from './magazine-spacing'
import { MagazineInteractiveStates } from './magazine-interactive'
import { MagazineDarkPreview } from './magazine-dark-preview'
import { MagazineAccentThemes } from './magazine-accents'

export default function MagazinePage() {
  return (
    <PageContainer
      title={
        <span className="sl-h1">
          <span className="text-primary">Noir Violet</span> Brand Guide
        </span>
      }
      subtitle="A design system reference for Man"
    >
      <style>{`
        @media print {
          .sl-page { overflow: visible !important; height: auto !important; }
          .sl-app-shell { height: auto !important; overflow: visible !important; }
          .sl-app-body { overflow: visible !important; }
          .sl-app-main { overflow: visible !important; }
          .sl-app-content { overflow: visible !important; }
          .no-print { display: none !important; }
          * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
          @page { margin: 1.5cm; size: A4; }
          .magazine-card { break-inside: avoid; page-break-inside: avoid; }
        }
      `}</style>

      <MagazineHero />
      <MagazineColorPalette />
      <MagazineTypography />
      <MagazineComponentGallery />
      <MagazineSpacingRadius />
      <MagazineInteractiveStates />
      <MagazineDarkPreview />
      <MagazineAccentThemes />

      <Separator />

      {/* ─── Footer ───────────────────────────────────────────── */}
      <div className="flex items-center justify-between pb-4">
        <div className="text-xs text-muted-foreground">Man &middot; Noir Violet Design System</div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary">Locked</Badge>
          <span className="font-mono text-[10px] text-muted-foreground">v1.0</span>
        </div>
      </div>
    </PageContainer>
  )
}
