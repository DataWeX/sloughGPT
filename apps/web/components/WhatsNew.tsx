'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { IconX, IconChevronRight, IconChevronLeft, IconCheck } from '@/components/ui'
import { whatsNewItems, type WhatsNewItem } from '@/lib/whats-new-data'
import { cn } from '@/lib/cn'

const STORAGE_KEY = 'man_whats_new_seen'

export function WhatsNewTrigger() {
  const [open, setOpen] = useState(false)
  const [hasUnseen, setHasUnseen] = useState(false)
  const [tourMode, setTourMode] = useState(false)

  useEffect(() => {
    const seen = localStorage.getItem(STORAGE_KEY)
    const latestId = whatsNewItems[0]?.id
    setHasUnseen(seen !== latestId)
  }, [])

  const handleOpen = () => {
    setOpen(true)
    setTourMode(false)
    localStorage.setItem(STORAGE_KEY, whatsNewItems[0]?.id || '')
    setHasUnseen(false)
  }

  const handleStartTour = () => {
    setTourMode(true)
  }

  return (
    <>
      <button
        onClick={handleOpen}
        className={cn(
          "fixed bottom-16 right-4 z-40 flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium shadow-lg transition-all hover:scale-105",
          hasUnseen
            ? "bg-primary text-primary-foreground shadow-primary/30 animate-pulse"
            : "bg-card border border-border/60 text-muted-foreground hover:text-foreground"
        )}
        aria-label="What's new"
      >
        <span className="text-sm">✨</span>
{hasUnseen && <span>What&apos;s new</span>}
{!hasUnseen && <span>Updates</span>}
      </button>
      <WhatsNewDialog
        open={open}
        onClose={() => setOpen(false)}
        tourMode={tourMode}
        onStartTour={handleStartTour}
      />
    </>
  )
}

export function WhatsNewDialog({
  open,
  onClose,
  tourMode,
  onStartTour,
}: {
  open: boolean
  onClose: () => void
  tourMode: boolean
  onStartTour: () => void
}) {
  const [step, setStep] = useState(0)
  const cardRefs = useRef<(HTMLDivElement | null)[]>([])
  const [exiting, setExiting] = useState(false)

  useEffect(() => {
    if (!open) { setStep(0); setExiting(false) }
  }, [open])

  useEffect(() => {
    if (tourMode && cardRefs.current[step]) {
      cardRefs.current[step]?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }, [step, tourMode])

  if (!open) return null

  const total = whatsNewItems.length
  const isTour = tourMode && total > 0

  const handleClose = () => {
    setExiting(true)
    setTimeout(onClose, 200)
  }

  return (
    <div
      className={cn(
        "fixed inset-0 z-50 flex items-center justify-center p-4 transition-opacity duration-200",
        exiting ? "opacity-0" : "opacity-100"
      )}
    >
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm" onClick={handleClose} />
      <div className={cn(
        "relative z-10 w-full max-w-lg max-h-[85vh] rounded-2xl border border-border/50 bg-background shadow-2xl transition-all duration-200 flex flex-col",
        exiting ? "scale-95 opacity-0" : "scale-100 opacity-100"
      )}>
        {/* Header */}
        <div className="sticky top-0 z-20 flex items-center justify-between px-5 py-3 border-b border-border/40 bg-background/90 backdrop-blur-sm rounded-t-2xl shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-lg">{isTour ? '🎯' : '✨'}</span>
            <h2 className="text-sm font-semibold">
              {isTour ? `Tour (${step + 1}/${total})` : "What's New"}
            </h2>
          </div>
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={handleClose}>
            <IconX className="h-4 w-4" />
          </Button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-4 py-3">
          {isTour ? (
            /* Single step with spotlight */
            <div className="relative">
              {whatsNewItems.map((item, i) => (
                <div
                  key={item.id}
                  ref={(el) => { cardRefs.current[i] = el }}
                  className={cn(
                    "transition-all duration-300 rounded-xl",
                    i === step
                      ? "opacity-100 scale-100 ring-2 ring-primary/30 ring-offset-2 ring-offset-background shadow-xl shadow-primary/10"
                      : "opacity-0 absolute inset-0 pointer-events-none"
                  )}
                >
                  {i === step && (
                    <div className="p-4">
                      <div className="flex items-center gap-3 mb-3">
                        <div className="flex items-center justify-center w-12 h-12 rounded-2xl bg-gradient-to-br from-primary/25 to-primary/10 border border-primary/30 text-2xl">
                          {item.icon}
                        </div>
                        <div>
                          <h3 className="text-base font-semibold">{item.title}</h3>
                          <span className="text-xs text-muted-foreground/60">{item.date}</span>
                        </div>
                      </div>
                      <p className="text-sm text-muted-foreground leading-relaxed mb-3">
                        {item.description}
                      </p>
                      <div className="flex items-center gap-2 flex-wrap">
                        {item.tags?.map(t => (
                          <span key={t} className="text-[10px] px-2 py-0.5 rounded-full bg-muted/60 text-muted-foreground/70 font-medium">
                            {t}
                          </span>
                        ))}
                      </div>
                      {item.href && (
                        <Link
                          href={item.href}
                          onClick={handleClose}
                          className="inline-flex items-center gap-1 mt-4 text-xs text-primary hover:text-primary/80 font-medium"
                        >
                          Try it now <IconChevronRight className="h-3 w-3" />
                        </Link>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            /* Full list */
            <div className="space-y-0">
              {whatsNewItems.map((item, i) => (
                <WhatsNewCard key={item.id} item={item} isLast={i === total - 1} />
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="sticky bottom-0 px-4 py-3 border-t border-border/40 bg-background/90 backdrop-blur-sm rounded-b-2xl shrink-0">
          {isTour ? (
            <div className="flex items-center justify-between gap-2">
              <Button
                variant="ghost"
                size="sm"
                className="h-8 text-xs"
                onClick={() => { if (step > 0) setStep(step - 1) }}
                disabled={step === 0}
              >
                <IconChevronLeft className="h-3 w-3 mr-1" /> Back
              </Button>
              <div className="flex gap-1">
                {Array.from({ length: total }).map((_, i) => (
                  <span
                    key={i}
                    className={cn(
                      "h-1.5 rounded-full transition-all duration-300",
                      i === step ? "w-4 bg-primary" : "w-1.5 bg-muted-foreground/20"
                    )}
                  />
                ))}
              </div>
              {step < total - 1 ? (
                <Button size="sm" className="h-8 text-xs" onClick={() => setStep(step + 1)}>
                  Next <IconChevronRight className="h-3 w-3 ml-1" />
                </Button>
              ) : (
                <Button size="sm" className="h-8 text-xs" onClick={handleClose}>
                  <IconCheck className="h-3 w-3 mr-1" /> Done
                </Button>
              )}
            </div>
          ) : (
            <div className="flex gap-2">
              <Button variant="outline" className="flex-1 text-sm h-8" onClick={onStartTour}>
                🎯 Take guided tour
              </Button>
              <Button className="flex-1 text-sm h-8" onClick={handleClose}>
                Got it
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function WhatsNewCard({ item, isLast }: { item: WhatsNewItem; isLast: boolean }) {
  return (
    <div className={cn("group relative", !isLast && "border-b border-border/20 pb-3 mb-3")}>
      <div className="flex gap-3">
        <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-primary/20 to-primary/5 border border-primary/20 text-lg shrink-0">
          {item.icon}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-medium">{item.title}</h3>
            <span className="text-[10px] text-muted-foreground/50">{item.date}</span>
          </div>
          <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed">{item.description}</p>
          <div className="flex items-center gap-2 mt-1.5">
            {item.tags?.map(t => (
              <span key={t} className="text-[9px] px-1.5 py-0.5 rounded-full bg-muted/60 text-muted-foreground/70 font-medium">
                {t}
              </span>
            ))}
            {item.href && (
              <Link
                href={item.href}
                onClick={(e) => e.stopPropagation()}
                className="inline-flex items-center gap-0.5 text-[10px] text-primary hover:text-primary/80 font-medium ml-auto"
              >
                Try it
                <IconChevronRight className="h-2.5 w-2.5" />
              </Link>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
