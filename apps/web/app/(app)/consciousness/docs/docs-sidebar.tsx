'use client'

/** Category nav: picks which endpoint group the list shows. */
import { CATEGORIES } from './docs-endpoints'
import type { DocsWorkbench } from './useDocsWorkbench'

export function DocsSidebar({ wb }: { wb: DocsWorkbench }) {
  const { activeCategory, setActiveCategory, t } = wb

  return (
    <nav className="w-48 shrink-0 space-y-1 sticky top-6 self-start">
      {CATEGORIES.map((cat) => (
        <button
          key={cat.id}
          type="button"
          onClick={() => setActiveCategory(cat.id)}
          className={`w-full text-left px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
            activeCategory === cat.id
              ? 'bg-primary/10 text-primary'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
          }`}
        >
          {t(cat.labelKey)}
        </button>
      ))}
    </nav>
  )
}
