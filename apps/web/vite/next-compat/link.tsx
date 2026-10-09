import { Link as RouterLink, type LinkProps as RouterLinkProps } from 'react-router-dom'
import type { AnchorHTMLAttributes, ReactNode } from 'react'

export type LinkProps = Omit<AnchorHTMLAttributes<HTMLAnchorElement>, 'href'> & {
  href: RouterLinkProps['to'] | string | URL
  replace?: boolean
  scroll?: boolean
  prefetch?: boolean
  passHref?: boolean
  children?: ReactNode
}

/** next/link → react-router Link (href→to). */
export default function Link({
  href,
  replace,
  scroll: _scroll,
  prefetch: _prefetch,
  passHref: _passHref,
  ...rest
}: LinkProps) {
  const to = typeof href === 'string' || href instanceof URL ? String(href) : href
  return <RouterLink replace={replace} to={to} {...(rest as Omit<RouterLinkProps, 'to'>)} />
}
