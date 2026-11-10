/**
 * The view model behind About's "Upcoming update" block.
 *
 * One record from two sources: the gateway's `update_upcoming` (a feed-offered
 * build, or a published advance notice) and the desktop updater's own lifecycle
 * state. A build the updater has found or staged is ready whatever the gateway
 * says. Plain string handling only.
 */
import type { UpdateState } from '../../hooks/useUpdateSubscription'

export type UpcomingUpdate = {
  state: 'announced' | 'ready'
  version: string
  /** `YYYY-MM-DD`, or "" when unknown. */
  date: string
  highlights: string[]
  /** An `https://` link to the full notes, or "". */
  notesUrl: string
}

export type GatewayUpcoming = {
  state: 'announced' | 'ready'
  version: string
  date: string
  highlights: string[]
  notes_url: string
} | null | undefined

const MAX_HIGHLIGHTS = 3
const MAX_HIGHLIGHT_CHARS = 140
/** Updater states in which a newer build is found, being fetched, or staged. */
const DESKTOP_READY_STATES = ['found', 'available', 'downloading', 'downloaded']

function cleanText(text: string): string {
  return text.split('**').join('').split('`').join('').split(' ').filter(Boolean).join(' ')
}

/** Up to three list items (or, with no list, plain lines) from a notes text. */
export function highlightsFromNotes(notes: string | undefined): string[] {
  const lines = (notes ?? '').slice(0, 4000).split('\n').map(line => line.trim())
  let items = lines
    .filter(line => line.startsWith('- ') || line.startsWith('* '))
    .map(line => cleanText(line.slice(2)))
  if (!items.some(Boolean)) items = lines.filter(line => line && !line.startsWith('#')).map(cleanText)
  return items
    .filter(Boolean)
    .map(item => (item.length > MAX_HIGHLIGHT_CHARS ? `${item.slice(0, MAX_HIGHLIGHT_CHARS - 1).trimEnd()}…` : item))
    .slice(0, MAX_HIGHLIGHTS)
}

/** `YYYY-MM-DD` as a UTC date, so the host time zone cannot move the day. */
export function dayToDate(day: string): Date | null {
  const parts = day.slice(0, 10).split('-').map(Number)
  if (parts.length !== 3 || parts.some(n => !Number.isInteger(n))) return null
  const [year, month, dayOfMonth] = parts
  if (month < 1 || month > 12 || dayOfMonth < 1 || dayOfMonth > 31) return null
  return new Date(Date.UTC(year, month - 1, dayOfMonth))
}

/** The link only when it is a plain `https://` URL; anything else is not drawn. */
export function safeNotesUrl(url: string | undefined): string {
  if (!url || !url.startsWith('https://') || url.length <= 'https://'.length) return ''
  return url.split('').some(ch => ch.trim() === '') ? '' : url
}

export function upcomingUpdate(gateway: GatewayUpcoming, desktop: UpdateState | null | undefined): UpcomingUpdate | null {
  const desktopVersion = desktop?.version
  const desktopReady = !!desktopVersion && DESKTOP_READY_STATES.includes(desktop?.state ?? '')
  if (desktopReady && (!gateway || gateway.version !== desktopVersion)) {
    return {
      state: 'ready',
      version: desktopVersion,
      date: (desktop?.pubDate ?? '').slice(0, 10),
      highlights: highlightsFromNotes(desktop?.notes),
      notesUrl: '',
    }
  }
  if (!gateway || !gateway.version) return null
  return {
    state: desktopReady ? 'ready' : gateway.state,
    version: gateway.version,
    date: gateway.date,
    highlights: gateway.highlights.slice(0, MAX_HIGHLIGHTS),
    notesUrl: safeNotesUrl(gateway.notes_url),
  }
}
