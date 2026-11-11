import { useRef, useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { useAppSelector } from '../../store'
import { slotRowByKey } from '../../store/slotIndex'
import { ChannelBrandIcon } from '../../components/ChannelBrandIcon'
import { useLabelTip } from '../../components/InstantTip'
import LinkedSurfacesSection from '../../components/LinkedSurfacesSection'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '../../components/ui/dropdown-menu'
import HeaderControl from './HeaderControl'

/**
 * The Bot thread's linked-channel control. A Bot's own chat has no session
 * menu, so once it was linked to Slack (or any channel) nothing on the page
 * could pause or unlink it. This opens the session menu's own channel rows
 * (`LinkedSurfacesSection`) under a header chip, so the verbs and what they
 * do are the same component everywhere; it renders only while the Bot's slot
 * has a link, and leaves with the last one.
 *
 * shortcut: the chip names the first link only; a Bot linked to two channels
 * still gets every row in the menu. Name them all if that becomes common.
 */
export default function BotLinkedChannels({ slotKey, inBar }: { slotKey: string; inBar: boolean }) {
  const { t } = useTranslation()
  const link = useAppSelector(s => slotRowByKey(s.dashboard.slots, slotKey))?.links?.[0]
  const label = link ? t('pages.membersPage.linked_to', { label: link.label }) : ''
  const tip = useLabelTip(label)
  const triggerRef = useRef<HTMLButtonElement | null>(null)
  // The tip names the chip; once its menu is open the rows say it, and a tip
  // left up would paint over the first row.
  const [open, setOpen] = useState(false)
  if (!link) {
    // The Unlink row keeps its menu open on select, so losing the last link
    // never closes the menu: drop the open state with the chip, or the menu
    // would pop open by itself when a link comes back.
    if (open) setOpen(false)
    return null
  }
  return (
    <DropdownMenu open={open} onOpenChange={setOpen}>
      <DropdownMenuTrigger asChild>
        <HeaderControl
          ref={triggerRef}
          inBar={inBar}
          chipClassName="glass-hover flex items-center justify-center gap-0.5 h-8 px-2 rounded-full shrink-0 text-text cursor-pointer"
          barClassName="mc-touch-hit flex items-center gap-0.5 p-2 rounded-md shrink-0 text-text hover:text-text-strong hover:bg-bg-hover cursor-pointer bg-transparent border-none"
          aria-label={label}
          data-testid="member-linked-channels"
          {...tip.handlers}
        >
          <ChannelBrandIcon channel={link.channel} size={inBar ? 16 : 14} />
          <ChevronDown size={11} className="shrink-0 text-muted" aria-hidden />
          {!open && tip.bubble}
        </HeaderControl>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="min-w-[220px]"
        onCloseAutoFocus={event => {
          // An Unlink of the last link removes this control in the same
          // commit, so Radix would restore focus to a detached trigger and
          // drop it on `document.body`. Hand it to the identity pill instead,
          // or the panel toggle while a docked Profile has the pill's place.
          if (triggerRef.current?.isConnected) return
          const target = document.querySelector<HTMLElement>(
            '[data-testid="member-identity-pill"], [data-testid="member-panel-toggle"]',
          )
          if (!target?.isConnected) return
          event.preventDefault()
          target.focus()
        }}
      >
        <LinkedSurfacesSection slotKey={slotKey} variant="dropdown" withoutSessionMenu />
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
