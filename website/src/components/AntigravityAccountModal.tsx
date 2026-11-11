import { useEffect, useState, type ReactNode } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  AlertCircle,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Eye,
  EyeOff,
  Info,
  Loader2,
  RefreshCw,
  Sparkles,
  UserRound,
} from 'lucide-react'

import { api } from '../api/client'
import {
  isAntigravityUsageReading,
  parseAntigravityUsagePayload,
  type AntigravityModelQuota,
  type AntigravityQuotaPool,
  type AntigravityUsageEnvelope,
  type AntigravityUsageState,
} from '../api/antigravityUsage'
import { fmtDateTime, fmtNumber, fmtPercent, fmtTime, toDate } from '../i18n/format'
import { i18nT } from '../i18n/t'
import { safeGetItem, safeSetItem } from '../utils/safeStorage'
import Clickable from './Clickable'
import ErrorNotice from './ErrorNotice'
import Modal from './Modal'

const EMAIL_HIDDEN_KEY = 'kirocrew:account-email-hidden'
const IN_FLIGHT_RECHECK_MS = 3000

type QuotaRefreshNotice =
  | { kind: 'in_flight' }
  | { kind: 'stale' }
  | { kind: 'failed' }

function classifyRefreshError(err: unknown): QuotaRefreshNotice {
  const e = err as { status?: unknown } | null
  if (typeof e === 'object' && e !== null && e.status === 409) return { kind: 'in_flight' }
  return { kind: 'failed' }
}

function refreshNoticeMessage(notice: QuotaRefreshNotice): string {
  switch (notice.kind) {
    case 'in_flight':
      return i18nT('components.antigravityAccountModal.refresh_in_flight')
    case 'stale':
      return i18nT('components.antigravityAccountModal.refresh_returned_stale')
    default:
      return i18nT('components.antigravityAccountModal.quota_unavailable')
  }
}

function emailInitials(email: string): string {
  const localPart = email.split('@', 1)[0] ?? ''
  const parts = localPart.split(/[^A-Za-z0-9]+/).filter(Boolean)
  if (parts.length >= 2) return `${parts[0][0]}${parts[1][0]}`.toUpperCase()
  return (parts[0] ?? '?').slice(0, 2).toUpperCase()
}

function formatResetTimestamp(iso: string | null): string | null {
  const d = toDate(iso)
  if (!d) return null
  const now = new Date()
  const sameDay =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate()
  return sameDay ? fmtTime(d) : fmtDateTime(d)
}

function poolTitleNode(pool: AntigravityQuotaPool): ReactNode {
  const lower = pool.group.toLowerCase()
  if (lower.includes('gemini')) {
    return i18nT('components.antigravityAccountModal.pool_gemini_models')
  }
  if (lower.includes('claude') || lower.includes('gpt')) {
    return i18nT('components.antigravityAccountModal.pool_claude_gpt_models')
  }
  if (!pool.group) {
    return i18nT('components.antigravityAccountModal.window_rolling')
  }
  return <span data-i18n-opaque>{pool.group}</span>
}

function poolWindowLabel(windowKind: string): string {
  if (windowKind === '5h') return i18nT('components.antigravityAccountModal.window_5h')
  if (windowKind === 'weekly') return i18nT('components.antigravityAccountModal.window_weekly')
  return i18nT('components.antigravityAccountModal.window_rolling')
}

function barFillClass(fraction: number): string {
  if (fraction <= 0.05) return 'bg-danger'
  if (fraction < 0.2) return 'bg-warn'
  return 'bg-accent'
}

function pctToneClass(fraction: number): string {
  if (fraction <= 0.05) return 'text-danger'
  if (fraction < 0.2) return 'text-warn'
  return 'text-text-strong'
}

function useAntigravityQuotaRefresh() {
  const queryClient = useQueryClient()
  const [notice, setNotice] = useState<QuotaRefreshNotice | null>(null)
  const [checkedAt, setCheckedAt] = useState<Date | null>(null)

  const mutation = useMutation({
    mutationFn: () => api.antigravityUsageRefresh(),
    retry: false,
    onMutate: () => setNotice(null),
    onSuccess: (res: AntigravityUsageEnvelope) => {
      const parsed = parseAntigravityUsagePayload(res)
      queryClient.setQueryData(['antigravity-usage'], parsed)
      void queryClient.invalidateQueries({ queryKey: ['antigravity-usage'] })
      if (parsed === 'api-key' || parsed === 'signin-required') {
        return
      }
      if (!isAntigravityUsageReading(parsed)) {
        setNotice({ kind: 'failed' })
        return
      }
      if (parsed.stale) {
        setNotice({ kind: 'stale' })
        return
      }
      setCheckedAt(new Date())
    },
    onError: err => setNotice(classifyRefreshError(err)),
  })

  const inFlight = notice?.kind === 'in_flight'
  useEffect(() => {
    if (!inFlight) return
    let cancelled = false
    const timer = window.setTimeout(() => {
      void queryClient.invalidateQueries({ queryKey: ['antigravity-usage'] }).then(() => {
        if (!cancelled) setNotice(current => (current?.kind === 'in_flight' ? null : current))
      })
    }, IN_FLIGHT_RECHECK_MS)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [inFlight, queryClient])

  const held = notice?.kind === 'in_flight'
  return { mutation, notice, checkedAt, held }
}

function RefreshButton({
  pending,
  held,
  onClick,
  compact,
}: {
  pending: boolean
  held: boolean
  onClick: () => void
  compact: boolean
}) {
  const idleLabel = i18nT('components.antigravityAccountModal.refresh_quota')
  const pendingLabel = i18nT('components.antigravityAccountModal.refreshing_quota')
  const shape = compact
    ? 'rounded-md border border-border bg-transparent px-2.5 py-1 text-[12px] font-medium text-muted hover:border-border-strong hover:text-text'
    : 'rounded-md bg-accent px-3 py-1.5 text-[13px] font-medium text-accent-fg hover:brightness-110'
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={pending || held}
      aria-busy={pending || undefined}
      className={`inline-flex items-center gap-1.5 self-start transition-colors disabled:cursor-default disabled:opacity-60 focus-ring-accent ${shape}`}
    >
      {pending
        ? <Loader2 className="lucide-inline animate-spin" aria-hidden="true" />
        : <RefreshCw className="lucide-inline" aria-hidden="true" />}
      {pending ? pendingLabel : idleLabel}
    </button>
  )
}

function AccountIdentity({ usage }: { usage: AntigravityUsageState }) {
  const [emailHidden, setEmailHidden] = useState(
    () => safeGetItem(EMAIL_HIDDEN_KEY) !== '0',
  )

  const toggleEmailVisibility = () => {
    setEmailHidden(hidden => {
      const next = !hidden
      safeSetItem(EMAIL_HIDDEN_KEY, next ? '1' : '0')
      return next
    })
  }

  const reading = isAntigravityUsageReading(usage) ? usage : undefined
  const email = reading?.email ?? null
  const tierName = reading?.tierName ?? null

  return (
    <div className="flex flex-col items-center px-4 py-3 text-center">
      <div className="relative flex h-14 w-14 shrink-0 items-center justify-center rounded-full border border-accent/30 bg-accent/10 text-accent shadow-sm">
        {email ? (
          <span aria-hidden="true" className="text-[17px] font-semibold tracking-[0.04em]" data-i18n-opaque>
            {emailInitials(email)}
          </span>
        ) : (
          <UserRound className="h-6 w-6" strokeWidth={1.7} />
        )}
      </div>
      <div className="mt-3 min-w-0 max-w-full">
        {usage === null ? (
          <div className="flex items-center justify-center gap-2 text-[13px] text-muted">
            <Loader2 className="lucide-inline animate-spin" /> {i18nT('components.antigravityAccountModal.checking_account')}
          </div>
        ) : !reading || (!email && !tierName) ? (
          <div className="flex items-center justify-center gap-2 text-[13px] text-muted">
            <AlertCircle className="lucide-inline" /> {i18nT('components.antigravityAccountModal.account_details_unavailable')}
          </div>
        ) : (
          <>
            {email && (
              <div className="flex min-w-0 items-center justify-center gap-2">
                <span
                  data-i18n-opaque
                  className={`min-w-0 truncate text-[16px] font-semibold leading-6 text-text-strong transition-[filter,opacity] duration-150 ${emailHidden ? 'select-none blur-[5px] opacity-60' : ''}`}
                  title={emailHidden ? undefined : email}
                >
                  {email}
                </span>
                <Clickable
                  onClick={toggleEmailVisibility}
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border text-muted transition-colors hover:border-accent/35 hover:bg-accent/10 hover:text-accent focus-ring-accent"
                  aria-label={i18nT(emailHidden ? 'components.antigravityAccountModal.show_email' : 'components.antigravityAccountModal.hide_email')}
                  title={i18nT(emailHidden ? 'components.antigravityAccountModal.show_email' : 'components.antigravityAccountModal.hide_email')}
                >
                  {emailHidden ? <Eye className="lucide-inline" /> : <EyeOff className="lucide-inline" />}
                </Clickable>
              </div>
            )}
            <div className="mt-2.5 flex flex-wrap items-center justify-center gap-2">
              {tierName && (
                <span
                  data-i18n-opaque
                  className="inline-flex items-center rounded-full border border-accent/30 bg-accent/10 px-2.5 py-0.5 text-[12px] font-medium text-accent"
                >
                  {tierName}
                </span>
              )}
              <span className="inline-flex items-center rounded-full border border-border bg-bg px-2.5 py-0.5 text-[12px] text-muted">
                {i18nT('components.antigravityAccountModal.oauth_badge')}
              </span>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function QuotaSkeleton() {
  return (
    <div className="flex flex-col gap-3" aria-label={i18nT('components.antigravityAccountModal.checking_quota')}>
      <div className="skeleton h-16 w-full rounded-lg" />
      <div className="skeleton h-16 w-full rounded-lg" />
      <div className="skeleton h-10 w-full rounded-lg" />
    </div>
  )
}

function QuotaPoolCard({ pool }: { pool: AntigravityQuotaPool }) {
  const pctStr = fmtPercent(pool.remainingFraction)
  const resetStr = formatResetTimestamp(pool.resetTime)
  const widthPct = Math.min(Math.max(pool.remainingFraction * 100, 0), 100)
  const windowText = poolWindowLabel(pool.window)

  return (
    <div className="rounded-lg border border-border bg-bg-elevated/40 p-3">
      <div className="flex items-baseline justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-[13px] font-semibold text-text-strong">
            {poolTitleNode(pool)}
          </div>
          <div className="text-[11px] text-muted">
            {windowText}
          </div>
        </div>
        <span className={`shrink-0 font-mono text-[15px] font-bold tabular-nums ${pctToneClass(pool.remainingFraction)}`}>
          {i18nT('components.antigravityAccountModal.remaining_label', { pct: pctStr })}
        </span>
      </div>
      <div
        role="progressbar"
        aria-label={windowText}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pool.remainingPct}
        className="mt-2.5 h-2 w-full overflow-hidden rounded-full bg-border"
      >
        <div
          className={`h-full rounded-full ${barFillClass(pool.remainingFraction)}`}
          style={{ width: `${widthPct}%` }}
        />
      </div>
      {resetStr && (
        <div className="mt-1.5 text-right text-[11px] text-muted">
          {i18nT('components.antigravityAccountModal.resets_at', { time: resetStr })}
        </div>
      )}
    </div>
  )
}

function ModelsBreakdown({ models }: { models: AntigravityModelQuota[] }) {
  const [expanded, setExpanded] = useState(false)
  if (models.length === 0) return null

  return (
    <section className="rounded-lg border border-border">
      <button
        type="button"
        onClick={() => setExpanded(v => !v)}
        aria-expanded={expanded}
        className="flex w-full cursor-pointer items-center justify-between gap-2 bg-transparent px-3 py-2.5 text-left text-[12px] font-medium text-text hover:bg-bg-hover focus-ring-accent rounded-lg"
      >
        <span>
          {i18nT('components.antigravityAccountModal.models_breakdown_title', {
            count: fmtNumber(models.length),
          })}
        </span>
        {expanded
          ? <ChevronDown className="lucide-inline text-muted shrink-0" aria-hidden="true" />
          : <ChevronRight className="lucide-inline text-muted shrink-0" aria-hidden="true" />}
      </button>
      {expanded && (
        <div className="max-h-56 divide-y divide-border overflow-y-auto border-t border-border px-3">
          {models.map(model => {
            const pctStr = fmtPercent(model.remainingFraction)
            const resetStr = formatResetTimestamp(model.resetTime)
            const widthPct = Math.min(Math.max(model.remainingFraction * 100, 0), 100)
            return (
              <div key={model.id} className="py-2">
                <div className="flex items-baseline justify-between gap-3">
                  <span
                    data-i18n-opaque
                    className="min-w-0 truncate text-[12px] font-medium text-text"
                    title={model.displayName}
                  >
                    {model.displayName}
                  </span>
                  <span className={`shrink-0 font-mono text-[12px] font-semibold tabular-nums ${pctToneClass(model.remainingFraction)}`}>
                    {pctStr}
                  </span>
                </div>
                <div className="mt-1 flex items-center gap-2">
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-border">
                    <div
                      className={`h-full rounded-full ${barFillClass(model.remainingFraction)}`}
                      style={{ width: `${widthPct}%` }}
                    />
                  </div>
                  {resetStr && (
                    <span className="shrink-0 text-[10px] text-muted">
                      {resetStr}
                    </span>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}

function QuotaBody({ usage, onClose }: { usage: AntigravityUsageState; onClose: () => void }) {
  const refresh = useAntigravityQuotaRefresh()
  const refreshButton = (compact: boolean) => (
    <RefreshButton
      pending={refresh.mutation.isPending}
      held={refresh.held}
      onClick={() => refresh.mutation.mutate()}
      compact={compact}
    />
  )
  const refreshNotice = refresh.notice === null
    ? null
    : <ErrorNotice message={refreshNoticeMessage(refresh.notice)} askAgent onHandoff={onClose} />

  if (usage === null) return <QuotaSkeleton />

  if (usage === 'signin-required') {
    return (
      <ErrorNotice
        message={i18nT('components.antigravityAccountModal.quota_signin_required')}
        askAgent
        onHandoff={onClose}
      />
    )
  }

  if (usage === 'api-key') {
    return (
      <div className="flex items-start gap-2 rounded-lg border border-border bg-bg-elevated/40 p-3.5 text-[13px] text-muted">
        <Info className="lucide-inline shrink-0 mt-0.5" aria-hidden="true" />
        <span>{i18nT('components.antigravityAccountModal.quota_api_key_notice')}</span>
      </div>
    )
  }

  if (usage === 'failed') {
    const outcomeReplacesNotice = refresh.notice !== null
    return (
      <div className="flex flex-col gap-3">
        {outcomeReplacesNotice
          ? refreshNotice
          : (
            <ErrorNotice
              message={i18nT('components.antigravityAccountModal.quota_unavailable')}
              askAgent
              onHandoff={onClose}
            />
          )}
        {refreshButton(false)}
      </div>
    )
  }

  if (!isAntigravityUsageReading(usage) || (usage.pools.length === 0 && usage.models.length === 0)) {
    const outcomeReplacesNotice = refresh.notice !== null
    return (
      <div className="flex flex-col gap-3">
        {outcomeReplacesNotice
          ? refreshNotice
          : (
            <div className="flex items-center gap-2 rounded-lg border border-border bg-bg-elevated/40 p-3.5 text-[13px] text-muted">
              <AlertCircle className="lucide-inline shrink-0" />
              <span>{i18nT('components.antigravityAccountModal.quota_no_reading')}</span>
            </div>
          )}
        {refreshButton(false)}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      {usage.pools.map(pool => (
        <QuotaPoolCard key={pool.id} pool={pool} />
      ))}
      <ModelsBreakdown models={usage.models} />
      <div className="flex items-center justify-between gap-3 text-[12px] text-muted">
        <span>
          {refresh.checkedAt
            ? i18nT('components.antigravityAccountModal.checked_at', { time: fmtTime(refresh.checkedAt) })
            : usage.stale && refresh.notice?.kind !== 'stale'
              ? i18nT('components.antigravityAccountModal.reading_may_be_stale')
              : usage.fetchedAt > 0
                ? i18nT('components.antigravityAccountModal.checked_at', { time: fmtTime(usage.fetchedAt) })
                : null}
        </span>
        {refreshButton(true)}
      </div>
      {refreshNotice}
      <p className="text-[11px] leading-relaxed text-muted">
        {i18nT('components.antigravityAccountModal.quota_scope')}
      </p>
    </div>
  )
}

export interface AntigravityAccountModalProps {
  open: boolean
  onClose: () => void
  usage: AntigravityUsageState
}

export default function AntigravityAccountModal({
  open,
  onClose,
  usage,
}: AntigravityAccountModalProps) {
  const upgradeUri = isAntigravityUsageReading(usage) ? usage.upgradeUri : null
  return (
    <Modal
      open={open}
      onClose={onClose}
      title={(
        <span className="flex items-center gap-2">
          <Sparkles className="lucide-inline text-accent" />
          {i18nT('components.antigravityAccountModal.title')}
        </span>
      )}
      maxWidth={480}
    >
      <div className="flex flex-col gap-4">
        <AccountIdentity usage={usage} />
        <QuotaBody usage={usage} onClose={onClose} />
        {upgradeUri && (
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            <a
              href={upgradeUri}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-[12px] text-accent hover:underline"
            >
              {i18nT('components.antigravityAccountModal.manage_subscription')}
              <ExternalLink className="lucide-inline" />
            </a>
          </div>
        )}
      </div>
    </Modal>
  )
}
