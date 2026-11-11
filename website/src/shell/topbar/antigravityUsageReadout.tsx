import { useEffect, useState, type ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Info, Loader2, Sparkles } from 'lucide-react'
import { api } from '../../api/client'
import { configuredAcpBackend, type AcpBackendConfig } from '../../api/acpBackend'
import {
  antigravityUsageRefetchInterval,
  isAntigravityUsageReading,
  parseAntigravityUsagePayload,
  pickHeadlinePools,
  type AntigravityUsageState,
} from '../../api/antigravityUsage'
import { fmtPercent } from '../../i18n/format'
import { i18nT } from '../../i18n/t'

export interface AntigravityUsageReadout {
  open: boolean
  setOpen: (open: boolean) => void
  state: AntigravityUsageState
  hidden: boolean
}

export function useAntigravityUsageReadout(): AntigravityUsageReadout {
  const [open, setOpen] = useState(false)
  const { data: cfg } = useQuery<AcpBackendConfig>({
    queryKey: ['kirocrewConfig'],
    queryFn: () => api.kirocrewConfig(),
  })
  const onAntigravity = cfg !== undefined && configuredAcpBackend(cfg) === 'antigravity'

  const { data, isError } = useQuery<AntigravityUsageState>({
    queryKey: ['antigravity-usage'],
    queryFn: () => api.antigravityUsage().then(parseAntigravityUsagePayload),
    enabled: onAntigravity,
    refetchInterval: query =>
      antigravityUsageRefetchInterval(
        query.state.data ?? null,
        query.state.status === 'error',
        query.state.dataUpdateCount + query.state.errorUpdateCount,
      ),
  })

  const state: AntigravityUsageState = isError && !data ? 'failed' : (data ?? null)
  const hidden = !onAntigravity

  useEffect(() => {
    if (hidden) setOpen(false)
  }, [hidden])

  return { open, setOpen, state, hidden }
}

export function antigravityUsageSegment(
  { state, hidden, setOpen }: AntigravityUsageReadout,
  seg: string,
  isMobile: boolean,
): ReactNode {
  if (hidden) return null

  if (state === 'api-key') {
    const label = i18nT('app.antigravity_quota_api_key')
    return (
      <button
        key="antigravity-usage"
        type="button"
        data-testid="antigravity-usage-pill"
        className={`${seg} text-muted opacity-70`}
        onClick={() => setOpen(true)}
        title={label}
        aria-label={label}
      >
        <Sparkles size={12} className="text-accent" />
        <span className="font-mono text-[11px] tabular-nums">—</span>
        <Info size={11} aria-hidden="true" />
      </button>
    )
  }

  if (state === 'signin-required') {
    const label = i18nT('app.antigravity_quota_signin_required')
    return (
      <button
        key="antigravity-usage"
        type="button"
        data-testid="antigravity-usage-pill"
        className={`${seg} text-warn`}
        onClick={() => setOpen(true)}
        title={label}
        aria-label={label}
      >
        <Sparkles size={12} />
        <span className="font-mono text-[11px] tabular-nums">—</span>
      </button>
    )
  }

  if (state === 'failed') {
    const label = i18nT('app.antigravity_quota_unavailable')
    return (
      <button
        key="antigravity-usage"
        type="button"
        data-testid="antigravity-usage-pill"
        className={`${seg} text-muted opacity-60`}
        onClick={() => setOpen(true)}
        title={label}
        aria-label={label}
      >
        <Sparkles size={12} />
        <span className="font-mono text-[11px] tabular-nums">—</span>
      </button>
    )
  }

  if (state === null) {
    const label = i18nT('app.antigravity_quota_checking')
    return (
      <button
        key="antigravity-usage"
        type="button"
        data-testid="antigravity-usage-pill"
        className={`${seg} text-muted`}
        onClick={() => setOpen(true)}
        title={label}
        aria-label={label}
      >
        <Sparkles size={12} className="text-accent" />
        {!isMobile && <Loader2 size={11} className="animate-spin" />}
      </button>
    )
  }

  if (!isAntigravityUsageReading(state) || state.pools.length === 0) {
    const label = i18nT('app.antigravity_quota_no_reading')
    return (
      <button
        key="antigravity-usage"
        type="button"
        data-testid="antigravity-usage-pill"
        className={`${seg} text-muted opacity-60`}
        onClick={() => setOpen(true)}
        title={label}
        aria-label={label}
      >
        <Sparkles size={12} />
        <span className="font-mono text-[11px] tabular-nums">—</span>
      </button>
    )
  }

  const { gemini, thirdParty, fallback } = pickHeadlinePools(state.pools)
  const summaryText = gemini && thirdParty
    ? i18nT('components.antigravityAccountModal.pill_summary_pair', {
      gemini: fmtPercent(gemini.remainingFraction),
      claude: fmtPercent(thirdParty.remainingFraction),
    })
    : i18nT('components.antigravityAccountModal.pill_summary_single', {
      pct: fmtPercent((gemini ?? thirdParty ?? fallback)!.remainingFraction),
    })

  const headlineFractions = [gemini?.remainingFraction, thirdParty?.remainingFraction, fallback?.remainingFraction]
    .filter((v): v is number => typeof v === 'number')
  const minFraction = headlineFractions.length > 0 ? Math.min(...headlineFractions) : 1
  const toneClass = minFraction <= 0.05 ? 'text-danger' : minFraction < 0.2 ? 'text-warn' : ''
  const title = i18nT('app.antigravity_quota_title')

  return (
    <button
      key="antigravity-usage"
      type="button"
      data-testid="antigravity-usage-pill"
      className={state.stale ? `${seg} opacity-60` : seg}
      onClick={() => setOpen(true)}
      title={title}
      aria-label={title}
    >
      <Sparkles size={12} className="text-accent" />
      {!isMobile && (
        <span className={`tb-drop-usage font-mono text-[11px] whitespace-nowrap tabular-nums ${toneClass}`}>
          {summaryText}
        </span>
      )}
    </button>
  )
}
