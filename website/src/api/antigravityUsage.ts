export interface AntigravityQuotaPool {
  id: string
  group: string
  label: string
  window: string
  remainingFraction: number
  remainingPct: number
  resetTime: string | null
}

export interface AntigravityModelQuota {
  id: string
  displayName: string
  remainingFraction: number
  remainingPct: number
  resetTime: string | null
}

export interface AntigravityUsageReading {
  available: true
  authType: string
  email: string | null
  tierId: string | null
  tierName: string | null
  tierDescription: string | null
  upgradeUri: string | null
  pools: AntigravityQuotaPool[]
  models: AntigravityModelQuota[]
  fetchedAt: number
  stale: boolean
}

export type AntigravityUsageState =
  | AntigravityUsageReading
  | null
  | 'none'
  | 'failed'
  | 'signin-required'
  | 'api-key'

export interface AntigravityUsageEnvelope {
  usage?: Record<string, unknown> | null
}

function toFiniteFraction(val: unknown): number | null {
  if (typeof val !== 'number' || !Number.isFinite(val)) return null
  return Math.max(0, Math.min(1, val))
}

export function isAntigravityUsageReading(
  state: AntigravityUsageState,
): state is AntigravityUsageReading {
  return typeof state === 'object' && state !== null && state.available === true
}

export function parseAntigravityUsagePayload(payload: unknown): AntigravityUsageState {
  if (!payload || typeof payload !== 'object') return 'none'
  const raw = (payload as AntigravityUsageEnvelope).usage
  if (raw === null || raw === undefined) return null
  if (typeof raw !== 'object') return 'none'

  if (raw.available !== true) {
    const authState = typeof raw.auth_state === 'string' ? raw.auth_state : ''
    if (authState === 'signin_required') return 'signin-required'
    if (authState === 'api_key') return 'api-key'
    return 'none'
  }

  const rawPools = Array.isArray(raw.pools) ? raw.pools : []
  const pools: AntigravityQuotaPool[] = []
  for (const item of rawPools) {
    if (!item || typeof item !== 'object') continue
    const rec = item as Record<string, unknown>
    const frac = toFiniteFraction(rec.remaining_fraction)
    if (frac === null) continue
    const pct = typeof rec.remaining_pct === 'number' && Number.isFinite(rec.remaining_pct)
      ? Math.max(0, Math.min(100, Math.round(rec.remaining_pct)))
      : Math.round(frac * 100)
    pools.push({
      id: typeof rec.id === 'string' && rec.id ? rec.id : `pool-${pools.length}`,
      group: typeof rec.group === 'string' ? rec.group : '',
      label: typeof rec.label === 'string' ? rec.label : '',
      window: typeof rec.window === 'string' && rec.window ? rec.window : 'rolling',
      remainingFraction: frac,
      remainingPct: pct,
      resetTime: typeof rec.reset_time === 'string' && rec.reset_time ? rec.reset_time : null,
    })
  }

  const rawModels = Array.isArray(raw.models) ? raw.models : []
  const models: AntigravityModelQuota[] = []
  for (const item of rawModels) {
    if (!item || typeof item !== 'object') continue
    const rec = item as Record<string, unknown>
    const frac = toFiniteFraction(rec.remaining_fraction)
    if (frac === null || typeof rec.id !== 'string' || !rec.id) continue
    const pct = typeof rec.remaining_pct === 'number' && Number.isFinite(rec.remaining_pct)
      ? Math.max(0, Math.min(100, Math.round(rec.remaining_pct)))
      : Math.round(frac * 100)
    models.push({
      id: rec.id,
      displayName: typeof rec.display_name === 'string' && rec.display_name ? rec.display_name : rec.id,
      remainingFraction: frac,
      remainingPct: pct,
      resetTime: typeof rec.reset_time === 'string' && rec.reset_time ? rec.reset_time : null,
    })
  }

  return {
    available: true,
    authType: typeof raw.auth_type === 'string' ? raw.auth_type : 'oauth-personal',
    email: typeof raw.email === 'string' && raw.email ? raw.email : null,
    tierId: typeof raw.tier_id === 'string' && raw.tier_id ? raw.tier_id : null,
    tierName: typeof raw.tier_name === 'string' && raw.tier_name ? raw.tier_name : null,
    tierDescription: typeof raw.tier_description === 'string' && raw.tier_description ? raw.tier_description : null,
    upgradeUri: typeof raw.upgrade_uri === 'string' && raw.upgrade_uri ? raw.upgrade_uri : null,
    pools,
    models,
    fetchedAt: typeof raw.fetched_at === 'number' && Number.isFinite(raw.fetched_at) ? raw.fetched_at : 0,
    stale: raw.stale === true,
  }
}

export function antigravityUsageRefetchInterval(
  state: AntigravityUsageState,
  isError: boolean,
  updates: number,
): number | false {
  if (state === 'signin-required' || state === 'api-key') return false
  if (state === null || (isError && updates < 3)) return 3_000
  return 60_000
}

export function pickHeadlinePools(pools: AntigravityQuotaPool[]): {
  gemini: AntigravityQuotaPool | null
  thirdParty: AntigravityQuotaPool | null
  fallback: AntigravityQuotaPool | null
} {
  const gemini = pools.find(p => p.group.toLowerCase().includes('gemini')) ?? null
  const thirdPartyCandidates = pools.filter(p => !p.group.toLowerCase().includes('gemini'))
  const thirdParty =
    thirdPartyCandidates.find(p => p.window === '5h') ??
    thirdPartyCandidates[0] ??
    null
  return {
    gemini,
    thirdParty,
    fallback: pools[0] ?? null,
  }
}
