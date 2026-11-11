import { describe, expect, it } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, fireEvent } from '@testing-library/react'
import {
  antigravityUsageRefetchInterval,
  isAntigravityUsageReading,
  parseAntigravityUsagePayload,
  pickHeadlinePools,
} from '../api/antigravityUsage'
import AntigravityAccountModal from '../components/AntigravityAccountModal'
import { antigravityUsageSegment } from '../shell/topbar/antigravityUsageReadout'

describe('antigravityUsage', () => {
  it('parses OAuth quota payloads and classifies auth sentinels', () => {
    expect(parseAntigravityUsagePayload({ usage: null })).toBeNull()
    expect(parseAntigravityUsagePayload({ usage: { available: false, auth_state: 'signin_required' } })).toBe('signin-required')
    expect(parseAntigravityUsagePayload({ usage: { available: false, auth_state: 'api_key' } })).toBe('api-key')

    const parsed = parseAntigravityUsagePayload({
      usage: {
        available: true,
        auth_type: 'oauth-personal',
        email: 'dev@example.com',
        tier_id: 'g1-ultra-tier',
        tier_name: 'Google AI Ultra',
        pools: [
          {
            id: 'gemini-models:quota-remaining',
            group: 'Gemini Models',
            label: '5h Limit Remaining',
            window: '5h',
            remaining_fraction: 0.7,
            remaining_pct: 70,
            reset_time: '2026-11-11T09:44:23Z',
          },
          {
            id: 'claude-and-gpt-models:five-hour-limit-remaining',
            group: 'Claude and GPT models',
            label: 'Five Hour Limit Remaining',
            window: '5h',
            remaining_fraction: 0.88,
            remaining_pct: 88,
            reset_time: '2026-11-11T10:02:50Z',
          },
        ],
        models: [
          {
            id: 'gemini-3.1-pro-high',
            display_name: 'Gemini 3.1 Pro (High)',
            remaining_fraction: 0.7,
            remaining_pct: 70,
            reset_time: '2026-11-11T09:44:23Z',
          },
        ],
        fetched_at: 1700000000,
        stale: false,
      },
    })

    expect(isAntigravityUsageReading(parsed)).toBe(true)
    if (!isAntigravityUsageReading(parsed)) return
    expect(parsed.tierName).toBe('Google AI Ultra')
    expect(parsed.pools).toHaveLength(2)
    expect(parsed.models).toHaveLength(1)

    const headlines = pickHeadlinePools(parsed.pools)
    expect(headlines.gemini?.remainingPct).toBe(70)
    expect(headlines.thirdParty?.remainingPct).toBe(88)
    expect(antigravityUsageRefetchInterval(parsed, false, 1)).toBe(60_000)
    expect(antigravityUsageRefetchInterval('signin-required', false, 1)).toBe(false)
  })

  it('renders topbar pill and account modal with remaining percentages and model breakdown', () => {
    const reading = parseAntigravityUsagePayload({
      usage: {
        available: true,
        email: 'dev@example.com',
        tier_name: 'Google AI Ultra',
        pools: [
          {
            id: 'g',
            group: 'Gemini Models',
            label: '5h Limit Remaining',
            window: '5h',
            remaining_fraction: 0.7,
            remaining_pct: 70,
            reset_time: '2026-11-11T09:44:23Z',
          },
          {
            id: 'c',
            group: 'Claude and GPT models',
            label: 'Five Hour Limit Remaining',
            window: '5h',
            remaining_fraction: 0.88,
            remaining_pct: 88,
            reset_time: '2026-11-11T10:02:50Z',
          },
        ],
        models: [
          {
            id: 'gemini-3.1-pro-high',
            display_name: 'Gemini 3.1 Pro (High)',
            remaining_fraction: 0.7,
            remaining_pct: 70,
            reset_time: '2026-11-11T09:44:23Z',
          },
        ],
        fetched_at: 1700000000,
        stale: false,
      },
    })

    expect(antigravityUsageSegment({ open: false, setOpen: () => {}, state: reading, hidden: true }, 'seg', false)).toBeNull()

    const pill = antigravityUsageSegment(
      { open: false, setOpen: () => {}, state: reading, hidden: false },
      'seg',
      false,
    )
    const qc = new QueryClient()
    render(
      <QueryClientProvider client={qc}>
        <div>{pill}</div>
        <AntigravityAccountModal open onClose={() => {}} usage={reading} />
      </QueryClientProvider>,
    )

    expect(screen.getByTestId('antigravity-usage-pill')).toHaveTextContent('70%')
    expect(screen.getByTestId('antigravity-usage-pill')).toHaveTextContent('88%')
    expect(screen.getByText('Google AI Ultra')).toBeInTheDocument()

    // Expand per-model breakdown
    const breakdownBtn = screen.getByRole('button', { expanded: false })
    fireEvent.click(breakdownBtn)
    expect(screen.getByText('Gemini 3.1 Pro (High)')).toBeInTheDocument()
  })
})
