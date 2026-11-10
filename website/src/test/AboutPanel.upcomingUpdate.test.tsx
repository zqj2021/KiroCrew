// The advance notice of the next update, inline under About > Updates.
//
// Contract under test:
// - no newer build and no notice -> nothing is drawn, not even an empty card
// - an announced version -> version, expected date, up to three highlights, the
//   notes link and the takes-effect-on-restart line, labelled "Announced"
// - an offered or staged build -> the same block titled "Available update", with no
//   pill, pointing at the update control above to install it
// - it is a plain section inside the Updates card: never a dialog, and it adds
//   no button of its own (the card's Update or Restart control stays the only one)
// - it promises no time: the line says the update takes effect when it is restarted
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, act, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Provider } from 'react-redux'
import { MemoryRouter } from 'react-router-dom'
import { store } from '../store'
import { sseStatus } from '../store/dashboardSlice'
import { AboutPanel } from '../pages/settings/AboutPanel'
import { AboutPanelUpcomingUpdate } from '../pages/settings/AboutPanelUpcomingUpdate'
import { highlightsFromNotes, safeNotesUrl, upcomingUpdate } from '../pages/settings/upcomingUpdate'
import { i18nT } from '../i18n/t'

const BLANK_STATUS = {
  uptime: '1m', sessions: 0, messages: 0, cron_jobs: 0, subagents: 0, lessons: 0,
} as const

const ANNOUNCED = {
  state: 'announced' as const,
  version: '0.13.0',
  date: '2026-11-18',
  highlights: ['Faster startup', 'Clearer doctor output', 'Fewer prompts'],
  notes_url: 'https://github.com/kirodotdev/KiroCrew/releases/tag/v0.13.0',
}

function publish(upcoming: unknown) {
  act(() => { store.dispatch(sseStatus({ ...BLANK_STATUS, update_upcoming: upcoming } as never)) })
}

function mount(ui: React.ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <Provider store={store}>
      <QueryClientProvider client={qc}>
        <MemoryRouter>{ui}</MemoryRouter>
      </QueryClientProvider>
    </Provider>,
  )
}

describe('AboutPanelUpcomingUpdate', () => {
  beforeEach(() => {
    delete (window as unknown as { updateAPI?: unknown }).updateAPI
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true, status: 200, json: async () => ({}), text: async () => '{}', headers: new Headers({ 'content-type': 'application/json' }),
    })))
  })
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    store.dispatch(sseStatus({ ...BLANK_STATUS } as never))
  })

  it('none: draws nothing when no newer version is announced or offered', () => {
    publish(null)
    mount(<AboutPanelUpcomingUpdate updateState={null} />)
    expect(screen.queryByTestId('upcoming-update')).toBeNull()
  })

  it('none: an older gateway that sends no field draws nothing either', () => {
    store.dispatch(sseStatus({ ...BLANK_STATUS } as never))
    mount(<AboutPanelUpcomingUpdate updateState={null} />)
    expect(screen.queryByTestId('upcoming-update')).toBeNull()
  })

  it('announced: version, expected date, three highlights, notes link and the restart line', () => {
    publish(ANNOUNCED)
    mount(<AboutPanelUpcomingUpdate updateState={null} />)
    const block = screen.getByTestId('upcoming-update')
    expect(block.getAttribute('data-state')).toBe('announced')
    expect(within(block).getByText('Upcoming update')).toBeTruthy()
    expect(screen.getByTestId('upcoming-update-meta').textContent).toBe('Version 0.13.0 · Expected Nov 18, 2026')
    expect(screen.getByTestId('upcoming-update-state').textContent).toBe('Announced')
    const items = within(screen.getByTestId('upcoming-update-highlights')).getAllByRole('listitem')
    expect(items.map(li => li.textContent)).toEqual(ANNOUNCED.highlights)
    const link = screen.getByTestId('upcoming-update-notes') as HTMLAnchorElement
    expect(link.getAttribute('href')).toBe(ANNOUNCED.notes_url)
    expect(link.getAttribute('rel')).toContain('noopener')
    expect(screen.getByTestId('upcoming-update-takes-effect').textContent)
      .toBe('Once it is released, it takes effect the next time you restart Kiro Crew.')
    // A fact, not a control: no button of its own.
    expect(within(block).queryByRole('button')).toBeNull()
  })

  it('ready: a build the feed offers says it is available and points at the update control', () => {
    publish({ ...ANNOUNCED, state: 'ready' })
    mount(<AboutPanelUpcomingUpdate updateState={null} />)
    expect(screen.getByTestId('upcoming-update').getAttribute('data-state')).toBe('ready')
    expect(screen.getByTestId('upcoming-update-meta').textContent).toBe('Version 0.13.0 · Released Nov 18, 2026')
    // Titled as an available update with no state pill, and the line points at the
    // card's own control instead of promising a restart does the install.
    const block = screen.getByTestId('upcoming-update')
    expect(within(block).getByText('Available update')).toBeTruthy()
    expect(within(block).queryByText('Upcoming update')).toBeNull()
    expect(screen.queryByTestId('upcoming-update-state')).toBeNull()
    expect(screen.getByTestId('upcoming-update-takes-effect').textContent)
      .toBe('Install it with the update control above.')
  })

  it('ready: beside the Update now control the heading and line do not read as one click being not enough', async () => {
    store.dispatch(sseStatus({
      ...BLANK_STATUS,
      update_upcoming: { ...ANNOUNCED, state: 'ready' },
      update_available: true,
      update_check_status: 'succeeded',
      update_latest_version: '0.13.0',
      update_latest_version_display: '0.13.0',
    } as never))
    mount(<AboutPanel />)
    const block = await screen.findByTestId('upcoming-update')
    expect(await screen.findByText('Update available')).toBeTruthy()
    expect(screen.getAllByRole('button', { name: /Update now/ }).length).toBeGreaterThan(0)
    expect(within(block).getByText('Available update')).toBeTruthy()
    expect(within(block).queryByText('Upcoming update')).toBeNull()
    expect(block.textContent).not.toContain('not from the button')
    expect(block.textContent).not.toContain('restart the gateway')
  })

  it('ready: a build the desktop updater has staged, with no gateway record at all', () => {
    publish(null)
    mount(<AboutPanelUpcomingUpdate updateState={{
      state: 'downloaded', version: '0.13.0', pubDate: '2026-11-18T09:00:00Z', notes: '- One\n- Two',
    }} />)
    expect(screen.getByTestId('upcoming-update').getAttribute('data-state')).toBe('ready')
    expect(screen.getByTestId('upcoming-update-meta').textContent).toBe('Version 0.13.0 · Released Nov 18, 2026')
    expect(within(screen.getByTestId('upcoming-update-highlights')).getAllByRole('listitem')).toHaveLength(2)
    expect(screen.queryByTestId('upcoming-update-notes')).toBeNull()
  })

  it('draws a link only for an https notes URL and omits unknown parts', () => {
    publish({ ...ANNOUNCED, date: '', highlights: [], notes_url: 'javascript:alert(1)' })
    mount(<AboutPanelUpcomingUpdate updateState={null} />)
    expect(screen.getByTestId('upcoming-update-meta').textContent).toBe('Version 0.13.0')
    expect(screen.queryByTestId('upcoming-update-highlights')).toBeNull()
    expect(screen.queryByTestId('upcoming-update-notes')).toBeNull()
  })

  it('has a catalog string for every sentence it shows', () => {
    for (const key of ['title', 'title_ready', 'announced', 'highlights', 'notes_link']) {
      const value = i18nT(`pages.settings.aboutPanel.upcoming_update_${key}`)
      expect(value).not.toContain('upcoming_update_')
    }
  })
})

describe('Upcoming update inside the About page', () => {
  beforeEach(() => {
    delete (window as unknown as { updateAPI?: unknown }).updateAPI
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true, status: 200, json: async () => ({}), text: async () => '{}', headers: new Headers({ 'content-type': 'application/json' }),
    })))
  })
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    store.dispatch(sseStatus({ ...BLANK_STATUS } as never))
  })

  it('sits inside the Updates card and is never a dialog', async () => {
    publish(ANNOUNCED)
    mount(<AboutPanel />)
    const block = await screen.findByTestId('upcoming-update')
    expect(block.closest('[role="dialog"], [role="alertdialog"], [aria-modal="true"]')).toBeNull()
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(screen.queryByRole('alertdialog')).toBeNull()
    const updatesTitle = screen.getByText('Updates')
    const card = updatesTitle.closest('div[class*="card"], section, div')!.parentElement!
    expect(card.contains(block)).toBe(true)
  })

  it('draws nothing in the page when there is no notice', async () => {
    publish(null)
    mount(<AboutPanel />)
    await screen.findByText('Updates')
    expect(screen.queryByTestId('upcoming-update')).toBeNull()
  })
})

describe('upcomingUpdate view model', () => {
  it('prefers the staged desktop build over a gateway notice for another version', () => {
    const got = upcomingUpdate(ANNOUNCED, { state: 'found', version: '0.12.9', pubDate: '2026-11-12T00:00:00Z' })
    expect(got).toMatchObject({ state: 'ready', version: '0.12.9', date: '2026-11-12', notesUrl: '' })
  })

  it('keeps the gateway notes when the desktop build is the same version', () => {
    const got = upcomingUpdate(ANNOUNCED, { state: 'downloaded', version: '0.13.0' })
    expect(got).toMatchObject({ state: 'ready', version: '0.13.0', notesUrl: ANNOUNCED.notes_url })
    expect(got?.highlights).toEqual(ANNOUNCED.highlights)
  })

  it('is null when nothing is found or announced', () => {
    expect(upcomingUpdate(null, null)).toBeNull()
    expect(upcomingUpdate(undefined, { state: 'not-available' })).toBeNull()
    expect(upcomingUpdate(undefined, { state: 'checking', version: '0.13.0' })).toBeNull()
  })

  it('takes list items as highlights, three at most, with plain lines as the fallback', () => {
    expect(highlightsFromNotes('## Added\n- **One**\n* `Two`\n- Three\n- Four')).toEqual(['One', 'Two', 'Three'])
    expect(highlightsFromNotes('# Title\nOnly a line')).toEqual(['Only a line'])
    expect(highlightsFromNotes(undefined)).toEqual([])
  })

  it('accepts only https links without whitespace', () => {
    expect(safeNotesUrl('https://example.com/a')).toBe('https://example.com/a')
    expect(safeNotesUrl('http://example.com')).toBe('')
    expect(safeNotesUrl('https://exa mple.com')).toBe('')
    expect(safeNotesUrl('https://')).toBe('')
    expect(safeNotesUrl(undefined)).toBe('')
  })
})
