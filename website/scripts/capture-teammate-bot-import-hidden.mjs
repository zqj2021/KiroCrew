/**
 * Screenshot harness: a team Crew teammate on the Bots page, who keeps New bot
 * and finds no Import from file, since an import card writes rules, skills and
 * schedules only the Crew owner may.
 *
 *   node scripts/capture-teammate-bot-import-hidden.mjs <outDir> <distDir>
 *
 * Runs the REAL built SPA behind `serveDist` (127.0.0.1, ephemeral port) with
 * every `/api/**` answered from fixtures (`stubDashboardApi`). The roster is the
 * narrowed teammate row `GET /api/members` sends a relay teammate, `/api/auth/me`
 * answers `owner_ok: false`, and the Shared Crew preview is off (a relay teammate
 * is admitted either way). The `empty` scene is the empty-roster hero. Every frame asserts what it
 * photographs before writing the PNG.
 */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'
import { join, resolve } from 'node:path'

import { json } from './lib/boot-api.mjs'
import { serveDist, DEFAULT_DIST } from './lib/serve-dist.mjs'
import { stubDashboardApi, logPageProblems } from './lib/stub-dashboard-api.mjs'

const OUT = process.argv[2] || join(process.env.KIROCREW_SCRATCH || '/tmp', 'teammate-bot-import-hidden')
const DIST = process.argv[3] ? resolve(process.argv[3]) : DEFAULT_DIST
mkdirSync(OUT, { recursive: true })

const teammateRow = (name, display, description) => ({
  name, slug: name, display_name: display, description, avatar: {}, source: 'kirocrew', dashboard_created: true,
})
const MEMBERS = [
  teammateRow('reviewer', 'Reviewer', 'Reviews pull requests'),
  teammateRow('oncall', 'Oncall', 'Watches the pager'),
]
const extraFor = (members, meStatus = 200) => async (path, route) => {
  if (path === '/api/members') { await json(route, { members }); return true }
  if (path === '/api/auth/me' && meStatus !== 200) { await json(route, { error: 'unavailable' }, meStatus); return true }
  if (path === '/api/auth/me') { await json(route, { user: 'relay:kiro-user.ana', owner_ok: false, token_accepted: false, session_exp: 0, refresh_exp: 0 }); return true }
  if (path === '/api/teams') { await json(route, { teams: [] }); return true }
  if (path === '/api/crons') { await json(route, { jobs: [] }); return true }
  if (path === '/api/autonudge') { await json(route, { enabled: true, loops: [] }); return true }
  return false
}

const { srv, base } = await serveDist(DIST)
const browser = await chromium.launch()
let failed = false
function check(name, ok, detail = '') {
  console.log(`${name}: ${ok ? 'OK' : 'MISMATCH'} ${detail}`)
  if (!ok) failed = true
}

async function scene(theme) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 2, colorScheme: theme })
  const page = await context.newPage()
  logPageProblems(page)
  await stubDashboardApi(page, {
    theme, extra: extraFor(MEMBERS),
    localStorageEntries: { 'mc-lang': 'en', 'mc-crewmates-onboarded': '1', 'mc-crewmates-page-entered': '1', 'mc-nav': '1' },
  })
  await page.goto(`${base}/members?member=reviewer`, { waitUntil: 'domcontentloaded' })
  const chip = page.getByTestId('crewmate-switcher')
  await chip.waitFor({ state: 'visible', timeout: 30000 })
  await page.waitForTimeout(1500)
  // The switcher: the other Bots and New bot, with no Import from file.
  await chip.click()
  await page.getByTestId('crewmate-switcher-list').waitFor({ state: 'visible', timeout: 10000 })
  check(`[${theme}] switcher offers New bot`, (await page.getByTestId('crewmate-switcher-create').count()) === 1)
  check(`[${theme}] switcher offers no Import from file`, (await page.getByTestId('crewmate-switcher-import').count()) === 0)
  await page.waitForTimeout(300)
  await page.screenshot({ path: join(OUT, `teammate-switcher-${theme}.png`), clip: { x: 0, y: 0, width: 900, height: 520 } })
  // The pinned roster's "+" menu: New bot live, no Import.
  await page.getByTestId('crewmate-switcher-roster').click()
  const add = page.getByTestId('member-add')
  await add.waitFor({ state: 'visible', timeout: 10000 })
  await add.click()
  const menu = page.getByTestId('member-add-menu')
  await menu.waitFor({ state: 'visible', timeout: 10000 })
  await menu.getByTestId('member-add-crewmate').waitFor({ state: 'visible', timeout: 10000 })
  check(`[${theme}] New bot is live`, (await menu.getByTestId('member-add-crewmate-held').count()) === 0)
  check(`[${theme}] Import from file absent`, (await menu.getByTestId('member-add-import').count()) === 0)
  await page.waitForTimeout(300)
  await page.screenshot({ path: join(OUT, `teammate-add-menu-${theme}.png`), clip: { x: 0, y: 0, width: 900, height: 520 } })
  await context.close()
}

async function emptyScene(theme) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 2, colorScheme: theme })
  const page = await context.newPage()
  logPageProblems(page)
  await stubDashboardApi(page, {
    theme, extra: extraFor([]),
    localStorageEntries: { 'mc-lang': 'en', 'mc-crewmates-onboarded': '1', 'mc-crewmates-page-entered': '1', 'mc-nav': '1' },
  })
  await page.goto(`${base}/members`, { waitUntil: 'domcontentloaded' })
  const hero = page.getByTestId('crewmate-empty-hero').last()
  await hero.waitFor({ state: 'visible', timeout: 30000 })
  await page.waitForTimeout(1500)
  check(`[empty/${theme}] hero offers New bot`, (await hero.getByTestId('crewmate-empty-cta').count()) === 1)
  check(`[empty/${theme}] hero offers no Import from file`, (await page.getByTestId('crewmate-empty-import').count()) === 0)
  await page.screenshot({ path: join(OUT, `teammate-empty-hero-${theme}.png`) })
  await context.close()
}

/** `/api/auth/me` answers 503: the roster says the owner check failed and Import is held. */
async function failedCheckScene(theme) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 2, colorScheme: theme })
  const page = await context.newPage()
  logPageProblems(page)
  await stubDashboardApi(page, {
    theme, extra: extraFor(MEMBERS, 503),
    localStorageEntries: { 'mc-lang': 'en', 'mc-crewmates-onboarded': '1', 'mc-crewmates-page-entered': '1', 'mc-nav': '1' },
  })
  await page.goto(`${base}/members?member=reviewer`, { waitUntil: 'domcontentloaded' })
  await page.getByTestId('crewmate-switcher').click({ timeout: 30000 })
  await page.getByTestId('crewmate-switcher-roster').click()
  const notice = page.getByTestId('member-roster-owner-check-error')
  await notice.waitFor({ state: 'visible', timeout: 10000 })
  check(`[failed/${theme}] roster says the owner check failed`, (await notice.innerText()).includes("Couldn't check whether you can import bots here"))
  await page.getByTestId('member-add').click()
  const menu = page.getByTestId('member-add-menu')
  await menu.waitFor({ state: 'visible', timeout: 10000 })
  check(`[failed/${theme}] Import from file held`, (await menu.getByTestId('member-add-import').count()) === 0)
  await page.waitForTimeout(300)
  await page.screenshot({ path: join(OUT, `owner-check-failed-${theme}.png`), clip: { x: 0, y: 0, width: 900, height: 520 } })
  await context.close()
}

try {
  await failedCheckScene('dark')
  await failedCheckScene('light')
  await scene('dark')
  await scene('light')
  await emptyScene('dark')
  await emptyScene('light')
} finally {
  await browser.close()
  srv.close()
}
if (failed) process.exit(1)
console.log(`wrote ${OUT}`)
