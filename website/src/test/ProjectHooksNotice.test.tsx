import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const setProjectHooksConsent = vi.fn()
vi.mock('../api/client', () => ({ api: { setProjectHooksConsent: (...a: unknown[]) => setProjectHooksConsent(...a) } }))

import ProjectHooksNotice from '../pages/chat/ProjectHooksNotice'
import { isProjectHooksNotice } from '../pages/chat/projectHooksNoticeMatch'
import type { ChatMessage } from '../types'

const row = (state: string, folder = '/work/repo'): ChatMessage =>
  ({
    role: 'notice',
    content: 'ℹ Project hooks are running because you trusted this folder in kiro-cli.',
    meta: {
      kind: 'spec_hooks',
      project_hooks: state,
      project_hooks_folder: folder,
      project_hooks_commands: [{ event: 'PreToolUse', when: 'before a tool runs', command: 'echo guard' }, { bad: 1 }],
    },
  }) as unknown as ChatMessage

describe('ProjectHooksNotice', () => {
  beforeEach(() => setProjectHooksConsent.mockReset())
  afterEach(() => setProjectHooksConsent.mockReset())

  it('matches only its own tags', () => {
    expect(isProjectHooksNotice(row('kiro_cli'))).toBe(true)
    expect(isProjectHooksNotice(row('other'))).toBe(false)
    expect(isProjectHooksNotice(row('off', ''))).toBe(false)
    expect(isProjectHooksNotice({ ...row('off'), role: 'assistant' } as ChatMessage)).toBe(false)
  })

  it('turns hooks off for the row folder when kiro-cli started them', async () => {
    setProjectHooksConsent.mockResolvedValue({ ok: true, enabled: false })
    render(<ProjectHooksNotice message={row('kiro_cli')} askAgent={false} />)
    fireEvent.click(screen.getByTestId('project-hooks-notice-toggle'))
    await waitFor(() => expect(screen.getByTestId('project-hooks-notice-saved')).toBeTruthy())
    expect(setProjectHooksConsent).toHaveBeenCalledWith('/work/repo', false)
  })

  it('turns hooks on from the off row and shows a refusal', async () => {
    setProjectHooksConsent.mockImplementation(async () => { throw new Error('refused') })
    render(<ProjectHooksNotice message={row('off')} askAgent={false} />)
    fireEvent.click(screen.getByTestId('project-hooks-notice-toggle'))
    await waitFor(() => expect(screen.getByTestId('project-hooks-notice-error').textContent).toContain('Reload the page'))
    expect(screen.getByTestId('project-hooks-notice-error').textContent).not.toContain('refused')
    expect(screen.getByTestId('project-hooks-notice-error-details').textContent).toContain('refused')
    expect(setProjectHooksConsent).toHaveBeenCalledWith('/work/repo', true)
  })

  it('lists each hook event and command, read-only, behind Show commands', () => {
    render(<ProjectHooksNotice message={row('off')} askAgent={false} />)
    const list = screen.getByTestId('project-hooks-notice-commands')
    expect(list.tagName).toBe('DETAILS')
    expect(list.textContent).toContain('before a tool runs')
    expect(list.textContent).toContain('echo guard')
    expect(list.querySelectorAll('li')).toHaveLength(1)
  })
})
