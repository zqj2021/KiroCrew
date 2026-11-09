/**
 * Isolated capture entry for the KAS folder-agent hooks notice (PR evidence).
 *
 * The real `ProjectHooksNotice` in each state the gateway sends: `off` (Turn on),
 * `on` (Turn off) and `kiro_cli` (Turn off). The copy is the gateway's own text
 * (`turn_spec_hooks._project_hooks_notice_text`). Clicking a link on this page
 * reaches the capture server, which has no `/api/project-hooks/consent`, so the
 * click shows the failure state with the server's reason. `?theme=light` for light.
 */
import { createRoot } from 'react-dom/client'
import { initI18n } from '../src/i18n/all'
import ProjectHooksNotice from '../src/pages/chat/ProjectHooksNotice'
import type { ChatMessage } from '../src/types'
import '../src/index.css'

const theme = new URLSearchParams(location.search).get('theme') === 'light' ? 'light' : 'dark'
document.documentElement.setAttribute('data-theme', theme)

const SOURCE =
  "This repo's agent file /work/repo/.kiro/agents/repo-guard.json gives agent repo-guard checks that run as commands on this computer before and after its tool calls and around your messages."

const TEXT: Record<string, string> = {
  off: `ℹ Project hooks are off for this folder. ${SOURCE} Open Show commands to read them first.`,
  on: `ℹ Project hooks are running because you turned them on for this folder. ${SOURCE}`,
  kiro_cli: `ℹ Project hooks are running because you trusted this folder in kiro-cli. ${SOURCE} Turn off stops them in Crew only; kiro-cli's trust is unchanged.`,
}

const row = (state: string): ChatMessage =>
  ({
    role: 'notice',
    content: TEXT[state],
    meta: {
      kind: 'spec_hooks',
      project_hooks: state,
      project_hooks_folder: '/work/repo',
      project_hooks_commands: [
        { event: 'PreToolUse', when: 'before a tool runs', command: './scripts/guard.sh' },
        { event: 'PostToolUse', when: 'after a tool runs', command: 'python3 scripts/audit.py' },
      ],
    },
  }) as unknown as ChatMessage

initI18n()
createRoot(document.getElementById('root')!).render(
    <div style={{ padding: 16, maxWidth: 760, display: 'flex', flexDirection: 'column', gap: 12 }}>
      {['off', 'on', 'kiro_cli'].map((s) => (
        <div key={s} data-state={s}>
          <ProjectHooksNotice message={row(s)} askAgent={false} />
        </div>
      ))}
    </div>,
)
