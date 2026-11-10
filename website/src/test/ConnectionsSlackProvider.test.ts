/**
 * Slack is a pre-registered Connections provider: its card renders "needs
 * configuration" until a Client ID is saved, and a configured `slack` MCP row at
 * Slack's own endpoint belongs to that card. The MCP Servers table reads the
 * second half (`connectionProviderForServer`) to send the row's sign-in to the
 * card instead of treating it as a server with its own OAuth client.
 */
import { describe, expect, it } from 'vitest'
import type { McpServer } from '../types'
import {
  CONNECTION_PROVIDERS,
  connectionProviderForServer,
  isPreregisteredProvider,
} from '../pages/connections/registry'

function row(name: string, url: string): McpServer {
  return { name, url, clientId: '123.456', status: 'needs_auth', source: 'mcp.json', enabled: true } as McpServer
}

describe('Slack Connections provider', () => {
  it('shows a card as a public pre-registered client', () => {
    const slack = CONNECTION_PROVIDERS.find(p => p.slug === 'slack')
    expect(slack).toBeDefined()
    expect(isPreregisteredProvider(slack!)).toBe(true)
    expect(slack!.auth?.confidential).toBe(false)
    expect(slack!.auth?.redirect_host).toBe('localhost')
  })

  it('owns a slack row at its own endpoint, with or without a trailing slash', () => {
    expect(connectionProviderForServer(row('slack', 'https://mcp.slack.com/mcp'))?.slug).toBe('slack')
    expect(connectionProviderForServer(row('slack', 'https://mcp.slack.com/mcp/'))?.slug).toBe('slack')
  })

  it('does not own a row at another endpoint or under another name', () => {
    expect(connectionProviderForServer(row('slack', 'https://slack.example.com/mcp'))).toBeUndefined()
    expect(connectionProviderForServer(row('my-slack', 'https://mcp.slack.com/mcp'))).toBeUndefined()
  })
})
