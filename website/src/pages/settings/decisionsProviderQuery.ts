/**
 * Query key for `GET /api/decisions/provider`, shared by the Decisions card and its
 * lazily loaded model picker. It lives outside the picker module so the card can
 * read the provider without pulling the picker into its own chunk.
 */
export const DECISIONS_PROVIDER_QUERY_KEY = ['decisionsProvider'] as const

/** Preset id for "no decision model"; the gateway's `local_models.PRESET_NONE`. */
export const PRESET_NONE = 'none'

/**
 * Whether the hosted default is still configured while the ceiling refuses its host.
 * The gateway reports `active: "none"` for it without rewriting the config, so the
 * provider's own `configured_endpoint` is what tells it from a deliberate "No model".
 */
export function isRefusedHostedDefault(data: {
  active: string
  hosted_permitted?: boolean
  configured_endpoint?: string
}): boolean {
  return (
    data.active === PRESET_NONE &&
    data.hosted_permitted === false &&
    typeof data.configured_endpoint === 'string' &&
    data.configured_endpoint.trim() !== PRESET_NONE
  )
}
