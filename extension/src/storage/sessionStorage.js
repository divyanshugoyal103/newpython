const ACTIVE_SESSION_KEY = 'activeSession';

/**
 * @typedef {{
 *   id: string,
 *   tabs: Record<number, import('../models/tabNode.js').TabNode>
 * }} ActiveSession
 */

/**
 * Read current active session from chrome storage.
 * @returns {Promise<ActiveSession | null>}
 */
export async function getActiveSession() {
  const stored = await chrome.storage.local.get(ACTIVE_SESSION_KEY);
  return stored[ACTIVE_SESSION_KEY] || null;
}

/**
 * Persist active session to chrome storage.
 * @param {ActiveSession} session
 * @returns {Promise<void>}
 */
export async function setActiveSession(session) {
  await chrome.storage.local.set({ [ACTIVE_SESSION_KEY]: session });
}

/**
 * Create and persist a new active session.
 * @returns {Promise<ActiveSession>}
 */
export async function initializeActiveSession() {
  const session = {
    id: crypto.randomUUID(),
    tabs: {},
  };

  await setActiveSession(session);
  return session;
}
