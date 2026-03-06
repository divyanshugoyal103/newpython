const STORAGE_KEY = 'sessions';

/**
 * Storage abstraction for session data.
 *
 * All Chrome storage usage is centralized here so that replacing
 * local storage with a cloud-backed provider can be done in one place.
 */
export const sessionStorage = {
  /**
   * Save a new session entry.
   * @param {Object} session
   * @returns {Promise<Object>} The saved session.
   */
  async saveSession(session) {
    const sessions = await this.getSessions();
    sessions.push(session);
    await chrome.storage.local.set({ [STORAGE_KEY]: sessions });
    return session;
  },

  /**
   * Read all stored sessions.
   * @returns {Promise<Array<Object>>}
   */
  async getSessions() {
    const result = await chrome.storage.local.get(STORAGE_KEY);
    return Array.isArray(result[STORAGE_KEY]) ? result[STORAGE_KEY] : [];
  },

  /**
   * Update an existing session by id.
   * @param {Object} updatedSession
   * @returns {Promise<Object>} The updated session.
   */
  async updateSession(updatedSession) {
    if (!updatedSession || !updatedSession.id) {
      throw new Error('updateSession requires a session object with an id.');
    }

    const sessions = await this.getSessions();
    const sessionIndex = sessions.findIndex((session) => session.id === updatedSession.id);

    if (sessionIndex === -1) {
      throw new Error(`Session with id "${updatedSession.id}" was not found.`);
    }

    sessions[sessionIndex] = {
      ...sessions[sessionIndex],
      ...updatedSession,
    };

    await chrome.storage.local.set({ [STORAGE_KEY]: sessions });
    return sessions[sessionIndex];
  },

  /**
   * Delete a session by id.
   * @param {string} sessionId
   * @returns {Promise<boolean>} True if a session was deleted.
   */
  async deleteSession(sessionId) {
    const sessions = await this.getSessions();
    const filteredSessions = sessions.filter((session) => session.id !== sessionId);

    if (filteredSessions.length === sessions.length) {
      return false;
    }

    await chrome.storage.local.set({ [STORAGE_KEY]: filteredSessions });
    return true;
  },
};

export default sessionStorage;
