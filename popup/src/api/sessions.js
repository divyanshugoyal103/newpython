/**
 * Fetch session list from extension runtime or fallback demo data.
 */
export async function fetchSessions() {
  if (typeof chrome !== 'undefined' && chrome.runtime?.sendMessage) {
    return new Promise((resolve, reject) => {
      chrome.runtime.sendMessage({ type: 'GET_SESSIONS' }, (response) => {
        if (chrome.runtime.lastError) {
          reject(chrome.runtime.lastError);
          return;
        }
        resolve(Array.isArray(response?.sessions) ? response.sessions : []);
      });
    });
  }

  return [
    { id: '1', title: 'Docs', path: 'Work/Research', lastAccessed: Date.now() },
    { id: '2', title: 'Email', path: 'Work', lastAccessed: Date.now() - 1000 },
    { id: '3', title: 'Video', path: 'Personal/Fun', lastAccessed: Date.now() - 5000 }
  ];
}
