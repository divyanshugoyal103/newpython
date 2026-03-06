/**
 * Internal representation of a browser tab used by tracking/session state.
 * @typedef {Object} TabNode
 * @property {number} id
 * @property {number | null} openerTabId
 * @property {string} url
 * @property {string} status
 * @property {number} createdAt
 * @property {number} updatedAt
 */

/**
 * Convert a Chrome tab object into the internal TabNode model.
 * @param {chrome.tabs.Tab} tab
 * @returns {TabNode}
 */
export function toTabNode(tab) {
  const now = Date.now();

  return {
    id: tab.id,
    openerTabId: typeof tab.openerTabId === 'number' ? tab.openerTabId : null,
    url: tab.url || '',
    status: tab.status || 'loading',
    createdAt: now,
    updatedAt: now,
  };
}
