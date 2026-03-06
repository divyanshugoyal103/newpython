import { toTabNode } from '../models/tabNode.js';

/**
 * @typedef {{
 *   getActiveSession: () => Promise<any>,
 *   initializeActiveSession: () => Promise<any>,
 *   setActiveSession: (session: any) => Promise<void>
 * }} SessionStorageApi
 */

/**
 * Encapsulates tab tracking and persistence behavior.
 */
export class TabTracker {
  /**
   * @param {SessionStorageApi} storageApi
   */
  constructor(storageApi) {
    this.storageApi = storageApi;
  }

  /**
   * Track a newly created tab.
   * @param {chrome.tabs.Tab} tab
   * @returns {Promise<void>}
   */
  async onTabCreated(tab) {
    if (typeof tab.id !== 'number') {
      return;
    }

    const tabNode = toTabNode(tab);
    await this.persistTab(tabNode);
  }

  /**
   * Track tab updates and persist the latest state.
   * @param {number} tabId
   * @returns {Promise<void>}
   */
  async onTabUpdated(tabId) {
    const tab = await chrome.tabs.get(tabId);

    if (typeof tab.id !== 'number') {
      return;
    }

    const tabNode = toTabNode(tab);
    await this.persistTab(tabNode);
  }

  /**
   * Upsert tab node into active session and persist.
   * @param {import('../models/tabNode.js').TabNode} tabNode
   * @returns {Promise<void>}
   */
  async persistTab(tabNode) {
    const session =
      (await this.storageApi.getActiveSession()) ||
      (await this.storageApi.initializeActiveSession());

    const existingNode = session.tabs[tabNode.id];
    const mergedNode = {
      ...existingNode,
      ...tabNode,
      createdAt: existingNode?.createdAt || tabNode.createdAt,
      updatedAt: Date.now(),
    };

    session.tabs[tabNode.id] = mergedNode;
    await this.storageApi.setActiveSession(session);
  }
}
