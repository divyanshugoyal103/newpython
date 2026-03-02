import {
  getActiveSession,
  initializeActiveSession,
  setActiveSession,
} from '../storage/sessionStorage.js';
import { TabTracker } from './tabTracker.js';

function registerBackgroundListeners() {
  const tabTracker = new TabTracker({
    getActiveSession,
    initializeActiveSession,
    setActiveSession,
  });

  chrome.tabs.onCreated.addListener((tab) => {
    tabTracker.onTabCreated(tab);
  });

  chrome.tabs.onUpdated.addListener((tabId) => {
    tabTracker.onTabUpdated(tabId);
  });
}

registerBackgroundListeners();
