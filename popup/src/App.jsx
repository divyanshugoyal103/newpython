import React, { useEffect, useMemo, useReducer } from 'react';
import { fetchSessions } from './api/sessions.js';
import { buildTree } from './tree/buildTree.js';
import { TreeNode } from './components/TreeNode.jsx';

const initialState = {
  loading: true,
  error: '',
  sessions: []
};

function reducer(state, action) {
  switch (action.type) {
    case 'load-start':
      return { ...state, loading: true, error: '' };
    case 'load-success':
      return { loading: false, error: '', sessions: action.sessions };
    case 'load-error':
      return { loading: false, error: action.error, sessions: [] };
    default:
      return state;
  }
}

export function App() {
  const [state, dispatch] = useReducer(reducer, initialState);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      dispatch({ type: 'load-start' });
      try {
        const sessions = await fetchSessions();
        if (!cancelled) {
          dispatch({ type: 'load-success', sessions });
        }
      } catch (error) {
        if (!cancelled) {
          dispatch({ type: 'load-error', error: error?.message || 'Failed to fetch sessions' });
        }
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const tree = useMemo(() => buildTree(state.sessions), [state.sessions]);

  return (
    <main>
      <h3>Session Tree</h3>
      {state.loading && <p>Loading...</p>}
      {!state.loading && state.error && <p role="alert">Error: {state.error}</p>}
      {!state.loading && !state.error && <TreeNode node={tree} />}
    </main>
  );
}
