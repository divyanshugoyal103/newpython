/**
 * Pure function: converts flat sessions into a nested tree by path.
 * @param {Array<{id:string,title?:string,path?:string,lastAccessed?:number}>} sessions
 */
export function buildTree(sessions) {
  const root = { id: 'root', name: 'Sessions', kind: 'folder', children: [] };
  const folderIndex = new Map([['', root]]);

  for (const session of sessions) {
    const rawPath = typeof session.path === 'string' ? session.path.trim() : '';
    const parts = rawPath ? rawPath.split('/').filter(Boolean) : ['Unsorted'];
    let pathKey = '';

    for (const part of parts) {
      const nextKey = pathKey ? `${pathKey}/${part}` : part;

      if (!folderIndex.has(nextKey)) {
        const folderNode = {
          id: `folder:${nextKey}`,
          name: part,
          kind: 'folder',
          children: []
        };
        folderIndex.set(nextKey, folderNode);
        folderIndex.get(pathKey).children.push(folderNode);
      }

      pathKey = nextKey;
    }

    folderIndex.get(pathKey).children.push({
      id: `session:${session.id}`,
      name: session.title || `Session ${session.id}`,
      kind: 'session',
      session
    });
  }

  return root;
}
