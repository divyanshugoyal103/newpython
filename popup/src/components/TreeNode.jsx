import React, { useState } from 'react';

export function TreeNode({ node }) {
  const [open, setOpen] = useState(true);

  if (node.kind === 'session') {
    return <div className="node">• {node.name}</div>;
  }

  return (
    <div className="node">
      <div className="row">
        <button type="button" onClick={() => setOpen((v) => !v)}>
          {open ? '-' : '+'}
        </button>
        <strong>{node.name}</strong>
      </div>
      {open && (
        <div className="children">
          {node.children.map((child) => (
            <TreeNode key={child.id} node={child} />
          ))}
        </div>
      )}
    </div>
  );
}
