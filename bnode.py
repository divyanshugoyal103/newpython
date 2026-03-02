"""Pure hierarchical tree engine.

`bNode` is a side-effect free data structure for building and exporting
hierarchical trees. It does not rely on browser/chrome APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class bNode:
    """Immutable tree node for hierarchical structures.

    Attributes:
        name: Human-readable node identifier.
        value: Optional payload for the node.
        children: Child nodes in insertion order.
    """

    name: str
    value: Optional[Any] = None
    children: List["bNode"] = field(default_factory=list)

    def add_child(self, child: "bNode") -> "bNode":
        """Return a new node with `child` appended.

        This method never mutates the current node.
        """
        return bNode(name=self.name, value=self.value, children=[*self.children, child])

    def extend(self, children: Iterable["bNode"]) -> "bNode":
        """Return a new node with multiple children appended."""
        return bNode(name=self.name, value=self.value, children=[*self.children, *children])

    def to_dict(self) -> Dict[str, Any]:
        """Convert the tree into a hierarchical dictionary structure."""
        return {
            "name": self.name,
            "value": self.value,
            "children": [child.to_dict() for child in self.children],
        }

    def to_lines(self, indent: str = "  ") -> List[str]:
        """Render the tree as indented text lines.

        Returns:
            List[str]: A pre-order textual representation.
        """

        lines: List[str] = []

        def _walk(node: "bNode", depth: int) -> None:
            prefix = indent * depth
            label = f"{node.name}" if node.value is None else f"{node.name}: {node.value}"
            lines.append(f"{prefix}{label}")
            for c in node.children:
                _walk(c, depth + 1)

        _walk(self, 0)
        return lines

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "bNode":
        """Build a tree from a dictionary in `to_dict` format."""
        children = [bNode.from_dict(item) for item in data.get("children", [])]
        return bNode(name=data["name"], value=data.get("value"), children=children)
