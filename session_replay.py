from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import uuid


@dataclass(frozen=True)
class TabDefinition:
    """Normalized tab structure used for save/replay."""
    tab_id: str
    title: str
    order: int
    parent_id: Optional[str] = None


class SessionReplayManager:
    """Pure session replay logic (UI-agnostic)."""

    def normalize_tabs(self, tabs: List[Dict]) -> List[TabDefinition]:
        normalized: List[TabDefinition] = []
        for idx, tab in enumerate(tabs):
            normalized.append(
                TabDefinition(
                    tab_id=tab["tab_id"],
                    title=tab["title"],
                    order=int(tab.get("order", idx)),
                    parent_id=tab.get("parent_id"),
                )
            )
        return normalized

    def save_session(self, session_name: str, tabs: List[Dict]) -> Dict:
        normalized_tabs = self.normalize_tabs(tabs)
        return {
            "session_id": str(uuid.uuid4()),
            "session_name": session_name,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tabs": [tab.__dict__ for tab in normalized_tabs],
        }

    def replay_order(self, tabs: List[Dict]) -> List[Dict]:
        """Replay plan preserving original order and parent-child hierarchy."""
        normalized = self.normalize_tabs(tabs)
        by_parent: Dict[Optional[str], List[TabDefinition]] = {}
        for tab in normalized:
            by_parent.setdefault(tab.parent_id, []).append(tab)

        for children in by_parent.values():
            children.sort(key=lambda x: x.order)

        replayed: List[Dict] = []

        def walk(parent_id: Optional[str]) -> None:
            for node in by_parent.get(parent_id, []):
                replayed.append(node.__dict__)
                walk(node.tab_id)

        walk(None)
        return replayed

    def replay_tree(self, tabs: List[Dict]) -> List[Dict]:
        """Return roots with recursive children to help UI rendering."""
        normalized = self.normalize_tabs(tabs)
        by_parent: Dict[Optional[str], List[TabDefinition]] = {}
        for tab in normalized:
            by_parent.setdefault(tab.parent_id, []).append(tab)

        for children in by_parent.values():
            children.sort(key=lambda x: x.order)

        def node_to_dict(node: TabDefinition) -> Dict:
            return {
                "tab_id": node.tab_id,
                "title": node.title,
                "order": node.order,
                "parent_id": node.parent_id,
                "children": [node_to_dict(child) for child in by_parent.get(node.tab_id, [])],
            }

        return [node_to_dict(root) for root in by_parent.get(None, [])]
