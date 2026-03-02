from bnode import bNode


def test_hierarchical_structure_dict_output():
    root = bNode("root").extend(
        [
            bNode("branch_a").add_child(bNode("leaf_a1", 10)),
            bNode("branch_b", "ok"),
        ]
    )

    assert root.to_dict() == {
        "name": "root",
        "value": None,
        "children": [
            {
                "name": "branch_a",
                "value": None,
                "children": [
                    {"name": "leaf_a1", "value": 10, "children": []},
                ],
            },
            {"name": "branch_b", "value": "ok", "children": []},
        ],
    }


def test_immutability_no_side_effects():
    root = bNode("root")
    updated = root.add_child(bNode("child"))

    assert root.to_dict() == {"name": "root", "value": None, "children": []}
    assert updated.to_dict() == {
        "name": "root",
        "value": None,
        "children": [{"name": "child", "value": None, "children": []}],
    }


def test_text_tree_output():
    tree = bNode("root").add_child(bNode("child", 1))
    assert tree.to_lines() == ["root", "  child: 1"]


def test_round_trip_from_dict():
    payload = {
        "name": "root",
        "value": "v",
        "children": [{"name": "n1", "value": None, "children": []}],
    }
    assert bNode.from_dict(payload).to_dict() == payload
