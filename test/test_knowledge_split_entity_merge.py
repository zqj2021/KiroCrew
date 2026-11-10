"""``merge_split_entities``: one node per real-world entity once an alias bridges two."""

from __future__ import annotations

import pytest
from knowledge_helpers import store_fixture  # noqa: F401

from kiro_crew.knowledge.entity_resolution import merge_split_entities
from kiro_crew.knowledge.store import MAX_ENTITY_ALIASES, KnowledgeStore

# Full alias lists: the cap may have dropped one that names a third entity.
_CAPPED_KEEPER = ["Music Bank", *(f"뮤직뱅크 {n}" for n in range(1, MAX_ENTITY_ALIASES))]
_CAPPED_TARGET = [f"Music Bank {n}" for n in range(MAX_ENTITY_ALIASES)]


def _mentioned(store, name, entity_type="program", aliases=None):
    """An entity mentioned by one item of its own, as an ingest leaves it."""
    eid = store.add_entity(name, entity_type, aliases=aliases)
    store.add_mention(store.add_item(f"doc about {name}", "content", "document"), eid)
    return eid


def _entity_ids(store):
    return {row[0] for row in store.db.execute("SELECT id FROM entities")}


def test_a_split_pair_becomes_one_entity_with_both_mentions_and_relations(store):
    korean = _mentioned(store, "뮤직뱅크")
    english = _mentioned(store, "Music Bank")
    kbs = store.add_entity("한국방송공사", "org")
    seoul = store.add_entity("Seoul", "place")
    store.add_entity_relation(korean, kbs, "part_of")
    store.add_entity_relation(english, seoul, "uses")
    # A later ingest of a Korean document attaches the English alias.
    store.add_entity_aliases(korean, ["Music Bank"])

    assert merge_split_entities(store) == 1

    assert _entity_ids(store) == {korean, kbs, seoul}
    mentions = store.db.execute(
        "SELECT COUNT(*) FROM mentions WHERE entity_id = ?", (korean,)
    ).fetchone()[0]
    targets = {
        row[0]
        for row in store.db.execute(
            "SELECT target_id FROM entity_relations WHERE source_id = ?", (korean,)
        )
    }
    assert (mentions, targets) == (2, {kbs, seoul})
    assert store.find_entity("Music Bank")["id"] == korean
    assert merge_split_entities(store) == 0


def test_a_cased_non_latin_target_pairs_by_casefold(store):
    english = _mentioned(store, "Moscow", "place", aliases=["Москва"])
    _mentioned(store, "Москва", "place")
    assert merge_split_entities(store) == 1
    assert _entity_ids(store) == {english}


def test_a_merged_entitys_aliases_never_bridge_a_later_merge(store):
    korean = _mentioned(store, "뮤직뱅크", aliases=["Music Bank"])
    _mentioned(store, "Music Bank", aliases=["Music Bank Show"])
    show = _mentioned(store, "Music Bank Show")
    assert merge_split_entities(store) == 1
    assert merge_split_entities(store) == 0
    assert _entity_ids(store) == {korean, show}


def test_nothing_is_paired_before_the_alias_fill_completes(store):
    _mentioned(store, "뮤직뱅크", aliases=["Music Bank"])
    _mentioned(store, "Music Bank")
    store.db.execute("DROP INDEX idx_entity_aliases_key")
    assert merge_split_entities(store) == 0
    assert len(_entity_ids(store)) == 2


@pytest.mark.parametrize(
    "entities",
    [
        # Alias-to-alias overlap: neither canonical name is the shared alias.
        [("뮤직뱅크", "program", ["Music Bank"]), ("음악방송", "program", ["Music Bank"])],
        [("뮤직뱅크", "program", ["Music Bank"]), ("Music Bank", "org", None)],
        # Two entities claim the same canonical: no unique target.
        [
            ("뮤직뱅크", "program", ["Music Bank"]),
            ("ミュージックバンク", "program", ["Music Bank"]),
            ("Music Bank", "program", None),
        ],
        # One entity's aliases name two canonicals: no unique target.
        [
            ("뮤직뱅크", "program", ["Music Bank", "ミュージックバンク"]),
            ("Music Bank", "program", None),
            ("ミュージックバンク", "program", None),
        ],
        # A chain: the middle entity has a partner on each side.
        [
            ("뮤직뱅크", "program", ["Music Bank"]),
            ("Music Bank", "program", ["ミュージックバンク"]),
            ("ミュージックバンク", "program", None),
        ],
        # A short Latin abbreviation is lookup-only.
        [("방탄소년단", "group", ["BTS"]), ("BTS", "group", None)],
        # A full alias list proves nothing unique, on either side.
        [("뮤직뱅크", "program", _CAPPED_KEEPER), ("Music Bank", "program", None)],
        [("뮤직뱅크", "program", ["Music Bank"]), ("Music Bank", "program", _CAPPED_TARGET)],
        # ...and still counts as a second claimant.
        [
            ("뮤직뱅크", "program", _CAPPED_KEEPER),
            ("ミュージックバンク", "program", ["Music Bank"]),
            ("Music Bank", "program", None),
        ],
    ],
    ids=[
        "alias-to-alias",
        "type-mismatch",
        "two-claimants",
        "two-targets",
        "chain",
        "abbreviation",
        "capped-keeper",
        "capped-target",
        "capped-claimant",
    ],
)
def test_pairs_the_ingestion_rule_rejects_stay_split(store, entities):
    ids = {_mentioned(store, name, entity_type, aliases) for name, entity_type, aliases in entities}
    assert merge_split_entities(store) == 0
    assert _entity_ids(store) == ids


def test_a_mutual_pair_merges_once_and_keeps_the_older_entity(store, caplog):
    korean = _mentioned(store, "뮤직뱅크", aliases=["Music Bank"])
    _mentioned(store, "Music Bank", aliases=["뮤직뱅크"])
    with caplog.at_level("WARNING", logger="kiro_crew.knowledge.entity_resolution"):
        assert merge_split_entities(store) == 1
    assert _entity_ids(store) == {korean}
    assert [r.message for r in caplog.records if r.levelname == "WARNING"] == []


def test_one_failing_pair_does_not_stop_the_others(store, monkeypatch):
    first = _mentioned(store, "뮤직뱅크", aliases=["Music Bank"])
    first_target = _mentioned(store, "Music Bank")
    second = _mentioned(store, "인기가요", aliases=["Inkigayo Show"])
    _mentioned(store, "Inkigayo Show")
    original = KnowledgeStore.merge_entities

    def failing_first(self, keep_id, merge_id):
        if keep_id == first:
            raise RuntimeError("database is locked")
        return original(self, keep_id, merge_id)

    monkeypatch.setattr(KnowledgeStore, "merge_entities", failing_first)
    assert merge_split_entities(store) == 1
    assert _entity_ids(store) == {first, first_target, second}
