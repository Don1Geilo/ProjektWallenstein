import pytest
import duckdb

from wallenstein.aliases import add_alias, alias_map


def _wordnet_available() -> bool:
    try:
        from nltk.corpus import wordnet  # type: ignore

        wordnet.synsets("car")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _wordnet_available(), reason="WordNet not available")
def test_alias_map_expands_synonyms(tmp_path):
    con = duckdb.connect(database=":memory:")
    add_alias(con, "AAA", "car")
    amap = alias_map(con, use_synonyms=True)
    assert "AAA" in amap
    assert any(syn in amap["AAA"] for syn in {"auto", "automobile"})
