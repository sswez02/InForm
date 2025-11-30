from src.indexer import TfIdfIndex
from src.models import Passage


def test_indexer_simple():
    passages = [
        Passage(
            id=1, study_id=1, section="abstract", text="Creatine increases strength."
        ),
        Passage(id=2, study_id=2, section="abstract", text="Running improves cardio."),
    ]

    idx = TfIdfIndex()
    idx.add_passages(passages)
    idx.build()

    results = idx.search("creatine strength", top_k=1)
    assert len(results) == 1
    top_passage, score = results[0]
    assert top_passage.study_id == 1
    assert score > 0
