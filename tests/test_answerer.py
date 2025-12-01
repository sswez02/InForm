from src.answerer import answer_query
from src.indexer import TfIdfIndex
from src.models import Study, Passage


def test_answer_returns_citations():
    studies = [
        Study(
            id=1,
            title="Test",
            authors="A B",
            year=2020,
            doi=None,
            journal=None,
            rating=4.0,
            tags=[],
        )
    ]
    passages = [
        Passage(
            id=1, study_id=1, section="abstract", text="Creatine increases strength."
        ),
    ]

    idx = TfIdfIndex()
    idx.add_passages(passages)
    idx.build()

    ans = answer_query(
        query="creatine strength",
        index=idx,
        studies=studies,
    )

    assert "Creatine increases strength" in ans.answer_text
    assert len(ans.references) == 1
    assert ans.references[0]["index"] == 1
