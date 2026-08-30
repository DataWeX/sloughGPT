"""Meaningful tests for CitationTracker — claim extraction, citation, formatting."""

from domains.cognitive.rag import CitationTracker, TextChunk


class TestExtractClaims:
    def test_extract_is_claim(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Paris is a city.")
        assert len(claims) == 1
        assert claims[0]["subject"] == "Paris"
        assert claims[0]["predicate"] == "a city"

    def test_extract_was_claim(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python was created in 1991.")
        assert len(claims) == 1
        assert claims[0]["subject"] == "Python"

    def test_extract_can_claim(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Dogs can swim.")
        assert len(claims) == 1
        assert claims[0]["subject"] == "Dogs"

    def test_extract_has_claim(self):
        ct = CitationTracker()
        claims = ct.extract_claims("France has 67 million people.")
        assert len(claims) == 1

    def test_extract_multiple_claims(self):
        ct = CitationTracker()
        text = "Paris is a city. Berlin is also a city."
        claims = ct.extract_claims(text)
        assert len(claims) == 2

    def test_extract_no_claims(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Hello world!")
        assert len(claims) == 0

    def test_extract_cross_sentence_not_captured(self):
        ct = CitationTracker()
        text = "I like Paris. Is a beautiful place."
        claims = ct.extract_claims(text)
        assert len(claims) <= 1

    def test_extract_positions(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Paris is great.")
        assert claims[0]["start"] >= 0
        assert claims[0]["end"] > claims[0]["start"]

    def test_extract_text_field_matches(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Tokyo is a city.")
        assert "Tokyo is a city" in claims[0]["text"]

    def test_extract_predicate_trailing_period_stripped(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Alice is big.")
        assert not claims[0]["predicate"].endswith(".")

    def test_extract_empty_text(self):
        ct = CitationTracker()
        claims = ct.extract_claims("")
        assert claims == []

    def test_extract_no_match_without_capitalized_subject(self):
        ct = CitationTracker()
        claims = ct.extract_claims("dogs can swim.")
        assert len(claims) == 0

    def test_extract_multiple_sentences_mixed(self):
        ct = CitationTracker()
        text = "Run fast. Alice is tall. Nothing here."
        claims = ct.extract_claims(text)
        assert len(claims) == 1
        assert claims[0]["subject"] == "Alice"

    def test_extract_claim_with_compound_name(self):
        ct = CitationTracker()
        claims = ct.extract_claims("New York is a city.")
        assert claims[0]["subject"] == "New York"

    def test_extract_positions_are_absolute(self):
        ct = CitationTracker()
        text = "Alpha is big. Beta is tall."
        claims = ct.extract_claims(text)
        assert len(claims) == 2
        assert claims[1]["start"] > claims[0]["start"]

    def test_extract_was_claim_without_created(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python was released.")
        assert len(claims) == 1
        assert claims[0]["predicate"] == "released"

    def test_extract_can_claim_with_verb(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Birds can fly.")
        assert len(claims) == 1
        assert claims[0]["subject"] == "Birds"
        assert claims[0]["predicate"] == "fly"

    def test_extract_has_claim_with_quantity(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Mars has two moons.")
        assert len(claims) == 1
        assert claims[0]["subject"] == "Mars"
        assert "two moons" in claims[0]["predicate"]

    def test_extract_multiple_patterns_in_sentence(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Alice has a cat and Alice can swim.")
        assert len(claims) >= 1

    def test_extract_two_compound_names(self):
        ct = CitationTracker()
        text = "New York is big. Los Angeles is big."
        claims = ct.extract_claims(text)
        assert len(claims) == 2
        assert claims[0]["subject"] == "New York"
        assert claims[1]["subject"] == "Los Angeles"

    def test_extract_claim_with_extra_spaces(self):
        ct = CitationTracker()
        claims = ct.extract_claims("  Alice   is   tall  today.  ")
        assert len(claims) == 1
        assert claims[0]["subject"] == "Alice"

    def test_extract_question_mark_terminates_sentence(self):
        ct = CitationTracker()
        text = "Is Paris a city? Berlin is a city."
        claims = ct.extract_claims(text)
        assert any(c["subject"] == "Berlin" for c in claims)

    def test_extract_exclamation_terminates_sentence(self):
        ct = CitationTracker()
        text = "Run fast! Alice is tall."
        claims = ct.extract_claims(text)
        assert any(c["subject"] == "Alice" for c in claims)

    def test_extract_multiple_is_claims(self):
        ct = CitationTracker()
        text = "Alpha is big. Beta is tall. Gamma is fast."
        claims = ct.extract_claims(text)
        assert len(claims) == 3

    def test_extract_end_position_greater_than_start(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Paris is a city.")
        for c in claims:
            assert c["end"] > c["start"]


class TestCite:
    def test_cite_with_sources(self):
        ct = CitationTracker()
        claim = {"subject": "Paris", "predicate": "a city", "text": "Paris is a city."}
        chunk = TextChunk(
            id="c1", content="Paris is the capital of France",
            metadata={"source": "wiki"}, embedding=None
        )
        cited = ct.cite(claim, [chunk])
        assert cited["supported"] is True
        assert len(cited["sources"]) == 1
        assert cited["sources"][0]["chunk_id"] == "c1"

    def test_cite_no_sources(self):
        ct = CitationTracker()
        claim = {"subject": "X", "predicate": "y", "text": "X y."}
        cited = ct.cite(claim, [])
        assert cited["supported"] is False
        assert cited["sources"] == []

    def test_cite_max_3_sources(self):
        ct = CitationTracker()
        claim = {"subject": "X", "predicate": "y", "text": "X y."}
        chunks = [
            TextChunk(id=f"c{i}", content=f"content{i}", metadata={}, embedding=None)
            for i in range(5)
        ]
        cited = ct.cite(claim, chunks)
        assert len(cited["sources"]) == 3

    def test_cite_preserves_claim_fields(self):
        ct = CitationTracker()
        claim = {"subject": "A", "predicate": "b", "text": "A b.", "extra": 42}
        cited = ct.cite(claim, [])
        assert cited["subject"] == "A"
        assert cited["predicate"] == "b"
        assert cited["text"] == "A b."
        assert cited["extra"] == 42

    def test_cite_source_content_truncated_to_200(self):
        ct = CitationTracker()
        claim = {"subject": "X", "predicate": "y", "text": "X y."}
        chunk = TextChunk(id="c1", content="x" * 500, metadata={}, embedding=None)
        cited = ct.cite(claim, [chunk])
        assert len(cited["sources"][0]["content"]) == 200

    def test_cite_source_metadata_included(self):
        ct = CitationTracker()
        claim = {"subject": "X", "predicate": "y", "text": "X y."}
        chunk = TextChunk(id="c1", content="data", metadata={"source": "book", "page": 5}, embedding=None)
        cited = ct.cite(claim, [chunk])
        assert cited["sources"][0]["metadata"]["source"] == "book"
        assert cited["sources"][0]["metadata"]["page"] == 5

    def test_cite_from_two_chunks(self):
        ct = CitationTracker()
        claim = {"subject": "X", "predicate": "y", "text": "X y."}
        chunks = [
            TextChunk(id="c1", content="a", metadata={}, embedding=None),
            TextChunk(id="c2", content="b", metadata={}, embedding=None),
        ]
        cited = ct.cite(claim, chunks)
        assert len(cited["sources"]) == 2

    def test_cite_supported_true_when_chunks_exist(self):
        ct = CitationTracker()
        claim = {"subject": "X", "predicate": "y", "text": "X y."}
        chunk = TextChunk(id="c1", content="data", metadata={}, embedding=None)
        cited = ct.cite(claim, [chunk])
        assert cited["supported"] is True

    def test_cite_content_short_not_truncated(self):
        ct = CitationTracker()
        claim = {"subject": "X", "predicate": "y", "text": "X y."}
        chunk = TextChunk(id="c1", content="short", metadata={}, embedding=None)
        cited = ct.cite(claim, [chunk])
        assert cited["sources"][0]["content"] == "short"

    def test_cite_exactly_3_chunks(self):
        ct = CitationTracker()
        claim = {"subject": "X", "predicate": "y", "text": "X y."}
        chunks = [
            TextChunk(id=f"c{i}", content=f"content{i}", metadata={}, embedding=None)
            for i in range(3)
        ]
        cited = ct.cite(claim, chunks)
        assert len(cited["sources"]) == 3

    def test_cite_metadata_missing_source_key(self):
        ct = CitationTracker()
        claim = {"subject": "X", "predicate": "y", "text": "X y."}
        chunk = TextChunk(id="c1", content="data", metadata={"author": "test"}, embedding=None)
        cited = ct.cite(claim, [chunk])
        assert "author" in cited["sources"][0]["metadata"]


class TestFormatCitations:
    def test_format_empty(self):
        ct = CitationTracker()
        assert ct.format_citations() == ""

    def test_format_with_claims(self):
        ct = CitationTracker()
        ct.claims = [
            {"text": "Paris is a city", "sources": [{"metadata": {"source": "wiki"}}]},
            {"text": "Berlin is a city", "sources": []},
        ]
        output = ct.format_citations()
        assert "[1] Paris is a city" in output
        assert "[2] Berlin is a city" in output
        assert "→ wiki" in output

    def test_format_claim_without_sources(self):
        ct = CitationTracker()
        ct.claims = [{"text": "X is Y", "sources": []}]
        output = ct.format_citations()
        assert "[1] X is Y" in output
        assert "→" not in output

    def test_format_multiple_claims_numbered(self):
        ct = CitationTracker()
        ct.claims = [
            {"text": f"Claim {i}", "sources": []}
            for i in range(5)
        ]
        output = ct.format_citations()
        for i in range(1, 6):
            assert f"[{i}]" in output

    def test_format_unknown_source(self):
        ct = CitationTracker()
        ct.claims = [
            {"text": "X is Y", "sources": [{"metadata": {}}]}
        ]
        output = ct.format_citations()
        assert "→ Unknown" in output

    def test_format_source_with_missing_metadata_key(self):
        ct = CitationTracker()
        ct.claims = [
            {"text": "X is Y", "sources": [{"metadata": {"author": "test"}}]}
        ]
        output = ct.format_citations()
        assert "→ Unknown" in output

    def test_format_preserves_claim_order(self):
        ct = CitationTracker()
        ct.claims = [
            {"text": "First", "sources": []},
            {"text": "Second", "sources": []},
            {"text": "Third", "sources": []},
        ]
        output = ct.format_citations()
        pos_first = output.index("[1] First")
        pos_second = output.index("[2] Second")
        pos_third = output.index("[3] Third")
        assert pos_first < pos_second < pos_third

    def test_format_newline_separated(self):
        ct = CitationTracker()
        ct.claims = [
            {"text": "A", "sources": []},
            {"text": "B", "sources": []},
        ]
        output = ct.format_citations()
        lines = output.strip().split("\n")
        assert len(lines) == 2

    def test_format_single_claim_no_arrow(self):
        ct = CitationTracker()
        ct.claims = [{"text": "Solo claim", "sources": []}]
        output = ct.format_citations()
        assert output == "[1] Solo claim"

    def test_format_claim_with_multiple_sources(self):
        ct = CitationTracker()
        ct.claims = [
            {"text": "X is Y", "sources": [
                {"metadata": {"source": "wiki"}},
                {"metadata": {"source": "book"}},
            ]}
        ]
        output = ct.format_citations()
        assert "→ wiki" in output
        assert "→ book" in output

    def test_format_numbering_starts_at_one(self):
        ct = CitationTracker()
        ct.claims = [{"text": "Only claim", "sources": []}]
        output = ct.format_citations()
        assert output.startswith("[1]")

    def test_format_10_claims(self):
        ct = CitationTracker()
        ct.claims = [{"text": f"Claim {i}", "sources": []} for i in range(10)]
        output = ct.format_citations()
        assert "[10]" in output


class TestClaimsStorage:
    def test_extract_claims_stores_internally(self):
        ct = CitationTracker()
        ct.extract_claims("Paris is a city.")
        assert len(ct.claims) == 1

    def test_extract_claims_replaces_previous(self):
        ct = CitationTracker()
        ct.extract_claims("Paris is great. Berlin is fun.")
        ct.extract_claims("Tokyo is a city.")
        assert len(ct.claims) == 1
        assert ct.claims[0]["subject"] == "Tokyo"

    def test_claims_empty_by_default(self):
        ct = CitationTracker()
        assert ct.claims == []

    def test_claims_list_type(self):
        ct = CitationTracker()
        assert isinstance(ct.claims, list)

    def test_extract_claims_returns_same_as_stored(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Paris is a city.")
        assert ct.claims is claims

    def test_extract_claims_stores_subject_predicate_text_keys(self):
        ct = CitationTracker()
        ct.extract_claims("Paris is a city.")
        for claim in ct.claims:
            assert "subject" in claim
            assert "predicate" in claim
            assert "text" in claim

    def test_extract_claims_stores_positions(self):
        ct = CitationTracker()
        ct.extract_claims("Paris is a city.")
        for claim in ct.claims:
            assert "start" in claim
            assert "end" in claim
            assert isinstance(claim["start"], int)
            assert isinstance(claim["end"], int)
