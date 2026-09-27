from melbourne_mate.retrieval.encoders import encoder_params


def test_dense_candidates_use_fixed_revisions():
    mini = encoder_params("all-minilm")
    mpnet = encoder_params("multi-qa-mpnet")

    assert mini.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert mini.revision == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert mpnet.model == "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    assert mpnet.revision == "d51b22a1dfa8184e9258074e56e2875e50612dca"


def test_unknown_dense_candidate_is_rejected():
    try:
        encoder_params("not-a-model")
    except ValueError as exc:
        assert "unknown encoder" in str(exc)
    else:
        raise AssertionError("unknown encoder should fail")
