import mars_router


def test_confidence_plain():
    assert mars_router.parse_confidence("answer\nconfidence: 0.82") == 0.82


def test_confidence_percent():
    assert mars_router.parse_confidence("answer\nconfidence: 82%") == 0.82


def test_confidence_markdown_bold_label():
    text = "answer\n**confidence**: 0.91"
    assert mars_router.parse_confidence(text) == 0.91


def test_confidence_markdown_around_colon():
    text = "answer\nconfidence **:** `0.4`"
    assert mars_router.parse_confidence(text) == 0.4


def test_confidence_missing():
    assert mars_router.parse_confidence("no score here") is None
