from bedtime import prompts, readability


def test_syllables_rough_but_sane():
    assert readability.count_syllables("cat") == 1
    assert readability.count_syllables("sleepy") == 2
    assert readability.count_syllables("butterfly") == 3


def test_title_is_ignored_in_counts():
    stats = readability.measure("# A Very Long Title Here\n\nThe cat sat. The dog ran.")
    assert stats.words == 6 and stats.sentences == 2


def test_simple_text_is_low_grade_and_complex_text_is_flagged():
    simple = "# T\n\n" + "The cat sat on the mat. " * 60
    hard = "# T\n\n" + (
        "The extraordinarily meticulous cartographer contemplated innumerable unprecedented "
        "geographical irregularities, simultaneously documenting considerable observational "
        "discrepancies throughout the expedition. "
    ) * 20
    assert readability.measure(simple).fk_grade < 2
    assert not any("Reading level" in n for n in readability.check(simple, 6, 360))
    assert any("Reading level" in n for n in readability.check(hard, 6, 360))


def test_length_and_stated_moral_notes():
    notes = readability.check("# T\n\nThe moral of the story is be kind.", 7, 500)
    assert any("words" in n for n in notes)
    assert any("stated lesson" in n for n in notes)


def test_target_words_scales_with_age_and_length():
    assert prompts.target_words(5, "medium") < prompts.target_words(10, "medium")
    assert prompts.target_words(7, "short") < prompts.target_words(7, "long")
    assert prompts.target_words(99, "medium") == prompts.target_words(10, "medium")


def test_stated_lesson_variants_are_caught():
    assert any("stated lesson" in n for n in readability.check("# T\n\nThe monster had learned that friends help.", 7, 10))
    assert not any("stated lesson" in n for n in readability.check("# T\n\nShe learned to whistle.", 7, 10))
