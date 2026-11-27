import pydantic
import pytest

from dapo_math import AnswerParams, verify_math_answer


def test_numbers_and_numeric_strings_normalised():
    assert AnswerParams(answer=5).answer == "5"
    assert AnswerParams(answer=-3).answer == "-3"
    assert AnswerParams(answer=5.0).answer == "5"
    assert AnswerParams(answer=2.5).answer == "2.5"
    assert AnswerParams(answer=" +05 ").answer == "5"
    assert AnswerParams(answer="7.").answer == "7"
    assert AnswerParams(answer="+2.50").answer == "2.50"
    assert AnswerParams(answer=".5").answer == ".5"


def test_non_numeric_strings_untouched():
    for s in ["\\frac{1}{2}", "3/4", "sqrt(2)", "(1,2)", "1,234", "x+1", "12 apples", "1e5"]:
        assert AnswerParams(answer=s).answer == s
    assert AnswerParams(answer=" 3/4 ").answer == "3/4"


def test_bool_rejected():
    with pytest.raises(pydantic.ValidationError):
        AnswerParams(answer=True)


def test_numeric_arg_graded_like_string():
    for raw in (12, 12.0, "12", " +12 ", "12."):
        assert verify_math_answer(AnswerParams.model_validate({"answer": raw}).answer, "12")
    assert not verify_math_answer(AnswerParams(answer=13).answer, "12")
