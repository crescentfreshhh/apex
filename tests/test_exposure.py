"""The megaboard's exposure record: weighted showings, responses, grades."""

from peaks.exposure import ExposureStore

DAY = 86400.0


def test_passed_over_needs_showings_on_several_days_and_no_response(tmp_path):
    st = ExposureStore(tmp_path / "x.json")
    t0 = 1_760_000_000.0
    st.record({"a": 3, "b": 0.25}, now=t0)
    st.record({"a": 3}, now=t0 + DAY)
    assert not st.summary("a")["passed"]                      # 6 showings, 2 days
    st.record({"a": 2.5}, now=t0 + 2 * DAY)
    s = ExposureStore(tmp_path / "x.json").summary("a")        # persisted
    assert s == {"showings": 8.5, "days": 3, "pos": 0, "neg": 0, "passed": True}
    st.record(neg={"a": 1})
    assert st.summary("a")["passed"]                           # 👎 doesn't clear it
    st.record(pos={"a": 1})
    assert not st.summary("a")["passed"]                       # a save / pivot does
    assert st.summary("b")["showings"] == 0.25 and st.summary("zzz")["showings"] == 0


def test_a_grade_starts_over_and_undo_restores(tmp_path):
    st = ExposureStore(tmp_path / "x.json")
    for d in range(3):
        st.record({"a": 3}, now=1_760_000_000.0 + d * DAY)
    old = st.reset("a")
    assert st.summary("a")["showings"] == 0
    st.restore("a", old)
    assert st.summary("a")["passed"]
    st.drop(["a"])
    assert st.all() == {}


def test_nonsense_is_ignored(tmp_path):
    st = ExposureStore(tmp_path / "x.json")
    assert st.record({"a": -1, "b": 1e6}, pos={"c": 0}) == 0
    (tmp_path / "bad.json").write_text("{not json")
    assert ExposureStore(tmp_path / "bad.json").all() == {}
