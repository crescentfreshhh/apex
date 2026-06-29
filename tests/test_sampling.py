from peaks.sampling import plan_timestamps


def test_plan_basic_spacing():
    ts = plan_timestamps(duration=10.0, interval=2.0)
    # starts half an interval in, steps by interval, stays < duration
    assert ts == [1.0, 3.0, 5.0, 7.0, 9.0]


def test_plan_custom_offset():
    assert plan_timestamps(10.0, 5.0, offset=0.0) == [0.0, 5.0]


def test_plan_short_clip():
    # duration shorter than the first offset -> no samples
    assert plan_timestamps(0.5, 2.0) == []


def test_plan_zero_or_negative():
    assert plan_timestamps(0.0, 2.0) == []
    assert plan_timestamps(10.0, 0.0) == []
    assert plan_timestamps(-5.0, 2.0) == []


def test_plan_all_within_duration():
    ts = plan_timestamps(100.0, 3.0)
    assert all(0 <= t < 100.0 for t in ts)
    assert ts == sorted(ts)
