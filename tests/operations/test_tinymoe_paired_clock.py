"""The primary paired clock includes each backend's update preparation."""
from megakernel_lab.tinymoe import benchmark


def test_native_stage_is_inside_complete_update_clock(monkeypatch):
    events=[]
    times=iter((7.0,7.25))
    def clock():
        events.append("clock")
        return next(times)
    monkeypatch.setattr(benchmark.time,"perf_counter",clock)
    monkeypatch.setattr(benchmark.C,"synchronize",lambda device:events.append("sync:"+device))
    result={"loss":"result"}
    def stage(): events.append("stage")
    def update():
        events.append("update")
        return result
    actual,seconds=benchmark.complete_update_wall(update,stage=stage)
    assert actual is result and seconds==.25
    assert events==["sync:cuda","clock","stage","update","sync:cuda","clock"]


def test_conventional_preparation_has_the_same_clock_boundary(monkeypatch):
    events=[]
    times=iter((9.0,9.5))
    def clock():
        events.append("clock")
        return next(times)
    monkeypatch.setattr(benchmark.time,"perf_counter",clock)
    monkeypatch.setattr(benchmark.C,"synchronize",lambda device:events.append("sync:"+device))
    def update():
        events.extend(("schedule","forward/backward","clip/optimizer/ema"))
        return 11
    actual,seconds=benchmark.complete_update_wall(update)
    assert actual==11 and seconds==.5
    assert events==["sync:cuda","clock","schedule","forward/backward","clip/optimizer/ema","sync:cuda","clock"]
