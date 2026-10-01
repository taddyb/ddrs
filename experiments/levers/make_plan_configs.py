"""Configs for `ddrs plan` summed-Q' baselines over the training years WY1986-1995 (testing window 1985/10/01..1995/09/30)
for the two daily Q' stores not replayed (dHBV2 lumped AORC, HydroDL LSTM AORC). The baseline is routing-free, so the
checkpoint does not matter; the config is the store's 2026-09-13 run snapshot with only the testing window changed.
The replays of UH (s42), DIST and LSTM write the same baseline for their windows as a side effect of `ddrs run`.
"""
import pathlib

RUNS = pathlib.Path("/home/tbindas/projects/ddrs/.ddrs/runs")
HERE = pathlib.Path(__file__).resolve().parent
for name, rid in {"lumped": "2026-09-13T13-56-50Z-train-and-test", "hydrodl": "2026-09-13T17-21-58Z-train-and-test",
                  "uh": "2026-09-27T07-29-47Z-train-and-test"}.items():
    txt = (RUNS / rid / "config.yaml").read_text()
    old_t = "testing:\n  start_time: 1995/10/01\n  end_time: 2010/09/30\n"
    assert txt.count(old_t) == 1, name
    txt = txt.replace(old_t, "testing:\n  start_time: 1985/10/01\n  end_time: 1995/09/30\n", 1)
    (HERE / f"plan_{name}_train10.yaml").write_text(f"# summed-Q' baseline over WY1986-1995 for {rid}'s store\n" + txt)
    print("wrote", name)
