from sim.qecc.kasai import KasaiCode, Kasai_Code, all_schedules
import numpy as np

tmp = Kasai_Code(
    12, 3, 768,
    (763, 679, 397, 61, 697, 373),
    (435, 69, 330, 18, 612, 246),
    (289, 257, 625, 41, 193, 449),
    (496, 640, 200, 524, 672, 672),
    # Hx=Hx,
    # Hz=Hz
)

andrei = KasaiCode(
    L=12,
    J=3,
    P=768,
    f_params=[(763, 435), (679, 69), (397, 330), (61, 18), (697, 612), (373, 246)],
    g_params=[(289, 496), (257, 640), (625, 200), (41, 524), (193, 672), (449, 672)]
)

MAX_SCHEDULES = 5
scheds = all_schedules(andrei, max_schedules=MAX_SCHEDULES)
print(f"Found {len(scheds)} valid schedules.")

assert np.array_equal(tmp.Hx, andrei.Hx), "Hx does not match"
assert np.array_equal(tmp.Hz, andrei.Hz), "Hz does not match"