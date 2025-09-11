import vapoursynth as vs
import havsfunc as haf

core = vs.core

from typing import List

def average(clip: vs.VideoNode, weights: List[float], divisor: float = None):
    def get_offset_clip(offset: int) -> vs.VideoNode:
        if offset > 0:
            return clip[offset:] + clip[-1] * offset
        elif offset < 0:
            return clip[0] * -offset + clip[:offset]
        else:
            return clip

    diameter = len(weights)
    radius = diameter // 2

    if divisor is None:
        divisor = sum(weights)

    assert diameter % 2 == 1, "An odd number of weights is required."

    clips = [get_offset_clip(offset) for offset in range(-radius, radius + 1)]

    expr = ""
    # expr_vars = "xyzabcdefghijklmnopqrstuvw"
    expr_vars = []
    for i in range(0, 1024): expr_vars += [f"src{i}"]

    for var, weight in zip(expr_vars[:diameter], weights):
        expr += f"{var} {weight} * "

    expr += "+ " * (diameter - 1)
    expr += f"{divisor} /" if divisor != 1 else ""
    # https://github.com/AkarinVS/vapoursynth-plugin
    clip = core.akarin.Expr(clips, expr)

    return clip

clip = core.std.BlankClip(width=1902, height=1080, fpsnum=60)

video = average(clip, [0.25,0.5,0.25])



video.set_output()                                                                                                                                                       