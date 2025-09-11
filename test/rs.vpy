import vapoursynth as vs
import havsfunc as haf

core = vs.core

clip = core.std.BlankClip(width=1902, height=1080, fpsnum=60)

video = core.frameblenderrs.Blend(clip, [0.25,0.5,0.25])

video.set_output()                                                                                                                                                       