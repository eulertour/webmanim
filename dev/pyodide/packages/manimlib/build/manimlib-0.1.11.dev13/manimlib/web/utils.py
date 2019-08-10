import js
import time

from manimlib.mobject.svg.tex_mobject import SingleStringTexMobject

def mobject_to_dict(mobject):
    return {
        "points": mobject.points,
        "get_stroke_width_background_true": mobject.get_stroke_width(True),
        "get_stroke_width_background_false": mobject.get_stroke_width(False),
        "get_stroke_rgbas_background_true": mobject.get_stroke_rgbas(True)[0],
        "get_stroke_rgbas_background_false": mobject.get_stroke_rgbas(False)[0],
        "get_fill_rgbas": mobject.get_fill_rgbas()[0],
    }
