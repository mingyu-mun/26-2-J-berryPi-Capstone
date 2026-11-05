from .base import TeacherInput, TeacherModel, sample_weights
from .ensemble import EnsembleTeacher
from .lgbm import LGBMTeacher
from .rule import RuleTeacher

REGISTRY = {"rule": RuleTeacher, "lgbm": LGBMTeacher, "ensemble": EnsembleTeacher}
try:   # torch 가 없으면 B 만 빠진다
    from .rrnet import RRNetTeacher
    REGISTRY["rrnet"] = RRNetTeacher
except ImportError:   # pragma: no cover
    RRNetTeacher = None

__all__ = ["TeacherInput", "TeacherModel", "sample_weights", "REGISTRY", "RuleTeacher", "LGBMTeacher",
           "RRNetTeacher", "EnsembleTeacher"]
