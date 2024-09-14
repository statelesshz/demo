from enum import Enum


class InputAndOutputType(Enum):
    TEXT = 1
    IMAGE = 2
    VIDEO = 3
    AUDIO = 4
    EMBEDDING = 5

    # 分类
    CLASS = 6
    # 序列标注
    CLASS_LIST = 7
    # 回归
    NUMBER = 8
    # 检测框架
    BOXES = 9
    common = -1