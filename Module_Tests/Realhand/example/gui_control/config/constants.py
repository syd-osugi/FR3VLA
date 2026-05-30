# hand_config_const.py
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from types import MappingProxyType

@dataclass(frozen=True)        # frozen=True 让实例真正只读
class HandConfig:
    joint_names: List[str] = field(default_factory=list)
    joint_names_en: Optional[List[str]] = None
    init_pos: List[int] = field(default_factory=list)
    preset_actions: Optional[Dict[str, List[int]]] = None

# ------------------------------------------------------------------
# 常量字典（仅构建一次）
# ------------------------------------------------------------------
_HAND_CONFIGS: Dict[str, HandConfig] = {
    "L30": HandConfig(
        joint_names=[
            "Thumb Side Swing", "Thumb Rotation", "Thumb Bend", "Thumb Tip",
            "Index Side Swing", "Index Base Bend", "Index Tip",
            "Middle Side Swing", "Middle Base", "Middle Tip",
            "Ring Side Swing", "Ring Base", "Ring Tip",
            "Little Side Swing", "Little Base", "Little Tip",
            "Wrist"
        ],
        init_pos=[255] * 17,
        preset_actions={
            "Open": [255] * 17,
            "Fist": [0] * 17
        }
    ),
    "L25": HandConfig(
        joint_names=["Thumb base", "Index finger base", "Middle finger base", "Ring finger base", "Little finger base",
                     "Thumb abduction", "Index finger abduction", "Middle finger abduction", "Ring finger abduction", "Little finger abduction",
                     "Thumb roll", "Reserved", "Reserved", "Reserved", "Reserved", "Thumb middle joint", "Index finger middle joint",
                     "Middle finger middle joint", "Ring finger middle joint", "Little finger middle joint", "Thumb tip", "Index finger tip",
                     "Middle finger tip", "Ring finger tip", "Little finger tip"],

        init_pos=[255] * 25,
        preset_actions={
            "Fist": [0] * 25,
            "Open": [255] * 25,
            "OK": [255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 0, 0, 255, 255,
                   0, 0, 0, 255, 255]

        }
    ),
    "L21": HandConfig(
        joint_names=["Thumb base", "Index finger base", "Middle finger base", "Ring finger base", "Little finger base",
                     "Thumb abduction", "Index finger abduction", "Middle finger abduction", "Ring finger abduction", "Little finger abduction",
                     "Thumb roll", "Reserved", "Reserved", "Reserved", "Reserved", "Thumb middle joint", "Reserved",
                     "Reserved", "Reserved", "Reserved", "Thumb tip", "Index finger tip", "Middle finger tip",
                     "Ring finger tip", "Little finger tip"],

        init_pos=[255] * 25
    ),
    "L20": HandConfig(
        joint_names=["Thumb base", "Index finger base", "Middle finger base", "Ring finger base", "Little finger base",
                     "Thumb abduction", "Index finger abduction", "Middle finger abduction", "Ring finger abduction", "Little finger abduction",
                     "Thumb yaw", "Reserved", "Reserved", "Reserved", "Reserved", "Thumb tip", "Index finger tip",
                     "Middle finger tip", "Ring finger tip", "Little finger tip"],
        init_pos=[255, 255, 255, 255, 255, 255, 10, 100, 180, 240, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        preset_actions={
            "Fist": [40, 0, 0, 0, 0, 131, 10, 100, 180, 240, 19, 255, 255, 255, 255, 135, 0, 0, 0, 0],
            "Open": [255, 255, 255, 255, 255, 255, 10, 100, 180, 240, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "OK": [191, 95, 255, 255, 255, 136, 107, 100, 180, 240, 72, 255, 255, 255, 255, 116, 99, 255, 255, 255],
            "Thumbs Up": [255, 0, 0, 0, 0, 127, 10, 100, 180, 240, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0],

            
        }
    ),
    "G20": HandConfig(
        joint_names=["Thumb base", "Index finger base", "Middle finger base", "Ring finger base", "Little finger base",
                     "Thumb abduction", "Index finger abduction", "Middle finger abduction", "Ring finger abduction", "Little finger abduction",
                     "Thumb yaw", "Reserved", "Reserved", "Reserved", "Reserved", "Thumb tip", "Index finger tip",
                     "Middle finger tip", "Ring finger tip", "Little finger tip"],
        init_pos=[255, 255, 255, 255, 255, 255, 193, 148, 105, 42, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        preset_actions={
            "Thumbs Up": [255, 0, 0, 0, 0, 255, 162, 162, 144, 100, 210, 255, 255, 255, 255, 255, 0, 0, 0, 0],
            "Fist": [96, 0, 0, 0, 0, 0, 193, 158, 128, 91, 132, 255, 255, 255, 255, 144, 0, 0, 0, 0],
            "Open": [255, 255, 255, 255, 255, 255, 193, 148, 105, 42, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "OK": [148, 110, 255, 255, 255, 44, 164, 100, 114, 127, 178, 255, 255, 255, 255, 94, 71, 255, 255, 255],
            "Thumb to Middle": [191, 255, 55, 255, 255, 96, 95, 100, 114, 127, 105, 255, 255, 255, 255, 94, 255, 108, 255, 255],
            "Thumb to Ring": [191, 255, 255, 72, 255, 115, 95, 100, 114, 127, 60, 255, 255, 255, 255, 94, 255, 255, 97, 255],
            "Thumb to Little": [191, 255, 255, 255, 55, 0, 95, 100, 114, 121, 70, 255, 255, 255, 255, 94, 255, 255, 255, 100],
            "Ready 1": [255, 0, 0, 0, 0, 255, 162, 162, 144, 100, 210, 255, 255, 255, 255, 255, 0, 0, 0, 0],
            "One": [96, 255, 0, 0, 0, 0, 190, 161, 127, 80, 68, 255, 255, 255, 255, 144, 255, 0, 0, 0],
            "Two": [96, 255, 255, 0, 0, 0, 190, 66, 127, 80, 68, 255, 255, 255, 255, 144, 255, 255, 0, 0],
            "Three": [96, 255, 255, 255, 0, 0, 200, 132, 76, 80, 68, 255, 255, 255, 255, 144, 255, 255, 255, 0],
            "Four": [80, 255, 255, 255, 255, 78, 200, 132, 114, 48, 129, 255, 255, 255, 255, 64, 255, 255, 255, 255],
            "Five": [255, 255, 255, 255, 255, 255, 193, 148, 105, 42, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "Six": [255, 0, 0, 0, 255, 255, 156, 126, 125, 42, 245, 255, 255, 255, 255, 255, 0, 0, 0, 255],
            "Seven": [38, 0, 0, 0, 0, 55, 156, 126, 125, 117, 145, 255, 255, 255, 255, 255, 164, 163, 0, 0],
            "Eight": [255, 255, 0, 0, 0, 255, 162, 162, 144, 100, 210, 255, 255, 255, 255, 255, 255, 0, 0, 0],
            "Nine": [67, 255, 0, 0, 0, 37, 162, 162, 144, 100, 85, 255, 255, 255, 255, 169, 0, 0, 0, 0],
            "Action 1": [255, 255, 255, 255, 255, 255, 193, 148, 105, 42, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "Action 2": [255, 255, 255, 255, 255, 255, 125, 129, 125, 130, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "Action 3": [255, 255, 255, 255, 255, 255, 193, 148, 105, 42, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "Action 4": [255, 255, 255, 255, 255, 255, 125, 129, 125, 130, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "Action 5": [255, 255, 255, 255, 255, 255, 255, 255, 247, 255, 210, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "Action 6": [255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 210, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "Action 7": [255, 255, 255, 255, 255, 255, 255, 255, 247, 255, 210, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "Action 8": [255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 210, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "Action 9": [255, 255, 255, 255, 255, 255, 125, 129, 125, 130, 210, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "Root 1": [0, 0, 0, 0, 0, 255, 125, 129, 125, 130, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "Root 2": [255, 255, 255, 255, 255, 255, 125, 129, 125, 130, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "Root 1": [0, 0, 0, 0, 0, 255, 125, 129, 125, 130, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "Root 2": [255, 255, 255, 255, 255, 255, 125, 129, 125, 130, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "Tip 1": [6, 0, 0, 0, 0, 255, 125, 129, 125, 130, 219, 255, 255, 255, 255, 125, 0, 0, 0, 0],
            "Tip 2": [6, 0, 0, 0, 0, 255, 125, 129, 125, 130, 219, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "Tip 3": [6, 0, 0, 0, 0, 255, 125, 129, 125, 130, 219, 255, 255, 255, 255, 125, 0, 0, 0, 0],
            "Tip 4": [6, 0, 0, 0, 0, 255, 125, 129, 125, 130, 219, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "Default": [255, 255, 255, 255, 255, 255, 193, 148, 105, 42, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255]
        }
    ),
    # Thumb joint (ball type) L10
    # "L10": HandConfig(
    #     joint_names_en=["thumb_cmc_pitch", "thumb_cmc_roll", "index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch",
    #                     "index_mcp_roll", "ring_mcp_roll", "pinky_mcp_roll", "thumb_cmc_yaw"],
    #     joint_names=["Thumb base", "Thumb abduction", "Index finger base", "Middle finger base", "Ring finger base",
    #                  "Little finger base", "Index finger abduction", "Ring finger abduction", "Little finger abduction", "Thumb rotation"],
    #     init_pos=[255] * 10,
    #     preset_actions={
    #         "Fist": [75, 128, 0, 0, 0, 0, 128, 128, 128, 57],
    #         "Open": [255, 128, 255, 255, 255, 255, 128, 128, 128, 128],
    #         "OK": [110, 128, 75, 255, 255, 255, 128, 128, 128, 68],
    #         "Thumbs Up": [255, 145, 0, 0, 0, 0, 0, 255, 255, 65]
    #     }
    # ),
    # Thumb joint (gear type) L10

    "L10": HandConfig(
        joint_names=["thumb_cmc_pitch", "thumb_cmc_roll", "index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch",
                        "index_mcp_roll", "ring_mcp_roll", "pinky_mcp_roll", "thumb_cmc_yaw"],
        init_pos=[255] * 10,
        preset_actions={
            "Open": [255, 255, 255, 255, 255, 255, 128, 67, 89, 255],
            "Thumbs Up": [255, 255, 0, 0, 0, 0, 128, 67, 89, 255],
            "Fist": [90, 0, 0, 0, 0, 0, 128, 67, 89, 197],
            "One": [55, 0, 255, 0, 0, 0, 128, 67, 89, 124],
            "Two": [55, 0, 255, 255, 0, 0, 128, 67, 89, 124],
            "Three": [116, 255, 255, 255, 255, 0, 128, 67, 89, 255],
            "Four": [0, 0, 255, 255, 255, 255, 128, 67, 89, 255],
            "Five": [255, 255, 255, 255, 255, 255, 128, 67, 89, 255],
            "Six": [255, 255, 0, 0, 0, 255, 128, 67, 89, 255],
            "Seven1": [255, 37, 119, 112, 0, 0, 128, 67, 89, 211],
            "Seven2": [91, 37, 119, 112, 0, 0, 128, 67, 89, 211],
            "Eight": [255, 255, 255, 0, 0, 0, 128, 67, 89, 255],
            "Nine": [59, 0, 134, 0, 0, 0, 128, 67, 89, 153],
            "Abduction0": [255, 0, 255, 255, 255, 255, 128, 67, 89, 153],
            "Abduction1": [0, 0, 255, 255, 255, 255, 255, 255, 255, 255],
            "Abduction2": [0, 0, 255, 255, 255, 255, 0, 0, 0, 255],
            "Abduction3": [0, 0, 255, 255, 255, 255, 255, 255, 255, 255],
            "Abduction4": [0, 0, 255, 255, 255, 255, 0, 0, 0, 255],
            "Abduction5": [0, 0, 255, 255, 255, 255, 255, 255, 255, 255],
            "Abduction6": [0, 0, 255, 255, 255, 255, 128, 67, 89, 255],
            "Flex1": [209, 0, 124, 122, 123, 122, 128, 67, 89, 255],
            "Flex2": [0, 0, 255, 255, 255, 255, 128, 67, 89, 255],
            "Flex3": [209, 0, 124, 122, 123, 122, 128, 67, 89, 255],
            "Flex4": [0, 0, 255, 255, 255, 255, 128, 67, 89, 255],
            "Flex5": [209, 0, 124, 122, 123, 122, 128, 67, 89, 255],
            "Flex6": [0, 0, 255, 255, 255, 255, 128, 67, 89, 255],
            "OK": [84, 39, 122, 255, 255, 255, 128, 67, 89, 255],
            "ThumbPressure1": [134, 39, 99, 255, 255, 255, 128, 67, 89, 255],
            "ThumbPressure2": [73, 39, 91, 255, 255, 255, 128, 67, 89, 255],
            "IndexPressure1": [151, 39, 190, 255, 255, 255, 128, 67, 89, 255],
            "IndexPressure2": [58, 39, 103, 255, 255, 255, 128, 67, 89, 255],
            "MiddlePressure1": [40, 39, 255, 128, 255, 255, 128, 67, 89, 209],
            "MiddlePressure2": [40, 39, 255, 89, 255, 255, 128, 67, 89, 209],
            "RingPressure1": [51, 39, 255, 255, 139, 255, 128, 67, 89, 154],
            "RingPressure2": [51, 39, 255, 255, 83, 255, 128, 67, 89, 154],
            "LittleFingerPressure1": [62, 39, 255, 255, 255, 155, 128, 67, 89, 101],
            "LittleFingerPressure2": [62, 39, 255, 255, 255, 75, 128, 67, 89, 101],
        }
    ),
    # Thumb joint (ball type) L7
    # "L7": HandConfig(
    #     joint_names=["大拇指弯曲", "大拇指横摆", "食指弯曲", "中指弯曲", "无名指弯曲",
    #                  "小拇指弯曲", "拇指旋转"],
    #     init_pos=[250] * 7,
    #     preset_actions={
    #         "点赞": [255, 111, 0, 0, 0, 0, 86],
    #         "握拳": [71, 79, 0, 0, 0, 0, 64],
    #         "张开": [255, 111, 250, 250, 250, 250, 55],
    #         "OK": [141, 111, 168, 250, 250, 250, 86],
            
    #     }
    # ),
    # Thumb joint (gear type) L7
    "L7": HandConfig(
        joint_names=["Thumb flexion", "Thumb yaw", "Index finger flexion", "Middle finger flexion", "Ring finger flexion",
                     "Little finger flexion", "Thumb rotation"],
        init_pos=[250] * 7,
        preset_actions={
            "Open": [255, 111, 250, 250, 250, 250, 55],
            "Thumbs Up": [255, 255, 0, 0, 0, 0, 255],
            "Thumbs Up 2": [255, 0, 0, 0, 0, 0, 255],
            "Fist": [65, 0, 0, 0, 0, 0, 93],
            "One": [66, 0, 255, 0, 0, 0, 93],
            "One1": [61, 0, 255, 0, 0, 0, 255],
            "Two": [0, 0, 255, 255, 0, 0, 255],
            "Three": [0, 0, 255, 255, 255, 0, 255],
            "Four": [0, 0, 255, 255, 255, 255, 119],
            "Five": [255, 111, 250, 250, 250, 250, 55],
            "OK": [99, 15, 146, 250, 250, 250, 206],
            "ThumbPressure1": [99, 15, 206, 250, 250, 250, 206],
            "ThumbPressure2": [109, 15, 70, 250, 250, 250, 206],
            "IndexPressure1": [99, 15, 206, 250, 250, 250, 206],
            "IndexPressure2": [69, 15, 140, 250, 250, 250, 206],
        }
    ),
    "O6": HandConfig(
        joint_names_en=["thumb_cmc_pitch", "thumb_cmc_yaw", "index_mcp_pitch", "middle_mcp_pitch", "pinky_mcp_pitch", "ring_mcp_pitch"],
        joint_names=["Thumb flexion", "Thumb yaw", "Index finger flexion", "Middle finger flexion", "Ring finger flexion", "Little finger flexion"],
        init_pos=[250] * 6,
        preset_actions={
            "Open": [250, 250, 250, 250, 250, 250],
            "One": [125, 18, 255, 0, 0, 0],
            "Two": [92, 87, 255, 255, 0, 0],
            "Three": [92, 87, 255, 255, 255, 0],
            "Four": [92, 87, 255, 255, 255, 255],
            "Five": [255, 255, 255, 255, 255, 255],
            "OK": [96, 100, 118, 250, 250, 250],
            "Thumbs Up": [250, 79, 0, 0, 0, 0],
            "Fist": [102, 18, 0, 0, 0, 0],
        }
    ),
    "L6": HandConfig(
        joint_names_en=["thumb_cmc_pitch", "thumb_cmc_yaw", "index_mcp_pitch", "middle_mcp_pitch", "pinky_mcp_pitch", "ring_mcp_pitch"],
        joint_names=["Thumb flexion", "Thumb yaw", "Index finger flexion", "Middle finger flexion", "Ring finger flexion", "Little finger flexion"],
        init_pos=[250] * 6,
        preset_actions={
            "Open": [250, 250, 250, 250, 250, 250],
            "One": [0, 18, 255, 0, 0, 0],
            "Two": [0, 39, 255, 255, 0, 0],
            "Three": [0, 39, 255, 255, 255, 0],
            "Four": [0, 0, 255, 255, 255, 255],
            "Five": [255, 255, 255, 255, 255, 255],
            "OK": [74, 13, 153, 255, 255, 255],
            "Thumbs Up": [255, 255, 0, 0, 0, 0],
            "Fist": [79, 11, 0, 0, 0, 0],
            "Sequence Action 1": [250, 250, 250, 250, 250, 250],
            "Sequence Action 2": [250, 250, 0, 250, 250, 0],
            "Sequence Action 3": [250, 250, 0, 0, 0, 0],
            "Sequence Action 4": [250, 250, 0, 0, 0, 255],
            "Sequence Action 5": [250, 250, 0, 0, 255, 255],
            "Sequence Action 6": [250, 250, 0, 255, 255, 255],
            "Sequence Action 7": [250, 250, 250, 250, 250, 250],
            "Index Pressure Prep 1": [0, 18, 255, 0, 0, 0],
            "Index Pressure Test": [9, 42, 55, 250, 250, 250],
            "Index Pressure Prep 2": [0, 18, 255, 0, 0, 0],
            "Thumb Pressure Prep 1": [139, 18, 130, 0, 0, 0],
            "Thumb Pressure Test": [39, 30, 122, 250, 250, 250],
            "Thumb Pressure Prep 2": [139, 18, 130, 0, 0, 0]
        }
    ),
}

HAND_CONFIGS = MappingProxyType(_HAND_CONFIGS)
