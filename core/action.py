"""부품의 이동(translation) 또는 회전(rotation)을 표현하는 모듈.

Action은 하나의 상태(State)를 다른 상태로 옮기는 단위 동작이다.
- action_type: 동작의 종류 (translation | rotation)
- value: 동작의 값 (x, y, z 축 순서, 3개 성분)
    translation -> [dx, dy, dz] 실수 이동량
    rotation    -> [rx, ry, rz] 각 축 회전각(90도의 배수)

회전 규약은 state.py와 동일한 scipy 'xyz' extrinsic 규약을 사용한다.
(State에 Action을 적용해 다음 State를 계산하는 전이 로직은 planner가 담당한다.)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


class ActionType(Enum):
    TRANSLATION = "translation"
    ROTATION = "rotation"


class InvalidActionException(Exception):
    """Action 생성 값이 규약을 벗어날 때 발생."""


@dataclass(frozen=True)
class Action:
    """부품의 이동 또는 회전을 표현하는 불변 클래스."""

    action_type: ActionType
    # translation 이면 float 3-tuple, rotation 이면 int 3-tuple 로 정규화한다.
    value: Tuple[float, float, float] | Tuple[int, int, int]
    def __post_init__(self) -> None:
        if not isinstance(self.action_type, ActionType):
            raise InvalidActionException(
                f"action_type must be an ActionType, received {type(self.action_type)}"
            )
        if len(self.value) != 3:
            raise InvalidActionException(
                f"value must contain exactly 3 components, received {len(self.value)}"
            )

        if self.action_type is ActionType.TRANSLATION:
            try:
                normalized_value = tuple(float(component) for component in self.value)
            except (TypeError, ValueError) as error:
                raise InvalidActionException(
                    f"translation value must be numeric, received {self.value!r}"
                ) from error
        else:
            try:
                normalized_value = tuple(int(component) for component in self.value)
            except (TypeError, ValueError) as error:
                raise InvalidActionException(
                    f"rotation value must be integer angles, received {self.value!r}"
                ) from error
            # State 는 각도를 {0,90,180,270} 로 제한하지만 Action 은 변화량이므로
            # 90 의 배수이기만 하면 음수(-90)나 360 이상도 허용한다.
            for angle in normalized_value:
                if angle % 90 != 0:
                    raise InvalidActionException(
                        f"rotation component {angle} must be a multiple of 90 degrees"
                    )

        object.__setattr__(self, "value", normalized_value)

    def is_translation(self) -> bool:
        return self.action_type is ActionType.TRANSLATION

    def is_rotation(self) -> bool:
        return self.action_type is ActionType.ROTATION

    def to_transformation_matrix(self) -> np.ndarray:
        """이 동작을 4x4 동차 변환 행렬(변화량)로 반환한다.

        주의: rotation 의 경우 이 행렬은 '월드 원점' 기준 순수 회전이며,
        회전 중심(부품 자기 중심) 보정은 포함되어 있지 않다.
        """
        transformation_matrix = np.eye(4)
        if self.is_translation():
            transformation_matrix[:3, 3] = self.value
        else:
            rotation_matrix = np.rint(
                Rotation.from_euler("xyz", self.value, degrees=True).as_matrix()
            )
            # -0.0 을 0.0 으로 통일해 직렬화 재현성을 확보한다. (state.py 와 동일)
            rotation_matrix[rotation_matrix == 0.0] = 0.0
            transformation_matrix[:3, :3] = rotation_matrix
        return transformation_matrix
