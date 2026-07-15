"""부품의 이동(translation) 또는 회전(rotation)을 표현하는 모듈.

Action은 하나의 상태(State)를 다른 상태로 옮기는 단위 동작이다.
- action_type: 동작의 종류 (translation | rotation)
- value: 동작의 값 (x, y, z 축 순서, 3개 성분)
    translation -> [dx, dy, dz] 실수 이동량
    rotation    -> [rx, ry, rz] 각 축 회전각(90도의 배수)

회전 규약은 state.py와 동일한 scipy 'XYZ' intrinsic 규약을 사용한다.
(부품 자체 축 기준, X -> Y -> Z 순서로 적용)
회전 중심은 부품의 무게 중심(center_mass)이며, 중심 좌표는 부품 로컬 좌표계 기준이다.
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

    def _to_origin_transformation_matrix(self) -> np.ndarray:
        """이 동작을 부품 로컬 좌표계 '원점' 기준의 4x4 동차 변환 행렬로 반환한다.

        내부 헬퍼다. rotation 의 경우 회전 중심(부품 무게 중심) 보정이 포함되지
        않은 순수 회전이므로 외부에서 직접 사용하지 않는다.
        규약상 회전 중심은 부품 무게 중심이므로, 외부에서는 반드시
        to_transformation_matrix_about() 를 사용한다.
        """
        transformation_matrix = np.eye(4)
        if self.is_translation():
            transformation_matrix[:3, 3] = self.value
        else:
            rotation_matrix = np.rint(
                Rotation.from_euler("XYZ", self.value, degrees=True).as_matrix()
            )
            # -0.0 을 0.0 으로 통일해 직렬화 재현성을 확보한다. (state.py 와 동일)
            rotation_matrix[rotation_matrix == 0.0] = 0.0
            transformation_matrix[:3, :3] = rotation_matrix
        return transformation_matrix

    def to_transformation_matrix_about(
        self, center: Tuple[float, float, float]
    ) -> np.ndarray:
        """이 동작을 중심점 center 기준의 4x4 동차 변환 행렬(변화량)로 반환한다.

        rotation 의 경우 T(center) · R · T(-center) 로 중심 보정된 행렬을 반환하며,
        center 점은 이 변환에 의해 움직이지 않는다(제자리 회전).
        translation 의 경우 중심과 무관하므로 center 는 무시된다.

        center 는 부품 로컬 좌표계 기준의 무게 중심 좌표다.
        """
        if len(center) != 3:
            raise InvalidActionException(
                f"center must contain exactly 3 coordinates, received {len(center)}"
            )
        try:
            normalized_center = tuple(float(coordinate) for coordinate in center)
        except (TypeError, ValueError) as error:
            raise InvalidActionException(
                f"center must contain numeric coordinates, received {center!r}"
            ) from error

        transformation_matrix = self._to_origin_transformation_matrix()
        if self.is_translation():
            return transformation_matrix

        translation_to_center = np.eye(4)
        translation_to_center[:3, 3] = normalized_center
        translation_from_center = np.eye(4)
        translation_from_center[:3, 3] = np.negative(normalized_center)

        return translation_to_center @ transformation_matrix @ translation_from_center