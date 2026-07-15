"""부품의 상태(State)를 표현하는 모듈.

State는 한 부품의 강체 자세(pose)를 나타낸다.
- position: 연속적인 위치 (x, y, z) 실수 좌표
- rotation: 이산적인 회전 각도 (x, y, z 축 순서). 각 성분은 {0, 90, 180, 270} 중 하나.

회전 각도의 축 순서와 오일러 규약은 프로젝트 전체에서 다음으로 통일한다.
    scipy 'XYZ' intrinsic (부품 자체 축 기준, X -> Y -> Z 순서로 적용)
    rotation[0] = X축 회전(roll)
    rotation[1] = Y축 회전(pitch)
    rotation[2] = Z축 회전(yaw)
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


ALLOWED_ROTATION_ANGLES = frozenset({0, 90, 180, 270})


class InvalidStateException(Exception):
    """State 생성 값이 규약(위치 3개 실수, 회전 3개 {0,90,180,270})을 벗어날 때 발생."""


@dataclass(frozen=True)
class State:
    """부품의 강체 자세(pose)를 표현하는 불변 클래스."""

    position: Tuple[float, float, float]
    rotation: Tuple[int, int, int]

    def __post_init__(self) -> None:
        if len(self.position) != 3:
            raise InvalidStateException(
                f"position must contain exactly 3 coordinates, received {len(self.position)}"
            )
        if len(self.rotation) != 3:
            raise InvalidStateException(
                f"rotation must contain exactly 3 angles, received {len(self.rotation)}"
            )

        try:
            normalized_position = tuple(float(coordinate) for coordinate in self.position)
        except (TypeError, ValueError) as error:
            raise InvalidStateException(
                f"position must contain numeric coordinates, received {self.position!r}"
            ) from error
        try:
            normalized_rotation = tuple(int(angle) for angle in self.rotation)
        except (TypeError, ValueError) as error:
            raise InvalidStateException(
                f"rotation must contain integer angles, received {self.rotation!r}"
            ) from error

        for angle in normalized_rotation:
            if angle not in ALLOWED_ROTATION_ANGLES:
                raise InvalidStateException(
                    f"rotation angle {angle} is not one of {sorted(ALLOWED_ROTATION_ANGLES)}"
                )

        object.__setattr__(self, "position", normalized_position)
        object.__setattr__(self, "rotation", normalized_rotation)

    def to_rotation_matrix(self) -> np.ndarray:
        """이 상태의 3x3 회전 행렬을 반환한다.

        회전 각도가 90도의 배수로 제한되어 행렬 성분이 {-1, 0, 1}로 확정되므로,
        부동소수점 잡음을 제거해 충돌 검사와 직렬화의 재현성을 확보한다.
        """
        rotation_matrix = np.rint(
            Rotation.from_euler("XYZ", self.rotation, degrees=True).as_matrix()
        )
        # np.rint 은 -0.0 을 그대로 남긴다. -0.0 과 0.0 은 값은 같지만 직렬화
        # 바이트가 달라(msgpack 등) 재현성을 깨므로 +0.0 으로 통일한다.
        rotation_matrix[rotation_matrix == 0.0] = 0.0
        return rotation_matrix

    def to_transformation_matrix(self) -> np.ndarray:
        """이 상태를 4x4 동차 변환 행렬로 반환한다 (부품 로컬 좌표 -> 월드 좌표)."""
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = self.to_rotation_matrix()
        transformation_matrix[:3, 3] = self.position
        return transformation_matrix