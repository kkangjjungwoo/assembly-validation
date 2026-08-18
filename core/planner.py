"""분해(disassembly) 경로를 탐색하는 RRT* 경로 계획 모듈.

이미 조립된 상태의 부품 하나(moving solid)를 나머지 부품들(고정 장애물) 사이에서
빼내는 충돌 없는 경로를 RRT*(Rapidly-exploring Random Tree Star)로 탐색한다.
탐색으로 얻은 분해 경로를 뒤집으면 최적 조립 경로가 된다(to_assembly_path).

상태 공간(hybrid)
-----------------
- 위치(position): 연속 실수 좌표 (x, y, z) in R^3.
- 회전(rotation): 이산 자세. {0,90,180,270} 각도 격자(64가지)를 회전 행렬로 환산하면
  서로 겹쳐 실제로는 24가지 고유 자세(정팔면체 회전군)만 존재한다. 서로 다른 오일러
  삼중항이 같은 자세를 나타내므로, 노드 중복 판정·목표 판정·재배선은 오일러 튜플이
  아니라 회전 '행렬'을 기준으로 비교하고, 각 고유 자세를 대표 오일러 하나로 정규화한다.

간선(edge) 모델
---------------
임의의 두 상태 A, B 사이의 이동은 최대 두 개의 단위 동작으로 분해한다(core.action 규약:
단위 동작은 이동 또는 회전 중 하나).
1. 회전 동작: A 의 자세를 B 의 자세로 바꾼다. 회전은 부품 자신의 '월드 무게중심'을
   기준으로 수행한다(core.action 이 남긴 무게중심 보정은 planner 몫이라는 규약을 따름).
   이산 자세 재배치이므로 결과 상태만 is_valid_state 로 검사한다.
2. 이동 동작: 회전 후 위치에서 B 의 위치까지 직선 이동한다. 연속 이동이므로
   is_path_segment_valid 로 구간 전체를 표본 검사한다(터널링 방지).

목표(goal)
----------
움직이는 부품의 월드 축정렬 경계상자(AABB)가 나머지 부품 전체의 AABB 로부터 여유값
이상 떨어지면(어느 한 축에서 분리) 부품이 조립체 밖으로 '추출'된 것으로 본다. 닫히지
않은(non-watertight) 메쉬에서도 안정적으로 판정할 수 있는 기하 기준이다.

충돌 판정
---------
판정 자체는 core.interference 가 하고, 이 모듈의 InterferenceSessionAdapter 가 State 를
4x4 변환으로 번역해 넘긴다. planner 는 어댑터의 세 메서드만 쓰므로 판정 근거를 바꾸어도
탐색 코드는 그대로다.
"""

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from core.action import Action, ActionType
from core.interference import InterferenceException, InterferenceSession
from core.state import State


class PlanningException(Exception):
    """경로 계획기의 입력·구성이 규약을 벗어날 때 발생."""


_ROTATION_GRID_ANGLES = (0, 90, 180, 270)


def _to_grid_rotation_matrix(euler_angles: Tuple[int, int, int]) -> np.ndarray:
    """격자 오일러 각도(90도 배수)를 성분이 {-1,0,1}로 확정된 회전 행렬로 변환한다.

    회전 규약은 state.py / action.py 와 반드시 동일해야 한다(scipy 'XYZ' intrinsic).
    planner 는 회전 행렬을 24가지 대표 오일러로 정규화해 State.rotation 에 저장하는데,
    State.to_rotation_matrix() 가 이 오일러를 다시 행렬로 해석하므로 규약이 어긋나면
    (예: 'xyz' extrinsic) 저장한 자세가 다른 자세로 재해석된다(격자 64개 중 40개 불일치).
    """
    rotation_matrix = np.rint(
        Rotation.from_euler("XYZ", euler_angles, degrees=True).as_matrix()
    )
    rotation_matrix[rotation_matrix == 0.0] = 0.0
    return rotation_matrix


def _to_matrix_key(rotation_matrix: np.ndarray) -> Tuple[int, ...]:
    """회전 행렬을 정수 키(딕셔너리 조회용)로 변환한다."""
    return tuple(np.rint(rotation_matrix).astype(int).flatten())


def _build_canonical_rotation_lookup() -> Tuple[Dict[Tuple[int, ...], Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """24가지 고유 회전 자세의 (행렬키 -> 대표 오일러) 조회표와 대표 오일러 목록을 만든다."""
    matrix_key_to_euler: Dict[Tuple[int, ...], Tuple[int, int, int]] = dict()
    canonical_eulers: List[Tuple[int, int, int]] = list()
    for euler_angles in itertools.product(_ROTATION_GRID_ANGLES, repeat=3):
        matrix_key = _to_matrix_key(_to_grid_rotation_matrix(euler_angles))
        if matrix_key not in matrix_key_to_euler:
            matrix_key_to_euler[matrix_key] = euler_angles
            canonical_eulers.append(euler_angles)
    return matrix_key_to_euler, canonical_eulers


# 24가지 고유 자세 조회표를 모듈 적재 시 한 번만 구축한다.
_MATRIX_KEY_TO_EULER, _CANONICAL_ROTATION_EULERS = _build_canonical_rotation_lookup()

# 오일러 3-tuple -> 대표 오일러 조회표. 격자 조합 4^3 = 64 가지를 회전행렬로 중복 제거하면
# 24 가지다((0,90,90) 과 (90,0,90) 은 같은 회전). 오일러 tuple 을 그대로 비교하면 같은
# 회전을 다르다고 오판하므로 대표 오일러로 정규화한 뒤 비교한다.
_EULER_TO_CANONICAL: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {
    euler_angles: _MATRIX_KEY_TO_EULER[_to_matrix_key(_to_grid_rotation_matrix(euler_angles))]
    for euler_angles in itertools.product(_ROTATION_GRID_ANGLES, repeat=3)
}


def _to_canonical_rotation(rotation: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """오일러 3-tuple 을 같은 회전을 나타내는 대표 오일러로 정규화한다."""
    rotation_key = tuple(int(component) % 360 for component in rotation)
    canonical = _EULER_TO_CANONICAL.get(rotation_key)
    if canonical is not None:
        return canonical
    # 격자를 벗어난 값(90도 배수가 아님)은 조회표에 없다. 행렬로 직접 정규화한다.
    return _MATRIX_KEY_TO_EULER[_to_matrix_key(_to_grid_rotation_matrix(rotation_key))]


def _to_canonical_rotation_euler(rotation_matrix: np.ndarray) -> Tuple[int, int, int]:
    """회전 행렬을 24가지 고유 자세의 대표 오일러 각도로 정규화한다."""
    matrix_key = _to_matrix_key(rotation_matrix)
    if matrix_key not in _MATRIX_KEY_TO_EULER:
        raise PlanningException(
            f"rotation matrix {matrix_key} is not a 90-degree grid orientation"
        )
    return _MATRIX_KEY_TO_EULER[matrix_key]


@dataclass(frozen=True)
class RRTStarConfig:
    """RRT* 탐색 하이퍼파라미터.

    Attributes:
        max_iteration_count: 최대 반복(샘플) 횟수.
        translation_step_size: 한 번의 확장에서 위치가 목표로 나아가는 최대 거리.
        neighbor_radius: 부모 재선택·재배선 시 이웃으로 볼 상태 거리 반경(metric 단위).
        goal_sample_rate: 무작위 샘플 대신 목표(추출) 영역을 샘플링할 확률 [0,1].
        rotation_distance_weight: 자세가 다를 때 비용에 더하는 길이(자세 전이 1회의 값).
            비용 함수의 회전 항이 지시함수이므로 '라디안당 길이' 가 아니라 '전이 1회당
            길이' 다. translation_step_size 와 같게 두면 '회전 한 번 = 한 스텝 이동' 이 된다.
        translation_interpolation_count: 이동 간선의 스윕 충돌 검사 표본 수(2 이상).
        rotation_interpolation_count: 회전 간선의 스윕 충돌 검사 표본 수(2 이상). 시작->끝
            자세를 SLERP 로 세분해 각 중간 자세의 관통을 검사한다(회전 도중 충돌 방지).
        removal_clearance: 목표 판정에서 부품 AABB 가 조립체 AABB 로부터 떨어져야 할 여유값.
        sampling_margin_ratio: 샘플링 경계상자를 조립체 AABB 대비 각 방향으로 확장하는 비율.
        does_sample_rotation: True 이면 자세도 무작위 샘플링, False 이면 시작 자세 고정(이동만).
        maximum_extension_step_count: 한 번의 반복에서 샘플 방향으로 연속으로 뻗는 최대 확장
            횟수(1 이상). 1이면 표준 RRT*(한 스텝), 크면 RRT-Connect 식 탐욕 확장으로 좁은
            축 통로를 따라 부품을 한 번에 끝까지 밀어낼 수 있다.
        stops_at_first_feasible_path: True 이면 목표에 도달하는 첫 실현 가능 경로를 찾는
            즉시 탐색을 중단하고 그 경로를 반환한다(최고 속도 우선). RRT* 의 점근적
            최적성을 포기하는 대신, 병렬 경주(여러 부품을 동시에 탐색해 먼저 성공한
            부품을 채택)에서 승자를 최대한 빨리 확정하는 용도이다. False 이면
            max_iteration_count 까지 돌며 더 낮은 비용의 경로를 계속 찾는다.
        random_seed: 난수 시드(재현성).
    """

    max_iteration_count: int
    translation_step_size: float
    neighbor_radius: float
    goal_sample_rate: float
    rotation_distance_weight: float
    translation_interpolation_count: int
    rotation_interpolation_count: int
    removal_clearance: float
    sampling_margin_ratio: float
    does_sample_rotation: bool
    maximum_extension_step_count: int
    stops_at_first_feasible_path: bool
    random_seed: int

    def __post_init__(self) -> None:
        if self.max_iteration_count < 1:
            raise PlanningException(
                f"max_iteration_count must be at least 1, received {self.max_iteration_count}"
            )
        if self.translation_step_size <= 0.0:
            raise PlanningException(
                f"translation_step_size must be positive, received {self.translation_step_size}"
            )
        if self.neighbor_radius <= 0.0:
            raise PlanningException(
                f"neighbor_radius must be positive, received {self.neighbor_radius}"
            )
        if not 0.0 <= self.goal_sample_rate <= 1.0:
            raise PlanningException(
                f"goal_sample_rate must be within [0, 1], received {self.goal_sample_rate}"
            )
        if self.rotation_distance_weight < 0.0:
            raise PlanningException(
                f"rotation_distance_weight must be non-negative, received {self.rotation_distance_weight}"
            )
        if self.translation_interpolation_count < 2:
            raise PlanningException(
                f"translation_interpolation_count must be at least 2, received {self.translation_interpolation_count}"
            )
        if self.rotation_interpolation_count < 2:
            raise PlanningException(
                f"rotation_interpolation_count must be at least 2, received {self.rotation_interpolation_count}"
            )
        if self.removal_clearance <= 0.0:
            raise PlanningException(
                f"removal_clearance must be positive, received {self.removal_clearance}"
            )
        if self.sampling_margin_ratio <= 0.0:
            raise PlanningException(
                f"sampling_margin_ratio must be positive, received {self.sampling_margin_ratio}"
            )
        if self.maximum_extension_step_count < 1:
            raise PlanningException(
                f"maximum_extension_step_count must be at least 1, received {self.maximum_extension_step_count}"
            )


@dataclass
class _TreeNode:
    """탐색 트리의 노드. 상태·부모 인덱스·시작으로부터의 누적 비용을 담는다."""

    state: State
    parent_index: Optional[int]
    cost_from_start: float


@dataclass(frozen=True)
class PlanningResult:
    """탐색 결과.

    Attributes:
        is_success: 목표(추출) 상태에 도달하는 경로를 찾았는지 여부.
        states: 시작부터 목표까지의 상태 나열(성공 시 길이 >= 1, 실패 시 비어 있음).
        actions: 상태 전이 동작 나열(len(states) - 1 개).
        cost: 경로의 총 비용(metric 거리 합). 실패 시 무한대.
        iteration_count: 실제 수행한 반복 횟수.
    """

    is_success: bool
    states: Tuple[State, ...]
    actions: Tuple[Action, ...]
    cost: float
    iteration_count: int


class InterferenceSessionAdapter:
    """InterferenceSession 을 planner 가 기대하는 충돌 세션 인터페이스로 감싼다.

    planner 는 State(연속 위치 + 이산 회전)로 말하고 판정기는 4x4 변환으로 말하므로
    그 번역을 이 어댑터가 담당한다. 제공하는 것은 세 메서드뿐이다 — is_valid_state,
    is_path_segment_valid, is_rotation_segment_valid. 회전 스윕의 기하 규약(무게중심
    고정 SLERP)은 아래 RRTStarPlanner 의 회전 규약과 동일해야 한다.

    FCL 관통 깊이를 대체한 근거(Cleaner.STEP 실측): 대형 껍데기가 자유축으로 나가는
    경로에서 간섭 증가는 모든 거리에서 정확히 0 인데 FCL depth 는 8~10 유닛 구간에서
    14.109 로 튀어 충돌로 판정한다. 이 가짜 관문이 껍데기의 유일한 탈출로를 막아 분해가
    8/13 에서 멈췄다.
    """

    def __init__(
        self,
        interference_session: InterferenceSession,
        moving_centroid_local: np.ndarray,
    ):
        """어댑터를 만든다.

        Args:
            interference_session: 이동 부품과 장애물이 이미 묶인 간섭 부피 세션.
            moving_centroid_local: 이동 부품의 로컬 무게중심. 회전 스윕의 고정점
                계산에 쓰며, planner 의 무게중심 기준 회전 규약과 일치해야 한다.
        """
        self.interference_session = interference_session
        self.moving_centroid_local = np.asarray(moving_centroid_local, dtype=float)

    @property
    def query_count(self) -> int:
        """지금까지의 판정 질의 수."""
        return self.interference_session.query_count

    @property
    def boolean_count(self) -> int:
        """AABB 사전 필터를 통과해 실제 불린이 수행된 횟수."""
        return self.interference_session.boolean_count

    def is_valid_state(self, moving_state: State) -> bool:
        """이 상태에서 조립 상태보다 간섭이 늘지 않으면 True."""
        return self.interference_session.is_transformation_valid(
            moving_state.to_transformation_matrix()
        )

    def is_path_segment_valid(
        self, start_state: State, goal_state: State, interpolation_count: int
    ) -> bool:
        """두 상태를 잇는 이동 구간 전체가 유효한지 검사한다.

        순수 이동 구간을 전제한다 — 회전이 다르면 예외를 던진다.
        """
        if interpolation_count < 2:
            raise InterferenceException(
                f"interpolation_count must be at least 2, received {interpolation_count}"
            )
        if start_state.rotation != goal_state.rotation:
            raise InterferenceException(
                "is_path_segment_valid assumes a pure translation segment; "
                f"rotations differ: {start_state.rotation} vs {goal_state.rotation}"
            )

        start_position = np.asarray(start_state.position, dtype=float)
        goal_position = np.asarray(goal_state.position, dtype=float)
        rotation_matrix = start_state.to_rotation_matrix()
        for interpolation_index in range(interpolation_count):
            interpolation_ratio = interpolation_index / (interpolation_count - 1)
            interpolated_position = (
                start_position + (goal_position - start_position) * interpolation_ratio
            )
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = interpolated_position
            if not self.interference_session.is_transformation_valid(transformation_matrix):
                return False
        return True

    def is_rotation_segment_valid(
        self, start_state: State, goal_state: State, interpolation_count: int
    ) -> bool:
        """두 상태를 잇는 회전 구간 전체가 유효한지 검사한다.

        회전 중심은 planner 의 규약과 같은 '이동 부품의 월드 무게중심'이다. 무게중심
        기준 회전은 회전 전후로 월드 무게중심을 고정하므로 시작·끝 상태는 position 이
        달라도 같은 월드 무게중심을 공유한다. 그 공유 무게중심을 고정점으로 SLERP
        보간한 각 중간 자세를 검사한다.
        """
        if interpolation_count < 2:
            raise InterferenceException(
                f"interpolation_count must be at least 2, received {interpolation_count}"
            )
        start_rotation_matrix = start_state.to_rotation_matrix()
        goal_rotation_matrix = goal_state.to_rotation_matrix()
        if np.allclose(start_rotation_matrix, goal_rotation_matrix):
            return self.is_valid_state(start_state)

        start_position = np.asarray(start_state.position, dtype=float)
        goal_position = np.asarray(goal_state.position, dtype=float)
        start_world_centroid = start_rotation_matrix @ self.moving_centroid_local + start_position
        goal_world_centroid = goal_rotation_matrix @ self.moving_centroid_local + goal_position
        if not np.allclose(start_world_centroid, goal_world_centroid, atol=1e-6):
            raise InterferenceException(
                "is_rotation_segment_valid assumes a pure rotation about the moving "
                "centroid; the start and goal states do not share a world centroid "
                f"({start_world_centroid} vs {goal_world_centroid})"
            )
        world_centroid = start_world_centroid

        key_rotations = Rotation.from_matrix(
            np.stack([start_rotation_matrix, goal_rotation_matrix])
        )
        slerp = Slerp([0.0, 1.0], key_rotations)
        for interpolation_index in range(interpolation_count):
            interpolation_ratio = interpolation_index / (interpolation_count - 1)
            interpolated_rotation_matrix = slerp([interpolation_ratio])[0].as_matrix()
            interpolated_position = (
                world_centroid - interpolated_rotation_matrix @ self.moving_centroid_local
            )
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = interpolated_rotation_matrix
            transformation_matrix[:3, 3] = interpolated_position
            if not self.interference_session.is_transformation_valid(transformation_matrix):
                return False
        return True


class RRTStarPlanner:
    """한 부품의 분해 경로를 RRT*로 탐색하는 경로 계획기.

    한 번에 하나의 부품(moving solid)만 움직이고 나머지는 조립 자세로 고정한다.
    충돌 검사는 생성 시점에 만든 간섭 부피 세션 하나를 재사용한다.
    """

    def __init__(
        self,
        moving_solid_id: int,
        solid_meshes: Dict[int, "object"],
        assembled_states: Dict[int, State],
        config: RRTStarConfig,
        interference_session: "object",
    ):
        """계획기를 구성한다.

        Args:
            moving_solid_id: 분해할(움직일) 부품의 식별자.
            solid_meshes: 부품 식별자 -> 로컬 좌표계 trimesh.Trimesh 형상.
            assembled_states: 부품 식별자 -> 조립 상태 자세. moving_solid_id 의 자세가
                탐색의 시작 상태가 된다.
            config: RRT* 하이퍼파라미터.
            interference_session: core.interference.InterferenceSession. 충돌을 '간섭
                길이' 로 판정한다 — 조립 상태의 간섭을 기준선으로 삼고 그보다 증가하면
                충돌로 본다.

        Raises:
            PlanningException: 부품 식별자가 맞지 않거나 조립 상태가 이미 충돌인 경우.
        """
        if moving_solid_id not in solid_meshes:
            raise PlanningException(
                f"moving solid {moving_solid_id} is not present in solid_meshes"
            )
        if moving_solid_id not in assembled_states:
            raise PlanningException(
                f"moving solid {moving_solid_id} is not present in assembled_states"
            )
        if set(solid_meshes.keys()) != set(assembled_states.keys()):
            raise PlanningException(
                "solid_meshes and assembled_states must share the same solid identifiers"
            )

        self.moving_solid_id = moving_solid_id
        self.solid_meshes = solid_meshes
        self.assembled_states = assembled_states
        self.config = config

        # 충돌 판정은 간섭 부피 하나로 고정한다. 어댑터가 planner 가 기대하는 세 메서드
        # (is_valid_state / is_path_segment_valid / is_rotation_segment_valid)를 제공한다.
        self.collision_session = InterferenceSessionAdapter(
            interference_session,
            np.asarray(solid_meshes[moving_solid_id].centroid, dtype=float),
        )

        self.start_state = assembled_states[moving_solid_id]
        if not self.collision_session.is_valid_state(self.start_state):
            raise PlanningException(
                "start (assembled) state is already in collision; the interference "
                "baseline must be calibrated from the assembled configuration"
            )

        moving_mesh = solid_meshes[moving_solid_id]
        self.moving_centroid_local = np.asarray(moving_mesh.centroid, dtype=float)
        self.moving_vertices_local = np.asarray(moving_mesh.vertices, dtype=float)

        self.obstacle_bounding_box = self._to_obstacle_bounding_box()
        self.sampling_lower_bound, self.sampling_upper_bound = self._to_sampling_bounds()
        self.random_generator = np.random.default_rng(config.random_seed)

        # 시작(조립) 상태에서 움직이는 부품의 월드 AABB. 목표 편향 샘플이 필요한 '이동량'을
        # 좌표계 혼동 없이 계산하는 데 쓴다. State.position 은 절대 월드 위치가 아니라 부품
        # 원본 메쉬 좌표에 더해지는 '이동(translation)'이므로, 목표 위치도 이동량으로 다뤄야 한다.
        moving_start_world_vertices = self._to_world_vertices(
            self.moving_vertices_local, self.start_state.to_transformation_matrix()
        )
        self.moving_start_lower_bound = moving_start_world_vertices.min(axis=0)
        self.moving_start_upper_bound = moving_start_world_vertices.max(axis=0)

        self.free_escape_axis_directions = self._to_free_escape_axis_directions()

    def _to_free_escape_axis_directions(self) -> List[Tuple[int, int]]:
        """시작 상태에서 한 스텝 곧게 밀 수 있는 축·부호 방향 목록을 반환한다.

        여섯 축정렬 방향(+/-X, +/-Y, +/-Z) 각각에 대해 translation_step_size 만큼의 직선
        이동이 충돌 없이 가능한지 한 번씩만 검사한다. 스냅 핏 조립에서 부품이 실제로
        빠져나갈 수 있는 방향은 대개 이 중 극소수이므로, 목표 편향 샘플을 이 방향들로
        집중시키기 위한 사전 정보로 쓴다. (반환 형식: (axis_index, sign) 튜플의 리스트,
        sign 은 +1 또는 -1). 비어 있으면 어떤 축으로도 한 스텝조차 못 미는 상태를 뜻한다.
        """
        free_axis_directions: List[Tuple[int, int]] = list()
        for axis_index in range(3):
            for sign in (1, -1):
                displacement = np.zeros(3, dtype=float)
                displacement[axis_index] = sign * self.config.translation_step_size
                probe_state = State(
                    position=tuple(np.asarray(self.start_state.position) + displacement),
                    rotation=self.start_state.rotation,
                )
                if self.collision_session.is_valid_state(
                    probe_state
                ) and self._segment_valid(
                    self.start_state,
                    probe_state,
                    self._interpolation_count_for(self.start_state, probe_state),
                ):
                    free_axis_directions.append((axis_index, sign))
        return free_axis_directions

    def _to_obstacle_bounding_box(self) -> np.ndarray:
        """고정 부품 전체의 월드 축정렬 경계상자를 [[min],[max]] (2x3)로 반환한다."""
        minimum_corners = list()
        maximum_corners = list()
        for solid_id, mesh in self.solid_meshes.items():
            if solid_id == self.moving_solid_id:
                continue
            transformation_matrix = self.assembled_states[solid_id].to_transformation_matrix()
            world_vertices = self._to_world_vertices(
                np.asarray(mesh.vertices, dtype=float), transformation_matrix
            )
            minimum_corners.append(world_vertices.min(axis=0))
            maximum_corners.append(world_vertices.max(axis=0))
        if len(minimum_corners) == 0:
            raise PlanningException(
                "assembly must contain at least one obstacle solid besides the moving solid"
            )
        return np.vstack(
            [np.min(minimum_corners, axis=0), np.max(maximum_corners, axis=0)]
        )

    def _to_sampling_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """전체 조립체 AABB 를 margin 비율로 확장한 샘플링 경계(하한, 상한)를 반환한다."""
        minimum_corners = list()
        maximum_corners = list()
        for solid_id, mesh in self.solid_meshes.items():
            transformation_matrix = self.assembled_states[solid_id].to_transformation_matrix()
            world_vertices = self._to_world_vertices(
                np.asarray(mesh.vertices, dtype=float), transformation_matrix
            )
            minimum_corners.append(world_vertices.min(axis=0))
            maximum_corners.append(world_vertices.max(axis=0))
        assembly_lower_bound = np.min(minimum_corners, axis=0)
        assembly_upper_bound = np.max(maximum_corners, axis=0)
        assembly_extent = assembly_upper_bound - assembly_lower_bound
        margin = assembly_extent * self.config.sampling_margin_ratio
        return assembly_lower_bound - margin, assembly_upper_bound + margin

    def _to_world_vertices(
        self, local_vertices: np.ndarray, transformation_matrix: np.ndarray
    ) -> np.ndarray:
        """로컬 정점들을 동차 변환행렬로 월드 좌표로 옮긴다."""
        rotation_part = transformation_matrix[:3, :3]
        translation_part = transformation_matrix[:3, 3]
        return local_vertices @ rotation_part.T + translation_part

    def get_next_state(self, state: State, action: Action) -> State:
        """상태에 동작을 적용한 다음 상태를 반환한다(회전은 부품 월드 무게중심 기준)."""
        current_rotation_matrix = state.to_rotation_matrix()
        current_position = np.asarray(state.position, dtype=float)

        if action.is_translation():
            next_position = current_position + np.asarray(action.value, dtype=float)
            return State(position=tuple(next_position), rotation=state.rotation)

        action_rotation_matrix = _to_grid_rotation_matrix(action.value)
        world_centroid = current_rotation_matrix @ self.moving_centroid_local + current_position
        next_rotation_matrix = action_rotation_matrix @ current_rotation_matrix
        next_position = (
            action_rotation_matrix @ (current_position - world_centroid) + world_centroid
        )
        return State(
            position=tuple(next_position),
            rotation=_to_canonical_rotation_euler(next_rotation_matrix),
        )

    def get_edge_actions(
        self, from_state: State, to_state: State
    ) -> List[Tuple[Action, State]]:
        """두 상태를 잇는 (동작, 결과 상태) 나열을 반환한다(회전 후 이동 순서, 최대 2개)."""
        edge_actions: List[Tuple[Action, State]] = list()
        intermediate_state = from_state

        from_rotation_matrix = from_state.to_rotation_matrix()
        to_rotation_matrix = to_state.to_rotation_matrix()
        if not np.allclose(from_rotation_matrix, to_rotation_matrix):
            rotation_delta_matrix = to_rotation_matrix @ from_rotation_matrix.T
            rotation_action = Action(
                action_type=ActionType.ROTATION,
                value=_to_canonical_rotation_euler(rotation_delta_matrix),
            )
            intermediate_state = self.get_next_state(from_state, rotation_action)
            edge_actions.append((rotation_action, intermediate_state))

        intermediate_position = np.asarray(intermediate_state.position, dtype=float)
        target_position = np.asarray(to_state.position, dtype=float)
        if not np.allclose(intermediate_position, target_position, atol=1e-9):
            translation_action = Action(
                action_type=ActionType.TRANSLATION,
                value=tuple(target_position - intermediate_position),
            )
            intermediate_state = self.get_next_state(intermediate_state, translation_action)
            edge_actions.append((translation_action, intermediate_state))

        return edge_actions
    def _segment_valid(self, start_state, goal_state, interpolation_count) -> bool:
        """이동 구간을 표본점마다 판정한다."""
        return self.collision_session.is_path_segment_valid(
            start_state, goal_state, interpolation_count
        )

    def _interpolation_count_for(self, first_state: State, second_state: State) -> int:
        """이동 거리에 비례한 스윕 표본 수를 돌려준다.

        고정 표본 수는 긴 간선에서 원리적으로 관통을 놓칠 수 있다 — 실측: 356.6 유닛을 한
        스텝에 이동하는 간선을 10 등분하면 표본 간격이 35.6 유닛이 되어, 그 사이에 있는
        부품(수십 유닛)을 통째로 건너뛴다. 그래서 표본 간격이 translation_step_size 를 넘지
        않도록 표본 수를 늘린다. 짧은 간선에서는 설정값 그대로이므로 기존 동작과 비용이
        유지된다.

        이 조정만으로 모든 관통이 사라지는 것은 아니다 — 삼각형 면제 방식을 쓰던 시절
        통제된 A/B(같은 시드, 검증 표본 고정, 이 조정만 토글)에서 관통 2 건이 소수점까지
        동일하게 남았다. 즉 근거는 '표본 간격이 부품 크기를 넘지 않게 한다' 는 것 하나이며,
        간섭 부피 판정에서는 그 두 관통이 재현되지 않는다.
        """
        distance = float(
            np.linalg.norm(
                np.asarray(second_state.position, dtype=float)
                - np.asarray(first_state.position, dtype=float)
            )
        )
        base_count = self.config.translation_interpolation_count
        if distance <= 0.0:
            return base_count
        required = int(np.ceil(distance / max(self.config.translation_step_size, 1e-9))) + 1
        return max(base_count, required)


    def is_edge_valid(self, from_state: State, to_state: State) -> bool:
        """두 상태를 잇는 간선의 모든 동작이 충돌 없이 유효한지 검사한다.

        회전 동작과 이동 동작 모두 구간 전체(스윕)를 검사한다. 회전은 시작->끝 자세를
        SLERP 로 세분해 각 중간 자세의 관통을 확인하고(회전 도중 충돌 방지), 이동은
        위치를 선형 보간해 검사한다(빠른 이동의 터널링 방지).
        """
        previous_state = from_state
        for action, resulting_state in self.get_edge_actions(from_state, to_state):
            if action.is_rotation():
                if not self.collision_session.is_rotation_segment_valid(
                    previous_state,
                    resulting_state,
                    self.config.rotation_interpolation_count,
                ):
                    return False
            else:
                if not self._segment_valid(
                    previous_state,
                    resulting_state,
                    self._interpolation_count_for(previous_state, resulting_state),
                ):
                    return False
            previous_state = resulting_state
        return True

    def get_state_distance(self, first_state: State, second_state: State) -> float:
        """두 상태 사이의 hybrid 거리를 반환한다.

            c(q1, q2) = ||p2 - p1||_2  +  w * 1[R1 != R2]

        회전 항은 자세가 다르면 w, 같으면 0 인 지시함수다. 자세 비교는 대표 오일러로
        정규화하므로 (0,90,90) 과 (90,0,90) 처럼 표현만 다른 같은 회전은 거리 0 이다.

        측지각을 쓰지 않는 이유는 행동 공간과의 정합성이다 — 한 회전 행동이 (rx, ry, rz)
        를 한꺼번에 적용하므로 90도와 180도가 모두 '행동 한 번' 인데, 측지각은 1.5708 과
        3.1416 을 준다. 그 결과 w = delta, neighbor_radius = 3*delta 에서 180도 떨어진
        자세는 위치가 같아도 이웃이 될 수 없어(w*pi > 3*delta) 순수 회전 쌍 552 개 중
        60.9% 만 이웃이 됐다. 지시함수에서는 100% 가 이웃 후보다.

        지시함수는 metric 이다 — 24x24 전수 검증으로 대칭성·0 대각·분리성·삼각부등식을
        확인했다.
        """
        translation_distance = float(
            np.linalg.norm(
                np.asarray(second_state.position) - np.asarray(first_state.position)
            )
        )
        is_same_rotation = _to_canonical_rotation(
            tuple(first_state.rotation)
        ) == _to_canonical_rotation(tuple(second_state.rotation))
        if is_same_rotation:
            return translation_distance
        return translation_distance + self.config.rotation_distance_weight

    def get_nearest_node(self, tree_nodes: List[_TreeNode], query_state: State) -> int:
        """트리에서 질의 상태에 metric 상 가장 가까운 노드의 인덱스를 반환한다."""
        nearest_index = 0
        nearest_distance = float("inf")
        for node_index, node in enumerate(tree_nodes):
            distance = self.get_state_distance(node.state, query_state)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = node_index
        return nearest_index

    def get_neighbor_indices(
        self, tree_nodes: List[_TreeNode], query_state: State
    ) -> List[int]:
        """질의 상태로부터 neighbor_radius 이내에 있는 트리 노드 인덱스 목록을 반환한다."""
        neighbor_indices: List[int] = list()
        for node_index, node in enumerate(tree_nodes):
            if self.get_state_distance(node.state, query_state) <= self.config.neighbor_radius:
                neighbor_indices.append(node_index)
        return neighbor_indices

    def sample_random_state(self) -> State:
        """샘플링 경계 안에서 무작위 상태를 만든다(목표 편향 포함).

        position 은 절대 위치가 아니라 이동량이므로, 부품의 시작 월드 AABB 가 확장된 샘플링
        영역([sampling_lower, sampling_upper]) 안에 놓이도록 축별 이동량 범위를 잡아 샘플링한다.
        """
        if self.random_generator.random() < self.config.goal_sample_rate:
            return self.get_goal_biased_state()

        translation_lower_bound = self.sampling_lower_bound - self.moving_start_lower_bound
        translation_upper_bound = self.sampling_upper_bound - self.moving_start_upper_bound
        sampled_position = self.random_generator.uniform(
            translation_lower_bound, translation_upper_bound
        )
        return State(
            position=tuple(sampled_position),
            rotation=self.sample_rotation(),
        )

    def sample_rotation(self) -> Tuple[int, int, int]:
        """자세를 샘플링한다. 설정에 따라 무작위 고유 자세 또는 시작 자세를 반환한다."""
        if not self.config.does_sample_rotation:
            return self.start_state.rotation
        rotation_index = int(self.random_generator.integers(len(_CANONICAL_ROTATION_EULERS)))
        return _CANONICAL_ROTATION_EULERS[rotation_index]

    def get_goal_biased_state(self) -> State:
        """부품을 조립체 밖으로 곧게 밀어낸 목표 영역 쪽 상태를 만든다.

        무작위 축과 방향(+/-)을 하나 골라, 시작(조립) 위치에서 '그 축으로만' 조립체 AABB 를
        벗어난 위치를 만든다. 나머지 두 좌표는 시작 위치 값을 그대로 유지하고 자세도 시작
        자세로 고정한다. 즉 축정렬 직선 뽑기(translational pull-out)를 목표로 편향한다.

        스냅 핏(꽉 끼워 맞춘) 조립에서는 충돌 없이 움직일 수 있는 방향이 축에 정렬된 좁은
        통로뿐인 경우가 많다. 다른 두 좌표까지 무작위로 흩뿌리면 대각선 목표가 되어
        steer_toward 가 대각선으로 나아가다 즉시 충돌하고 통로를 따라 트리가 자라지 못한다.
        따라서 목표 편향 샘플은 대각선이 아니라 축정렬 직선이어야 한다(전체 공간 탐색은
        목표 편향이 아닌 균일 샘플이 담당한다).

        나아가, 시작 상태에서 한 스텝이라도 곧게 밀 수 있었던 방향(free_escape_axis_directions)이
        있으면 그 방향들 중에서만 축·부호를 골라 편향한다. 여섯 방향을 균일하게 고르면
        막힌 방향(스냅 핏에서 대부분)으로 목표를 제안해 확장이 첫 스텝에서 무너지기 때문이다.
        빠져나갈 통로가 실제로 있는 방향으로 목표를 집중시켜 좁은 축 통로 탐색을 가속한다.
        """
        if len(self.free_escape_axis_directions) > 0:
            choice_index = int(
                self.random_generator.integers(len(self.free_escape_axis_directions))
            )
            axis_index, sign = self.free_escape_axis_directions[choice_index]
        else:
            axis_index = int(self.random_generator.integers(3))
            sign = 1 if bool(self.random_generator.integers(2)) else -1

        # 목표는 '이동량'으로 계산한다. is_goal_reached 의 역산: 부품 시작 AABB 를 그 축으로
        # 얼마나 밀어야 장애물 AABB 에서 removal_clearance 만큼 분리되는지 구한다.
        assembly_span = (
            self.obstacle_bounding_box[1, axis_index]
            - self.obstacle_bounding_box[0, axis_index]
        )
        moving_span = (
            self.moving_start_upper_bound[axis_index]
            - self.moving_start_lower_bound[axis_index]
        )
        overshoot = self.random_generator.uniform(0.0, assembly_span + moving_span)

        sampled_position = np.asarray(self.start_state.position, dtype=float).copy()
        if sign > 0:
            clearing_translation = (
                self.obstacle_bounding_box[1, axis_index]
                + self.config.removal_clearance
                - self.moving_start_lower_bound[axis_index]
            )
            sampled_position[axis_index] = (
                self.start_state.position[axis_index] + clearing_translation + overshoot
            )
        else:
            clearing_translation = (
                self.obstacle_bounding_box[0, axis_index]
                - self.config.removal_clearance
                - self.moving_start_upper_bound[axis_index]
            )
            sampled_position[axis_index] = (
                self.start_state.position[axis_index] + clearing_translation - overshoot
            )
        return State(
            position=tuple(sampled_position),
            rotation=self.start_state.rotation,
        )

    def steer_toward(self, from_state: State, to_state: State) -> State:
        """from_state 에서 to_state 방향으로 최대 step 만큼 나아간 새 상태를 만든다.

        자세는 목표 자세를 그대로 채택하고(이산 도약), 위치는 step 이내로 전진한다.
        """
        from_position = np.asarray(from_state.position, dtype=float)
        to_position = np.asarray(to_state.position, dtype=float)
        difference = to_position - from_position
        distance = float(np.linalg.norm(difference))

        if distance <= self.config.translation_step_size:
            steered_position = to_position
        else:
            steered_position = (
                from_position
                + difference / distance * self.config.translation_step_size
            )
        return State(position=tuple(steered_position), rotation=to_state.rotation)

    def is_goal_reached(self, state: State) -> bool:
        """부품이 이 상태에서 조립체 밖으로 추출되었는지(AABB 분리) 여부를 반환한다."""
        transformation_matrix = state.to_transformation_matrix()
        world_vertices = self._to_world_vertices(
            self.moving_vertices_local, transformation_matrix
        )
        moving_lower_bound = world_vertices.min(axis=0)
        moving_upper_bound = world_vertices.max(axis=0)

        for axis_index in range(3):
            positive_gap = (
                self.obstacle_bounding_box[0, axis_index] - moving_upper_bound[axis_index]
            )
            negative_gap = (
                moving_lower_bound[axis_index] - self.obstacle_bounding_box[1, axis_index]
            )
            if max(positive_gap, negative_gap) >= self.config.removal_clearance:
                return True
        return False

    def has_any_feasible_first_move(self) -> bool:
        """[Tier 0] 시작 상태에서 한 스텝이라도 움직일 수 있는지 값싸게 판정한다.

        여섯 축정렬 병진(free_escape_axis_directions, 생성 시 이미 계산) 중 하나라도
        있으면 True. 없으면 24개 이산 회전 각각을 부품 무게중심 기준으로 제자리 적용해
        회전 스윕이 유효한지 검사한다(회전으로만 첫 탈출이 열리는 경우 포착). 모두 막혀
        있으면 False — 이 상태에서는 어떤 단위 동작으로도 못 움직이므로 RRT* 를 돌릴 필요가
        없다(필요조건 기각). RRT* 한 번 대비 수 밀리초로 끝난다.
        """
        if len(self.free_escape_axis_directions) > 0:
            return True
        current_rotation_matrix = self.start_state.to_rotation_matrix()
        current_position = np.asarray(self.start_state.position, dtype=float)
        world_centroid = (
            current_rotation_matrix @ self.moving_centroid_local + current_position
        )
        # 24 개 이산 회전을 무게중심 고정 제자리 회전으로 배치해 회전 스윕을 검사한다
        # (회전으로만 첫 탈출이 열리는 경우 포착). 순차인 것은 의도다 — manifold3d 가
        # 불린 하나에 이미 여러 코어를 쓰므로 스레드를 더 붙이면 경쟁해서 느려진다.
        for euler in _CANONICAL_ROTATION_EULERS:
            if tuple(euler) == tuple(self.start_state.rotation):
                continue
            target_rotation_matrix = _to_grid_rotation_matrix(tuple(euler))
            next_position = world_centroid - target_rotation_matrix @ self.moving_centroid_local
            rotated_state = State(position=tuple(next_position), rotation=tuple(euler))
            try:
                if self.collision_session.is_rotation_segment_valid(
                    self.start_state, rotated_state, self.config.rotation_interpolation_count
                ):
                    return True
            except Exception:
                continue
        return False

    def try_straight_extraction(self) -> Optional[PlanningResult]:
        """[Tier 1] 시작에서 곧게 밀어 바로 추출되는지(회전 없이) 값싸게 시도한다.

        시작 상태에서 곧게 밀 수 있던 각 자유 축·부호에 대해, 그 축으로 목표 편향
        추출 위치(장애물 AABB + removal_clearance 를 넘는 이동량)까지 하나의 병진으로
        갈 수 있는지 스윕 검사한다. 성공하면 (시작, 목표) 두 상태 + 병진 1개짜리
        PlanningResult 를 즉시 반환한다. RRT* 없이 대부분의 '쉬운' 부품을 여기서 해소한다.

        스냅 핏의 좁은 축 통로에서 실제로 빠져나갈 수 있는 방향은 대개 자유 축뿐이므로,
        free_escape_axis_directions 방향만 시도한다. 실패하면 None 을 반환하고 상위에서
        풀 RRT* (Tier 2)로 넘어간다.
        """
        for axis_index, sign in self.free_escape_axis_directions:
            assembly_span = (
                self.obstacle_bounding_box[1, axis_index]
                - self.obstacle_bounding_box[0, axis_index]
            )
            moving_span = (
                self.moving_start_upper_bound[axis_index]
                - self.moving_start_lower_bound[axis_index]
            )
            if sign > 0:
                clearing_translation = (
                    self.obstacle_bounding_box[1, axis_index]
                    + self.config.removal_clearance
                    - self.moving_start_lower_bound[axis_index]
                )
            else:
                clearing_translation = (
                    self.obstacle_bounding_box[0, axis_index]
                    - self.config.removal_clearance
                    - self.moving_start_upper_bound[axis_index]
                )
            goal_position = np.asarray(self.start_state.position, dtype=float).copy()
            goal_position[axis_index] = (
                self.start_state.position[axis_index] + clearing_translation
            )
            goal_state = State(position=tuple(goal_position), rotation=self.start_state.rotation)
            if not self.is_goal_reached(goal_state):
                continue
            if not self.collision_session.is_valid_state(goal_state):
                continue
            if not self._segment_valid(
                self.start_state,
                goal_state,
                self._interpolation_count_for(self.start_state, goal_state),
            ):
                continue
            translation_action = Action(
                action_type=ActionType.TRANSLATION,
                value=tuple(
                    np.asarray(goal_state.position, dtype=float)
                    - np.asarray(self.start_state.position, dtype=float)
                ),
            )
            cost = self.get_state_distance(self.start_state, goal_state)
            return PlanningResult(
                is_success=True,
                states=(self.start_state, goal_state),
                actions=(translation_action,),
                cost=float(cost),
                iteration_count=0,
            )
        return None

    def execute_search(self) -> PlanningResult:
        """RRT* 탐색을 수행하고 분해 경로 결과를 반환한다.

        목표에 도달하는 경로를 찾으면 시작->목표 상태·동작 나열과 총 비용을 담아 반환한다.
        max_iteration_count 안에 찾지 못하면 is_success=False 결과를 반환한다.

        계층적 오라클: 풀 RRT* 를 돌리기 전에 두 값싼 단계를 먼저 시도한다.
          - [Tier 0] has_any_feasible_first_move(): 어떤 단위 동작으로도 못 움직이면
            (모든 축 병진·모든 회전이 막힘) 즉시 실패 반환. RRT* 반복 낭비를 막는다.
          - [Tier 1] try_straight_extraction(): 자유 축으로 곧게 밀어 한 번에 추출되면
            그 1-병진 경로를 즉시 반환. 대부분의 '쉬운' 부품을 RRT* 없이 해소한다.
        두 단계 모두 정확한 판정(거짓 양성 없음)이며, 여기서 결판나지 않는 애매한 소수만
        Tier 2(아래의 풀 RRT*)로 넘어간다.
        """
        if not self.has_any_feasible_first_move():
            return PlanningResult(
                is_success=False,
                states=tuple(),
                actions=tuple(),
                cost=float("inf"),
                iteration_count=0,
            )
        straight_result = self.try_straight_extraction()
        if straight_result is not None:
            return straight_result

        tree_nodes: List[_TreeNode] = [
            _TreeNode(state=self.start_state, parent_index=None, cost_from_start=0.0)
        ]
        best_goal_index: Optional[int] = None
        best_goal_cost = float("inf")

        performed_iteration_count = 0
        for iteration_index in range(self.config.max_iteration_count):
            performed_iteration_count = iteration_index + 1

            sampled_state = self.sample_random_state()
            source_index = self.get_nearest_node(tree_nodes, sampled_state)

            # 샘플 방향으로 충돌 전까지 연속 확장한다(RRT-Connect 식 탐욕 확장).
            # 좁은 축 통로에서 한 번의 반복만으로 부품을 끝까지 밀어낼 수 있다.
            for _ in range(self.config.maximum_extension_step_count):
                source_state = tree_nodes[source_index].state
                new_state = self.steer_toward(source_state, sampled_state)
                if new_state == source_state:
                    break
                if not self.collision_session.is_valid_state(new_state):
                    break
                if not self.is_edge_valid(source_state, new_state):
                    break

                new_node_index = self._insert_node(tree_nodes, source_index, new_state)

                if self.is_goal_reached(new_state):
                    if tree_nodes[new_node_index].cost_from_start < best_goal_cost:
                        best_goal_cost = tree_nodes[new_node_index].cost_from_start
                        best_goal_index = new_node_index

                # 목표 샘플에 도달했으면 확장을 멈춘다.
                if new_state == sampled_state:
                    break
                source_index = new_node_index

            # 최고 속도 우선: 첫 실현 가능 경로를 찾으면 즉시 중단한다.
            if self.config.stops_at_first_feasible_path and best_goal_index is not None:
                break

        if best_goal_index is None:
            return PlanningResult(
                is_success=False,
                states=tuple(),
                actions=tuple(),
                cost=float("inf"),
                iteration_count=performed_iteration_count,
            )

        states, actions = self.build_path(tree_nodes, best_goal_index)
        return PlanningResult(
            is_success=True,
            states=tuple(states),
            actions=tuple(actions),
            cost=best_goal_cost,
            iteration_count=performed_iteration_count,
        )

    def _insert_node(
        self, tree_nodes: List[_TreeNode], source_index: int, new_state: State
    ) -> int:
        """new_state 를 RRT* 규칙(최소비용 부모 선택 + 이웃 재배선)으로 트리에 넣고 인덱스를 반환한다.

        source_index 는 확장의 출발 노드로, 이웃이 하나도 없을 때의 기본 부모가 된다.
        """
        neighbor_indices = self.get_neighbor_indices(tree_nodes, new_state)
        parent_index, parent_cost = self._choose_parent(
            tree_nodes, neighbor_indices, source_index, new_state
        )
        new_node_index = len(tree_nodes)
        tree_nodes.append(
            _TreeNode(
                state=new_state,
                parent_index=parent_index,
                cost_from_start=parent_cost,
            )
        )
        self._rewire_neighbors(tree_nodes, neighbor_indices, new_node_index)
        return new_node_index

    def _choose_parent(
        self,
        tree_nodes: List[_TreeNode],
        neighbor_indices: List[int],
        nearest_index: int,
        new_state: State,
    ) -> Tuple[int, float]:
        """이웃 중 new_state 로의 유효 간선이 있고 누적 비용이 최소인 부모를 고른다."""
        best_parent_index = nearest_index
        best_cost = tree_nodes[nearest_index].cost_from_start + self.get_state_distance(
            tree_nodes[nearest_index].state, new_state
        )
        for neighbor_index in neighbor_indices:
            candidate_cost = tree_nodes[neighbor_index].cost_from_start + self.get_state_distance(
                tree_nodes[neighbor_index].state, new_state
            )
            if candidate_cost < best_cost and self.is_edge_valid(
                tree_nodes[neighbor_index].state, new_state
            ):
                best_cost = candidate_cost
                best_parent_index = neighbor_index
        return best_parent_index, best_cost

    def _rewire_neighbors(
        self,
        tree_nodes: List[_TreeNode],
        neighbor_indices: List[int],
        new_node_index: int,
    ) -> None:
        """new_node 를 거치면 더 싼 이웃들을 new_node 의 자식으로 재배선한다."""
        new_node = tree_nodes[new_node_index]
        for neighbor_index in neighbor_indices:
            if neighbor_index == new_node.parent_index:
                continue
            candidate_cost = new_node.cost_from_start + self.get_state_distance(
                new_node.state, tree_nodes[neighbor_index].state
            )
            if candidate_cost < tree_nodes[neighbor_index].cost_from_start and self.is_edge_valid(
                new_node.state, tree_nodes[neighbor_index].state
            ):
                tree_nodes[neighbor_index].parent_index = new_node_index
                tree_nodes[neighbor_index].cost_from_start = candidate_cost

    def build_path(
        self, tree_nodes: List[_TreeNode], goal_index: int
    ) -> Tuple[List[State], List[Action]]:
        """목표 노드에서 부모를 거슬러 올라가 시작->목표 상태·동작 나열을 만든다."""
        state_path: List[State] = list()
        node_index: Optional[int] = goal_index
        while node_index is not None:
            state_path.append(tree_nodes[node_index].state)
            node_index = tree_nodes[node_index].parent_index
        state_path.reverse()

        action_path: List[Action] = list()
        for segment_index in range(len(state_path) - 1):
            for action, _ in self.get_edge_actions(
                state_path[segment_index], state_path[segment_index + 1]
            ):
                action_path.append(action)
        return state_path, action_path

    def to_assembly_path(
        self, disassembly_result: PlanningResult
    ) -> Tuple[List[State], List[Action]]:
        """분해 결과를 뒤집어 조립 경로(상태·동작 나열)를 만든다.

        분해 상태 나열을 역순으로 놓고 이웃 상태 사이의 동작을 다시 계산해, 조립 방향의
        상태·동작 나열을 반환한다.
        """
        if not disassembly_result.is_success:
            raise PlanningException("cannot build an assembly path from a failed disassembly result")

        assembly_state_path = list(reversed(disassembly_result.states))
        assembly_action_path: List[Action] = list()
        for segment_index in range(len(assembly_state_path) - 1):
            for action, _ in self.get_edge_actions(
                assembly_state_path[segment_index], assembly_state_path[segment_index + 1]
            ):
                assembly_action_path.append(action)
        return assembly_state_path, assembly_action_path
