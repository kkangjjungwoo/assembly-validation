"""분해/조립 경로 탐색 결과를 msgpack 바이너리로 내보내는 모듈.

출력 구조는 README.md 의 Output Structure 를 그대로 따른다.

    metadata
        step_path          : 입력 STEP 파일 경로
        global_bbox        : 조립체 전체의 월드 축정렬 경계상자 {min:[x,y,z], max:[x,y,z]}
    solids
        <solid_id>
            name           : 부품 이름. STEP 에 저장된 이름을 쓰고, 얻지 못하면
                             형상 해시 6자리를 이름으로 쓴다(loader 가 대체한다).
                             이름이 겹치면 '이름#1', '이름#2' 로 구분한다.
                             solid_names 를 주지 않으면 이 키가 없다(하위호환).
            conversion     : 메쉬 변환(봉합) 결과. "성공" 이면 간섭 판정과 경로 탐색에
                             쓰인 부품이고, "실패" 면 형상만 저장된 부품이다(시각화에서
                             부품이 사라지지 않게 담되 탐색과 장애물에서 제외된다).
                             conversion_results 를 주지 않으면 이 키가 없다.
            mesh
                vertices   : 정점 좌표 목록 [[x,y,z], ...]
                faces      : 삼각형 정점 인덱스 목록 [[i,j,k], ...]
            state
                position   : 조립 상태 위치 [x,y,z]
                rotation   : 조립 상태 회전 [rx,ry,rz] (각 성분 {0,90,180,270})

    키는 위 순서대로 저장한다(msgpack 맵은 삽입 순서를 보존한다).
    trajectories
        <step_index>
            solid          : 이 단계에서 움직이는 부품의 식별자
            state
                position   : 동작 적용 후 도달한 상태의 위치
                rotation   : 동작 적용 후 도달한 상태의 회전
            action
                type       : "translation" | "rotation"
                value      : 동작 값 [x,y,z]

solids 와 trajectories 는 정수 키 맵으로 저장한다(solids 는 부품 식별자, trajectories 는
0 부터의 단계 순번). 부품 식별자는 연속이 아닐 수 있으므로 리스트가 아닌 맵으로 둔다.
"""

from typing import Dict, List, Sequence, Tuple, Optional

import msgpack
import numpy as np
import trimesh

from core.action import Action
from core.state import State


class TrajectorySerializationException(Exception):
    """내보내기/읽기 입력이 규약을 벗어나거나 직렬화가 실패할 때 발생."""


# 하나의 궤적 단계: (움직인 부품 식별자, 동작 적용 후 도달 상태, 적용한 동작).
TrajectoryStep = Tuple[int, State, Action]


class MsgpackTrajectorySerializer:
    """조립체 형상·상태·경로를 README 출력 구조의 msgpack 바이너리로 직렬화한다."""

    def __init__(
        self,
        step_path: str,
        solid_meshes: Dict[int, trimesh.Trimesh],
        assembled_states: Dict[int, State],
        extra_metadata: Optional[Dict[str, object]] = None,
        solid_names: Optional[Dict[int, str]] = None,
        conversion_results: Optional[Dict[int, str]] = None,
    ) -> None:
        """직렬화 대상 조립체를 받는다.

        Args:
            step_path: 입력 STEP 파일 경로(메타데이터에 기록).
            solid_meshes: 부품 식별자 -> 로컬 좌표계 Trimesh 형상.
            assembled_states: 부품 식별자 -> 조립(시작) 상태 자세.
            solid_names: 부품 식별자 -> 부품 이름. 부품 번호는 loader 반환 순서라 실행마다
                달라지지만 이름은 CAD 원본의 것이다. 주지 않으면 name 키를 쓰지 않는다.
            conversion_results: 부품 식별자 -> 메쉬 변환 결과 문자열. 봉합
                (watertight)에 성공했는지를 solids 항목의 name 다음 키로 기록한다.
                실패한 몸체도 형상은 저장되므로(시각화용) 이 키가 없으면 읽는 쪽이
                판정에 쓸 수 있는 몸체인지 알 수 없다. 주지 않으면 키를 쓰지 않는다.
            extra_metadata: metadata 에 추가로 기록할 항목. 기존 키(step_path,
                global_bbox)는 덮어쓰지 않는다.

        Raises:
            TrajectorySerializationException: solid_meshes 와 assembled_states 의
                부품 식별자 집합이 일치하지 않을 때.
        """
        if set(solid_meshes.keys()) != set(assembled_states.keys()):
            raise TrajectorySerializationException(
                "solid_meshes and assembled_states must share the same solid identifiers"
            )
        self.step_path = step_path
        self.extra_metadata = dict(extra_metadata) if extra_metadata else dict()
        self.solid_names = dict(solid_names) if solid_names else dict()
        self.conversion_results = (
            dict(conversion_results) if conversion_results else dict()
        )
        self.solid_meshes = solid_meshes
        self.assembled_states = assembled_states
        self.global_bounding_box = self.compute_global_bounding_box(
            solid_meshes, assembled_states
        )

    @staticmethod
    def compute_global_bounding_box(
        solid_meshes: Dict[int, trimesh.Trimesh],
        states: Dict[int, State],
    ) -> Dict[str, List[float]]:
        """조립체 전체를 월드 좌표로 변환했을 때의 축정렬 경계상자를 계산한다.

        Returns:
            {"min": [x, y, z], "max": [x, y, z]} 형태의 딕셔너리.
        """
        if not solid_meshes:
            raise TrajectorySerializationException(
                "solid_meshes must contain at least one solid to compute a bounding box"
            )
        lower_bounds: List[np.ndarray] = list()
        upper_bounds: List[np.ndarray] = list()
        for solid_id, mesh in solid_meshes.items():
            transformation_matrix = states[solid_id].to_transformation_matrix()
            local_vertices = np.asarray(mesh.vertices, dtype=float)
            homogeneous_vertices = np.hstack(
                [local_vertices, np.ones((len(local_vertices), 1))]
            )
            world_vertices = (transformation_matrix @ homogeneous_vertices.T).T[:, :3]
            lower_bounds.append(world_vertices.min(axis=0))
            upper_bounds.append(world_vertices.max(axis=0))
        global_lower = np.min(lower_bounds, axis=0)
        global_upper = np.max(upper_bounds, axis=0)
        return {
            "min": [float(value) for value in global_lower],
            "max": [float(value) for value in global_upper],
        }

    @staticmethod
    def _to_state_dictionary(state: State) -> Dict[str, List]:
        """State 를 README 의 state 하위 구조 딕셔너리로 변환한다."""
        return {
            "position": [float(coordinate) for coordinate in state.position],
            "rotation": [int(angle) for angle in state.rotation],
        }

    @staticmethod
    def _to_action_dictionary(action: Action) -> Dict[str, object]:
        """Action 을 README 의 action 하위 구조 딕셔너리로 변환한다."""
        if action.is_translation():
            value = [float(component) for component in action.value]
        else:
            value = [int(component) for component in action.value]
        return {"type": action.action_type.value, "value": value}

    @staticmethod
    def _to_mesh_dictionary(mesh: trimesh.Trimesh) -> Dict[str, List]:
        """Trimesh 를 README 의 mesh 하위 구조(정점·면 목록) 딕셔너리로 변환한다."""
        return {
            "vertices": np.asarray(mesh.vertices, dtype=float).tolist(),
            "faces": np.asarray(mesh.faces, dtype=int).tolist(),
        }

    def to_output_dictionary(
        self, trajectory_steps: Sequence[TrajectoryStep]
    ) -> Dict[str, object]:
        """README 출력 구조에 맞는 중첩 딕셔너리를 만든다.

        Args:
            trajectory_steps: (부품 식별자, 도달 상태, 적용 동작) 튜플의 순서열.
                일반적으로 planner 의 조립 경로(분해 경로의 역순)를 부품별로 이어 붙인
                것이며, 리스트의 순서가 곧 조립 단계 순서가 된다.

        Raises:
            TrajectorySerializationException: 궤적 단계가 참조하는 부품 식별자가
                solid_meshes 에 없을 때.
        """
        solids_output: Dict[int, object] = dict()
        for solid_id, mesh in self.solid_meshes.items():
            # 키 순서는 사람이 읽는 순서다 — 이름으로 부품을 찾고, 그 부품이 판정에
            # 쓰였는지(conversion) 본 다음 형상과 상태를 본다.
            entry: Dict[str, object] = dict()
            # 이름을 모르는 부품은 키 자체를 넣지 않는다 — 빈 문자열을 넣으면 읽는 쪽이
            # '이름이 빈 부품' 과 '이름을 못 얻은 부품' 을 구분하지 못한다.
            if solid_id in self.solid_names:
                entry["name"] = str(self.solid_names[solid_id])
            if solid_id in self.conversion_results:
                entry["conversion"] = str(self.conversion_results[solid_id])
            entry["mesh"] = self._to_mesh_dictionary(mesh)
            entry["state"] = self._to_state_dictionary(self.assembled_states[solid_id])
            solids_output[solid_id] = entry

        trajectories_output: Dict[int, object] = dict()
        for step_index, (solid_id, state, action) in enumerate(trajectory_steps):
            if solid_id not in self.solid_meshes:
                raise TrajectorySerializationException(
                    f"trajectory step {step_index} references unknown solid {solid_id}"
                )
            trajectories_output[step_index] = {
                "solid": int(solid_id),
                "state": self._to_state_dictionary(state),
                "action": self._to_action_dictionary(action),
            }

        metadata_output = {
            "step_path": self.step_path,
            "global_bbox": self.global_bounding_box,
        }
        # 추가 항목은 기존 키를 덮어쓰지 않는다(포맷 하위호환).
        for key, value in self.extra_metadata.items():
            if key not in metadata_output:
                metadata_output[key] = value

        return {
            "metadata": metadata_output,
            "solids": solids_output,
            "trajectories": trajectories_output,
        }

    def serialize_to_binary(
        self, trajectory_steps: Sequence[TrajectoryStep]
    ) -> bytes:
        """출력 구조를 msgpack 바이너리로 직렬화해 반환한다."""
        output_dictionary = self.to_output_dictionary(trajectory_steps)
        return msgpack.packb(output_dictionary, use_bin_type=True)

    def write_to_file(
        self, trajectory_steps: Sequence[TrajectoryStep], output_path: str
    ) -> None:
        """출력 구조를 msgpack 바이너리로 직렬화해 파일에 기록한다."""
        binary_data = self.serialize_to_binary(trajectory_steps)
        with open(output_path, "wb") as output_file:
            output_file.write(binary_data)

    @staticmethod
    def deserialize_from_binary(binary_data: bytes) -> Dict[str, object]:
        """msgpack 바이너리를 README 출력 구조 딕셔너리로 역직렬화한다.

        정수 키 맵(solids, trajectories)을 그대로 복원하기 위해 strict_map_key 를
        끈다. 반환 구조는 직렬화된 포맷을 그대로 반영한다(State/Action 객체로
        재구성하지 않는다).
        """
        return msgpack.unpackb(binary_data, raw=False, strict_map_key=False)

    @staticmethod
    def read_from_file(input_path: str) -> Dict[str, object]:
        """msgpack 파일을 읽어 README 출력 구조 딕셔너리로 반환한다."""
        with open(input_path, "rb") as input_file:
            binary_data = input_file.read()
        return MsgpackTrajectorySerializer.deserialize_from_binary(binary_data)
