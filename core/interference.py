"""간섭 길이 기반 충돌 판정 — 이 프로젝트의 근본 판정기.

조립된 STEP 은 설계상 부품이 서로 맞물려(압입·나사산·삽입) 초기 상태에서 이미 겹쳐
있다. 따라서 '겹침이 있으면 충돌' 은 시작 상태부터 기각한다. 겹침의 양을 재고 그것이
조립 상태보다 늘어났는지를 본다.

겹침의 양을 재는 대리 지표들은 전부 실패했다(각각 실측, 재시도 금지):
  FCL 관통 깊이   : 오목 껍데기에서 의미를 잃는다 — 간섭 0 인 자세에서 depth 14.1
  표면 voxel 개수 : 표면만 세므로 겹침 부피와 단조 대응하지 않는다
  삼각형 접촉 개수: 더 밀어 넣으면 깊이는 늘어나는데 개수는 줄어든다
  고체 voxel 채움 : 부피를 1.2~7.9 배 과대추정한다
  볼록 분해       : 조각 합집합이 원본의 1.9~3.9 배 — 부풀린 baseline 이 관통을 가려
                    산출 궤적에서 4 개 부품이 재질을 통과했다

겹침 부피는 유일하게 충실하다 — 떨어져 있으면 정확히 0, 겹칠수록 단조 증가한다.
manifold3d 메쉬 불린으로 계산하며 B-rep 불린 대비 309 배 빠르고(쌍당 0.0014 초)
겹치는 쌍에서 비율 중앙값 1.000, 떨어진 쌍 38/38 정확히 0 이다.

판정 단위는 부피가 아니라 그 세제곱근인 '간섭 길이' L = V^(1/3) 다 — 근거는
interference_length docstring 에 있다.

핵심 전제는 위상이 닫힌 메쉬다. 면마다 독립 삼각분할하면 공유 경계에서 정점이 중복되어
위상이 찢어지고 manifold3d 가 빈 Manifold(부피 0)를 만든다. 그래서 solid 를 한 번에
삼각분할하고 좌표로 전역 병합한다(triangulate_solid_preserving_topology).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class InterferenceException(Exception):
    """간섭 부피 판정기를 구성하거나 사용할 수 없을 때 발생한다."""


def triangulate_solid_preserving_topology(
    shape,
    linear_deflection: float = 0.02,
    angular_deflection: float = 0.5,
    merge_digits: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """B-rep solid 를 위상이 보존된 삼각형 메쉬로 변환한다.

    solid 전체를 한 번에 삼각분할한 뒤 면마다 나오는 노드를 좌표로 전역 병합한다. OCC 는
    인접 면이 공유 edge 의 노드를 공유하므로 좌표 병합만으로 면 경계가 이어져 위상이
    닫힌다. 면 방향(REVERSED)을 반영해 winding 을 맞춘다.

    Args:
        shape: TopoDS_Shape (solid).
        linear_deflection: 삼각분할 선형 허용오차.
        angular_deflection: 삼각분할 각도 허용오차(라디안).
        merge_digits: 정점 병합에 쓸 좌표 반올림 자릿수.

    Returns:
        (정점 배열 (N,3), 삼각형 인덱스 배열 (M,3)).

    gmsh 로 대체하지 않는 이유(15 부품 x 3 파일, B-rep 참값 대조): gmsh 최선 설정이
    부피오차 중앙 0.317% / 최대 7.556% 로 BRepMesh(0.077% / 2.704%)보다 나쁘면서 삼각형
    4 배·시간 10 배를 쓴다. 기각된 크기 규칙 — gmsh 기본값(부피오차 34%), 대각선 비례
    D/80, 표면적 비례.
    """
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods

    breptools.Clean(shape)
    BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)

    vertices: List[List[float]] = list()
    faces: List[List[int]] = list()
    index_of_coordinate: Dict[Tuple[float, float, float], int] = dict()

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is not None:
            transformation = location.Transformation()
            is_reversed = face.Orientation() == TopAbs_REVERSED
            local_indices: List[int] = list()
            for node_number in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(node_number).Transformed(transformation)
                coordinate = (
                    round(point.X(), merge_digits),
                    round(point.Y(), merge_digits),
                    round(point.Z(), merge_digits),
                )
                global_index = index_of_coordinate.get(coordinate)
                if global_index is None:
                    global_index = len(vertices)
                    index_of_coordinate[coordinate] = global_index
                    vertices.append([point.X(), point.Y(), point.Z()])
                local_indices.append(global_index)
            for triangle_number in range(1, triangulation.NbTriangles() + 1):
                first, second, third = triangulation.Triangle(triangle_number).Get()
                a = local_indices[first - 1]
                b = local_indices[second - 1]
                c = local_indices[third - 1]
                if a == b or b == c or a == c:
                    continue
                faces.append([a, c, b] if is_reversed else [a, b, c])
        explorer.Next()

    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def _build_manifold(vertices: np.ndarray, faces: np.ndarray, minimum_volume: float):
    """정점·삼각형에서 manifold3d Manifold 를 만든다. 위상이 열려 있으면 None."""
    import manifold3d

    try:
        mesh = manifold3d.Mesh(
            vert_properties=np.asarray(vertices, dtype=np.float32),
            tri_verts=np.asarray(faces, dtype=np.uint32),
        )
        manifold = manifold3d.Manifold(mesh)
    except Exception:
        return None
    if manifold.volume() <= minimum_volume:
        return None
    return manifold


def _weld_vertices(vertices: np.ndarray, faces: np.ndarray, tolerance: float):
    """좌표가 같은 정점을 통합하고, 퇴화·중복 삼각형을 버린다.

    STEP 을 면 단위로 삼각분할하면 인접한 두 면이 만나는 모서리에서 정점이 따로 생성된다.
    좌표는 같은데 번호가 달라 위상이 이어지지 않으므로 먼저 통합한다.
    """
    keys = np.round(vertices / tolerance) * tolerance if tolerance > 0 else vertices
    _, first_index, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    new_vertices = vertices[first_index]
    new_faces = inverse[faces]
    keep = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 2] != new_faces[:, 0])
    )
    new_faces = new_faces[keep]
    sorted_key = np.sort(new_faces, axis=1)
    _, unique_index = np.unique(sorted_key, axis=0, return_index=True)
    return new_vertices, new_faces[np.sort(unique_index)]


def _drop_nonmanifold_faces(vertices: np.ndarray, faces: np.ndarray):
    """한 변을 삼각형 3개 이상이 공유하면 면적이 작은 것부터 버려 2개로 맞춘다.

    manifold3d 는 변마다 정확히 두 삼각형을 요구한다. 어느 것을 버릴지는 형상 기여가
    가장 작은 것 — 면적 기준 — 으로 정한다. ca1a77(껍데기)에서 2 개가 제거되며, 그것이
    없으면 구멍을 메워도 닫히지 않는다(실측).
    """
    triangle = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1
    )
    edge_faces: Dict[Tuple[int, int], List[int]] = dict()
    for index, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault((u, v) if u < v else (v, u), []).append(index)
    removed = set()
    for members in edge_faces.values():
        alive = [m for m in members if m not in removed]
        if len(alive) <= 2:
            continue
        alive.sort(key=lambda index: areas[index])
        removed.update(alive[: len(alive) - 2])
    if not removed:
        return faces, 0
    keep = np.array([index not in removed for index in range(len(faces))])
    return faces[keep], len(removed)


def _unify_orientation(faces: np.ndarray):
    """인접 삼각형이 공유 변을 서로 반대 방향으로 쓰도록 방향을 맞춘다(성분마다 BFS).

    manifold3d 는 유향 half-edge 를 요구하므로 방향이 어긋난 변이 하나라도 있으면 빈
    결과를 낸다. 91ad4f 에서 12 개 면이 뒤집혀 있었다(실측).
    """
    edge_map: Dict[Tuple[int, int], List[int]] = dict()
    for index, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_map.setdefault((u, v) if u < v else (v, u), []).append(index)
    visited = np.zeros(len(faces), dtype=bool)
    result = faces.copy()
    flip_count = 0
    for seed in range(len(faces)):
        if visited[seed]:
            continue
        visited[seed] = True
        queue = [seed]
        while queue:
            current = queue.pop()
            a, b, c = result[current]
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                for neighbour in edge_map.get(key, ()):
                    if neighbour == current or visited[neighbour]:
                        continue
                    x, y, z = result[neighbour]
                    if ((x, y) == (u, v)) or ((y, z) == (u, v)) or ((z, x) == (u, v)):
                        result[neighbour] = result[neighbour][::-1]
                        flip_count += 1
                    visited[neighbour] = True
                    queue.append(neighbour)
    return result, flip_count


def _boundary_loops(faces: np.ndarray) -> List[List[int]]:
    """짝 없는 유향 변들을 이어 닫힌 고리로 만든다(=실제 구멍의 경계)."""
    directed = set()
    for a, b, c in faces:
        directed.update(((a, b), (b, c), (c, a)))
    open_edges = [(u, v) for (u, v) in directed if (v, u) not in directed]
    successor: Dict[int, List[int]] = dict()
    for u, v in open_edges:
        successor.setdefault(u, []).append(v)
    loops = []
    used = set()
    for start in open_edges:
        if start in used:
            continue
        loop = [start[0], start[1]]
        used.add(start)
        current = start[1]
        while True:
            options = [v for v in successor.get(current, []) if (current, v) not in used]
            if not options:
                break
            following = options[0]
            used.add((current, following))
            if following == loop[0]:
                break
            loop.append(following)
            current = following
            if len(loop) > 100000:
                break
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _loop_frame(vertices: np.ndarray, loop: List[int]) -> Tuple[np.ndarray, float]:
    """고리의 무게중심과 지름(가장 먼 두 점 거리 근사)."""
    points = vertices[loop]
    centre = points.mean(axis=0)
    diameter = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    return centre, diameter


def _pair_facing_loops(vertices: np.ndarray, loops: List[List[int]]):
    """벽의 안팎 면이 만드는 '마주보는 고리 쌍' 을 찾는다.

    두께가 있는 벽을 안팎 두 겹 면으로만 그리고 끝단 테두리를 빼먹은 몸체(STEP 의
    SHELL_BASED_SURFACE_MODEL 에 흔하다)에서는, 개구부마다 안쪽 고리와 바깥쪽 고리가
    한 쌍으로 나온다. 두 고리를 각각 원판으로 덮으면(star) 원판이 서로 겹쳐 부피가
    틀어진다 — 실측: 배럴 하우징이 85,396 으로 나왔는데 벽 재료 추정치는 71,776 이다
    (1.19 배 과대). 두 고리를 띠(ring)로 이어야 두께 그대로의 껍질이 된다.

    판정 기준은 형상에 독립적이다: 한 고리의 각 정점에서 다른 고리까지의 최단거리
    중앙값이 그 고리 지름보다 충분히 작으면(벽 두께 << 개구부 크기) 마주보는 쌍이다.

    Returns:
        (쌍 목록 [(고리 i, 고리 j)], 짝 없는 고리 인덱스 목록).
    """
    if len(loops) < 2:
        return [], list(range(len(loops)))

    frames = [_loop_frame(vertices, loop) for loop in loops]
    candidates = []
    for i in range(len(loops)):
        for j in range(i + 1, len(loops)):
            points_i = vertices[loops[i]]
            points_j = vertices[loops[j]]
            distances = np.linalg.norm(points_i[:, None, :] - points_j[None, :, :], axis=2)
            gap = float(np.median(distances.min(axis=1)))
            reference = min(frames[i][1], frames[j][1])
            if reference <= 0:
                continue
            # 벽 두께는 개구부 크기보다 훨씬 작다. 여유 있게 1/4 을 상한으로 둔다.
            if gap < 0.25 * reference:
                candidates.append((gap, i, j))

    candidates.sort()
    paired = set()
    pairs = []
    for _, i, j in candidates:
        if i in paired or j in paired:
            continue
        paired.update((i, j))
        pairs.append((i, j))
    unpaired = [index for index in range(len(loops)) if index not in paired]
    return pairs, unpaired


def _zip_loops(vertices: np.ndarray, first: List[int], second: List[int]):
    """두 고리를 동시에 훑으며 삼각형 띠를 만든다(지퍼).

    진행 판정은 '두 고리를 얼마나 소비했는가' 의 비율로 한다 — 덜 진행한 쪽을 먼저
    전진시킨다. 이전 판은 '다음 대각선이 짧은 쪽' 으로 정했는데, 두 고리의 시작점이
    반대편에 있으면 한쪽으로만 계속 전진해 부채꼴이 된다: 실측에서 삼각형 250 개 중
    240 개의 최장변이 5 를 넘고 최대 62.3(고리 지름)까지 갔다. 대각선 길이는 국소
    정보라 전역 진행을 보장하지 못한다.

    시작점을 맞춘 뒤 비율로 진행하면 두 고리를 나란히 소비하므로 띠가 고르게 만들어지고,
    각 고리의 변이 정확히 한 번씩 쓰여 위상이 닫힌다.
    """
    triangles = []
    i = j = 0
    total_first, total_second = len(first), len(second)
    while i < total_first or j < total_second:
        a_now = first[i % total_first]
        b_now = second[j % total_second]
        if i >= total_first:
            advance_first = False
        elif j >= total_second:
            advance_first = True
        else:
            advance_first = (i + 1) / total_first <= (j + 1) / total_second

        if advance_first:
            a_next = first[(i + 1) % total_first]
            triangles.append((a_next, a_now, b_now))
            i += 1
        else:
            b_next = second[(j + 1) % total_second]
            triangles.append((b_next, b_now, a_now))
            j += 1
    return triangles


def _bridge_loops(vertices: np.ndarray, first_loop: List[int], second_loop: List[int]):
    """두 고리를 삼각형 띠로 잇는다.

    마주보는 두 경계(안쪽 면과 바깥쪽 면)는 서로 반대 방향으로 돌아야 띠 삼각형의 방향이
    일관된다 — 두 면의 법선이 반대이기 때문이다. 두 고리의 상대 순회 방향은 형상마다
    다르므로 가정하지 않는다: 둘째 고리를 정방향·역방향 두 가지로 이어 보고 유향 변
    충돌이 적은 쪽을 택한다. 판정이 결과 자체로 이뤄지므로 형상에 독립적이다.

    (실측 근거: 방향을 가정하고 한 가지만 만들면 배럴 하우징에서 유향 변 56 개가
    충돌했고, 그 뒤 방향 통일 BFS 를 돌리면 640 면을 뒤집으며 396 개로 악화됐다.)

    Returns:
        삼각형 목록.
    """
    first = list(first_loop)
    second = list(second_loop)
    if len(first) < 3 or len(second) < 3:
        return []

    # 시작점 맞추기: 첫 고리의 0 번 정점에 가장 가까운 둘째 고리 정점부터 시작한다.
    def align(loop):
        start = int(np.argmin(np.linalg.norm(vertices[loop] - vertices[first[0]], axis=1)))
        return loop[start:] + loop[:start]

    forward = _zip_loops(vertices, first, align(second))
    backward = _zip_loops(vertices, first, align(second[::-1]))

    # 두 방향 중 띠가 짧은 쪽(=삼각형 최장변 합이 작은 쪽)을 고른다. 방향이 어긋나면
    # 띠가 꼬여 변이 길어지므로, 길이가 방향의 지표가 된다. 유향 변 충돌만으로 고르면
    # 두 방향이 동점일 때 판별이 안 된다.
    def strip_cost(triangles):
        if not triangles:
            return float("inf")
        points = vertices[np.asarray(triangles, dtype=np.int64)]
        edges = np.stack([
            np.linalg.norm(points[:, 1] - points[:, 0], axis=1),
            np.linalg.norm(points[:, 2] - points[:, 1], axis=1),
            np.linalg.norm(points[:, 0] - points[:, 2], axis=1),
        ], axis=1)
        return float(edges.max(axis=1).sum())

    return forward if strip_cost(forward) <= strip_cost(backward) else backward


def _star_cap(vertices: np.ndarray, loop: List[int], centre_index: int):
    """고리를 무게중심 정점으로 덮는다(star 삼각화)."""
    return [(loop[position], loop[(position + 1) % len(loop)], centre_index)
            for position in range(len(loop))]


def _apply_closure(vertices: np.ndarray, faces: np.ndarray, loops, pairs, unpaired):
    """주어진 짝짓기대로 띠와 덮개를 만들어 붙인다."""
    added = []
    for i, j in pairs:
        added.extend(_bridge_loops(vertices, loops[i], loops[j]))

    extra_vertices = []
    next_index = len(vertices)
    for index in unpaired:
        loop = loops[index]
        extra_vertices.append(vertices[loop].mean(axis=0).reshape(1, 3))
        added.extend(_star_cap(vertices, loop, next_index))
        next_index += 1

    if not added:
        return vertices, faces, 0
    new_vertices = np.vstack([vertices] + extra_vertices) if extra_vertices else vertices
    new_faces = np.vstack([faces, np.asarray(added, dtype=faces.dtype)])
    return new_vertices, new_faces, len(added)


def _close_boundaries(vertices: np.ndarray, faces: np.ndarray):
    """열린 경계를 닫는 후보들을 만든다 — 마주보는 고리는 띠로, 나머지는 star 로.

    같은 절차를 모든 몸체에 적용한다. 구멍이 몇 개짜리 부품이든, 두께 벽으로 그려진
    자유 SHELL 이든 형상에 따라 분기하지 않는다 — 고리 사이 간격이 고리 크기에 비해
    작은지만 본다.

    다만 그 판정이 항상 맞지는 않는다. 얇은 부품에서는 서로 다른 개구부의 고리도 가까워
    쌍으로 묶일 수 있는데(실측: Cleaner 의 5355c5 는 star 로 닫히던 것이 쌍으로 묶이자
    실패), 그때는 전부 star 로 덮는 쪽이 맞다. 그래서 후보를 둘 만들어 호출자가 닫히는
    쪽을 고르게 한다 — 형상별 예외를 두지 않고 결과로 판정한다.

    Returns:
        후보 목록 [(정점, 면, 추가 면 수, 이은 쌍 수, 덮개 수)]. 경계가 없으면 빈 목록.
    """
    loops = _boundary_loops(faces)
    if not loops:
        return []

    pairs, unpaired = _pair_facing_loops(vertices, loops)

    candidates = []
    if pairs:
        bridged_vertices, bridged_faces, added = _apply_closure(
            vertices, faces, loops, pairs, unpaired
        )
        candidates.append((bridged_vertices, bridged_faces, added, len(pairs), len(unpaired)))

    capped_vertices, capped_faces, added = _apply_closure(
        vertices, faces, loops, [], list(range(len(loops)))
    )
    candidates.append((capped_vertices, capped_faces, added, 0, len(loops)))
    return candidates


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """부호 부피. 음수면 법선이 안쪽을 향한다."""
    triangle = vertices[faces]
    return float(
        np.sum(np.einsum("ij,ij->i", triangle[:, 0], np.cross(triangle[:, 1], triangle[:, 2]))) / 6.0
    )


def repair_mesh_geometry(vertices: np.ndarray, faces: np.ndarray, minimum_volume: float):
    """메쉬를 닫아 (정점, 면, manifold, 단계 라벨) 을 돌려준다.

    repair_to_manifold 와 같은 절차지만 복구된 '기하' 를 함께 돌려준다 — STEP 을 메쉬로
    변환하는 시점에 적용해, 판정용 manifold 와 내보내는 메쉬가 같은 기하를 쓰게 하기
    위해서다. 원본 메쉬를 그대로 내보내면 구멍과 뒤집힌 법선이 시각화에 그대로 나타난다
    (실측: Hair Dryer 의 91ad4f 는 표면적의 1.138% 가 구멍이고 뒤집힌 면이 12 개라
    렌더러의 뒷면 제거로 내부가 드러난다).

    Returns:
        (vertices, faces, manifold, 라벨). manifold 가 None 이면 닫지 못한 것이며,
        그 경우에도 그때까지 정리된 기하를 돌려준다.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)

    direct = _build_manifold(vertices, faces, minimum_volume)
    if direct is not None:
        return vertices, faces, direct, "직접"

    extent = vertices.max(axis=0) - vertices.min(axis=0)
    scale = float(np.linalg.norm(extent))
    tolerance = 1e-6 * scale if scale > 0 else 0.0

    work_vertices, work_faces = _weld_vertices(vertices, faces, tolerance)
    work_faces, dropped = _drop_nonmanifold_faces(work_vertices, work_faces)
    work_faces, flips = _unify_orientation(work_faces)
    if _signed_volume(work_vertices, work_faces) < 0:
        work_faces = work_faces[:, ::-1]
    label = f"용접+비다양체{dropped}+방향{flips}"
    candidate = _build_manifold(work_vertices, work_faces, minimum_volume)
    if candidate is not None:
        return work_vertices, work_faces, candidate, label

    closures = _close_boundaries(work_vertices, work_faces)

    # 후보(띠 우선 / 전부 덮개)마다, 방향 통일한 것과 안 한 것을 시도한다. 띠는 이미
    # 방향이 맞춰져 있고 통일 BFS 가 안팎 면을 잇는 띠를 지나며 오히려 모순을 키우는
    # 반면(실측: 유향 변 충돌 56 -> 396), star 덮개는 방향이 임의라 통일이 필요하다.
    # 어느 쪽이 필요한지는 몸체마다 다르므로 결과로 판정한다.
    attempts = []
    for filled_vertices, filled_faces, added, bridged, capped in closures:
        detail = []
        if bridged:
            detail.append(f"띠{bridged}쌍")
        if capped:
            detail.append(f"덮개{capped}개")
        attempt_label = f"{label}+구멍({'+'.join(detail)}, {added}면)"
        unified_faces, _ = _unify_orientation(filled_faces)
        for candidate_faces in (filled_faces, unified_faces):
            oriented = candidate_faces
            if _signed_volume(filled_vertices, oriented) < 0:
                oriented = oriented[:, ::-1]
            candidate = _build_manifold(filled_vertices, oriented, minimum_volume)
            if candidate is not None:
                return filled_vertices, oriented, candidate, attempt_label
        attempts.append(attempt_label)
    return work_vertices, work_faces, None, (attempts[0] if attempts else label) + " 실패"


def repair_to_manifold(vertices: np.ndarray, faces: np.ndarray, minimum_volume: float):
    """부품별 설정 없이 메쉬를 닫아 manifold3d Manifold 로 만든다.

    같은 절차를 모든 부품에 적용한다. 실측(29 부품): 29/29 닫힘, 부피비 0.903~1.012,
    부품당 0.00~0.45 초.

    단계(위에서부터 필요한 만큼만):
      1) 직접 시도 — 24/29 가 여기서 통과
      2) 정점 용접(부품 크기의 1e-6) + 퇴화·중복 삼각형 제거
      3) 비다양체 변 정리 — 3 개 이상 공유하는 변에서 작은 면부터 제거
      4) 방향 통일 — 인접 삼각형 BFS
      5) 구멍 메우기 — 경계 고리마다 star 삼각화
      6) 바깥향 부호 결정

    구멍 메우기는 없던 면을 만들어 부피를 바꾼다. trimesh.fill_holes 는 큰 면을 임의로
    덮어 부피비 0.8497 로 망가뜨렸고, 여기서는 고리 단위 최소 면만 추가해 0.9784 다.

    Returns:
        (manifold, vertices, 단계 라벨). 닫지 못하면 manifold 가 None 이다.
    """
    repaired_vertices, _, manifold, label = repair_mesh_geometry(vertices, faces, minimum_volume)
    return manifold, repaired_vertices, label


def interference_length(volume: float) -> float:
    """겹침 부피에서 간섭의 특성 길이를 만든다: L = V^(1/3).

    부피는 L^3 로 스케일하므로 같은 임계값이 부품 크기에 따라 다른 물리 현상을 뜻한다.
    실측(막히는 자세 30개): 부피는 맞물림 깊이가 아니라 접촉 면적을 재고 있다 —
    log V ~ log A 기울기 1.06(상관 0.948), log V ~ log t 는 상관 0.464. 막히는 지점의
    값 산포가 부피 3,211배, 길이 14.8배다.

    두께 2V/A 를 쓰지 않는 이유: manifold3d 의 교집합 표면적이 인자 순서에 의존한다(같은
    쌍에서 부피는 346.8234 로 동일한데 면적이 2.276배 갈린다). 교집합을 다시 삼각분할해
    직접 적분해도 같은 값이므로 면적 계산이 아니라 교집합 메쉬가 방향에 따라 중복/내부
    면을 포함한다. 조립 18쌍의 순서 절대차:

        V^(1/3)            0.0000   <- 채택
        겹침 부피           0.0058
        2V/A               0.3795   (값 규모의 56%)
        부피/최대단면        0.2119   (22.8%)
        교집합 AABB 최소변   8.0000   (68.8%)

    비대칭한 기준은 보정 경로와 판정 경로가 다른 값을 얻어, 기준선이 크면 관통을 허가하고
    작으면 조립 상태부터 막는다.

    한계: V^(1/3) 은 등방 덩어리의 특성 길이이므로 넓고 얇은 접촉에서는 실제 침투 깊이보다
    큰 값을 낸다. 그 대신 순서 대칭이고 재현 가능하다 — 하이퍼파라미터로 조절하기 위한
    전제다.
    """
    if volume <= 0.0:
        return 0.0
    return float(volume) ** (1.0 / 3.0)


def _sum_pairwise_intersection_volume(moving_solids: List[object], obstacle_solids: List[object]) -> float:
    """두 Manifold 목록 사이의 겹침 부피 합.

    조각마다 AABB 로 먼저 걸러 불린 호출을 줄인다. 볼록 분해에서는 조각이 수십~수백
    개이므로 이 사전 필터가 속도를 지배한다.
    """
    total = 0.0
    obstacle_boxes = [solid.bounding_box() for solid in obstacle_solids]
    for moving in moving_solids:
        moving_box = moving.bounding_box()
        for obstacle, obstacle_box in zip(obstacle_solids, obstacle_boxes):
            if (
                moving_box[3] < obstacle_box[0]
                or obstacle_box[3] < moving_box[0]
                or moving_box[4] < obstacle_box[1]
                or obstacle_box[4] < moving_box[1]
                or moving_box[5] < obstacle_box[2]
                or obstacle_box[5] < moving_box[2]
            ):
                continue
            total += float((moving ^ obstacle).volume())
    return total


def _transform_manifold(manifold, transformation_matrix: np.ndarray):
    """4x4 변환을 Manifold 에 적용한다. manifold3d 는 3x4 (행 우선)를 받는다."""
    matrix = np.asarray(transformation_matrix, dtype=float)
    return manifold.transform(matrix[:3, :4])


def _baseline_pair_worker(checker, pairs, transformations, queue) -> None:
    """fork 자식에서 쌍별 간섭 깊이를 계산해 큐로 돌려준다(OCC 형상은 상속으로 전달).

    실패한 쌍은 값을 만들지 않고 사유와 함께 돌려준다 — 실패를 0.0 으로 바꾸면 그 쌍의
    기준선이 사라져, 설계상 맞물린 쌍이 '조립 상태에서 간섭 0' 으로 취급되고 부품이
    영원히 못 빠진다. 부모가 이 사유를 보고 예외를 던진다.
    """
    results = list()
    for first, second in pairs:
        try:
            depth = checker.interference_length_between(
                first, transformations[first], second, transformations[second]
            )
        except Exception as failure:
            results.append(((first, second), None, f"{type(failure).__name__}: {failure}"))
            continue
        results.append(((first, second), float(depth), None))
    queue.put(results)


def _run_baseline_pairs_parallel(checker, pairs, transformations, worker_count: int):
    """쌍 목록을 fork 워커로 분산 계산한다."""
    import multiprocessing

    context = multiprocessing.get_context("fork")
    count = max(1, min(worker_count, len(pairs)))
    chunks = [pairs[index::count] for index in range(count)]
    queue = context.Queue()
    processes = list()
    for chunk in chunks:
        if not chunk:
            continue
        process = context.Process(
            target=_baseline_pair_worker, args=(checker, chunk, transformations, queue)
        )
        process.start()
        processes.append(process)
    collected = dict()
    failures = list()
    for _ in processes:
        for key, value, reason in queue.get():
            if reason is not None:
                failures.append((key, reason))
                continue
            collected[key] = value
    for process in processes:
        process.join()
    if failures:
        detail = "; ".join(f"{pair}: {reason}" for pair, reason in failures[:5])
        raise InterferenceException(
            f"baseline calibration failed for {len(failures)} pair(s): {detail}"
        )
    return collected


@dataclass(frozen=True)
class InterferenceBudget:
    """부품 쌍마다 허용되는 간섭 길이의 상한 = 조립 상태 기준선 + margin.

    곱셈 배율은 쓰지 않는다 — baseline 이 큰 쌍에 그 baseline 만큼의 추가 관통을 허가해
    부품이 재질을 통과한다(실측: 배율 2.0 에서 Cleaner 13/13 이지만 관통 4건).

    baseline 도 길이로 보관한다 — 부피 baseline 에 길이 margin 을 더할 수는 없다. 단위
    선정 근거는 interference_length docstring 에 있다.
    """

    baseline_lengths: Dict[int, Dict[int, float]]
    margin: float

    def limit_for(self, moving_solid_id: int, obstacle_solid_id: int) -> float:
        """이 쌍에서 허용되는 간섭 깊이 상한."""
        baseline = self.baseline_lengths.get(moving_solid_id, dict()).get(obstacle_solid_id, 0.0)
        return baseline + self.margin


class InterferenceVolumeChecker:
    """간섭의 특성 길이 L = V^(1/3) 로 충돌을 판정한다(부피는 길이를 만드는 중간값).

    부품마다 위상 보존 삼각분할로 닫힌 메쉬를 만들고 메쉬 불린으로 간섭 부피를 잰다.
    닫히지 못한 부품은 판정에서 제외하고 사유를 남긴다(excluded_solid_ids).

    AABB 사전 필터를 항상 먼저 적용한다 — 경계 상자가 떨어져 있으면 간섭은 0 이므로
    불린을 아예 부르지 않는다. 실제 탐색에서 대부분의 질의가 여기서 끝난다.
    """

    def __init__(
        self,
        solid_shapes: Dict[int, object],
        linear_deflection: float = 0.02,
        angular_deflection: float = 0.5,
        merge_digits: int = 6,
        minimum_manifold_volume: float = 1.0,
    ):
        """STEP 의 B-rep solid 들로 판정기를 구성한다.

        부품마다 위상 보존 삼각분할 -> 일반 복구(용접·비다양체 정리·방향 통일·경계 닫기)
        순서로 닫힌 메쉬를 만든다. 닫히지 못한 부품은 판정에서 빼고 사유를 남긴다.

        Args:
            solid_shapes: 부품 번호 -> TopoDS_Shape (STEP 원본 solid).
            linear_deflection: 삼각분할 선형 허용오차. 곡률이 큰 면이 자동으로 촘촘해진다.
            angular_deflection: 삼각분할 각도 허용오차.
            merge_digits: 정점 병합 좌표 반올림 자릿수.
            minimum_manifold_volume: 닫힌 메쉬로 인정할 최소 부피.

        Raises:
            InterferenceException: 부품이 하나도 없거나 전부 제외된 경우.
        """
        if not solid_shapes:
            raise InterferenceException("at least one solid is required")

        self.solid_shapes = dict(solid_shapes)
        self.manifolds: Dict[int, object] = dict()
        self.local_bounds: Dict[int, np.ndarray] = dict()
        self.recovery_settings_used: Dict[int, str] = dict()
        # 메쉬 생성이나 복구가 실패해 판정에서 뺀 부품. 호출자는 이것을 결과에 기록해야
        # 한다 — 조용히 빠지면 부품 수가 줄어든 것을 알 수 없다.
        self.excluded_solid_ids: List[int] = list()
        self.exclusion_reasons: Dict[int, str] = dict()

        # 삼각분할은 순차로 한다 — 프로세스 병렬화가 OCC 와 양립하지 않는다(부모가 fork
        # 시점에 OCC 내부 스레드 64개를 띄운 상태라 잠긴 뮤텍스가 자식에 복제되어 교착;
        # futex_do_wait 로 확인). 이 단계의 실측 비용은 Cleaner 13부품에서 4.9~5.3초다.
        # baseline 보정은 fork 이전에 OCC 스레드를 쓰지 않아 병렬화가 정상 작동한다.
        for solid_id, shape in self.solid_shapes.items():
            try:
                vertices, faces = triangulate_solid_preserving_topology(
                    shape, linear_deflection, angular_deflection, merge_digits
                )
            except Exception as error:
                # 이 몸체를 삼각분할하지 못했다 — 판정에서 제외한다. 다른 메쉬러로
                # 폴백하지 않는 이유는 모듈 docstring 의 라이브러리 결정 근거에 있다.
                self.excluded_solid_ids.append(solid_id)
                self.exclusion_reasons[solid_id] = f"메쉬 생성 실패: {error}"
                continue
            if len(vertices) == 0 or len(faces) == 0:
                self.excluded_solid_ids.append(solid_id)
                self.exclusion_reasons[solid_id] = "삼각분할 결과가 비었다"
                continue
            self.local_bounds[solid_id] = np.vstack([vertices.min(axis=0), vertices.max(axis=0)])
            # 부품마다 다른 삼각분할 설정을 찾지 않고 같은 복구 절차를 모두에 적용한다.
            # 실측(29 부품): 29/29 닫힘, 부피비 0.903~1.012 — B-rep 폴백이 필요 없다.
            manifold, repaired_vertices, label = repair_to_manifold(
                vertices, faces, minimum_manifold_volume
            )
            if manifold is None:
                # 일반 복구(용접 -> 비다양체 정리 -> 방향 통일 -> 띠 잇기/star 덮개)로도
                # 닫히지 않았다 — 판정에서 제외한다. 조용히 빠지면 부품 수가 줄어든 것을
                # 알 수 없으므로 사유를 남긴다.
                self.excluded_solid_ids.append(solid_id)
                self.exclusion_reasons[solid_id] = f"복구 후에도 닫히지 않음: {label}"
                continue
            self.manifolds[solid_id] = manifold
            self.local_bounds[solid_id] = np.vstack(
                [repaired_vertices.min(axis=0), repaired_vertices.max(axis=0)]
            )
            if label != "직접":
                self.recovery_settings_used[solid_id] = label

        if self.excluded_solid_ids:
            # 제외된 부품을 solid_shapes 에서도 지운다. 이것을 빼먹으면 downstream 이
            # solid_shapes 를 순회하다 manifolds/local_bounds 에 없는 키를 찾아 KeyError 가
            # 난다(실측: Coway D4P 에서 _baseline_signature 가 KeyError: 23).
            for excluded_id in self.excluded_solid_ids:
                self.solid_shapes.pop(excluded_id, None)
                self.local_bounds.pop(excluded_id, None)

            # 제외를 조용히 넘기지 않는다. 판정 가능한 부품이 하나도 없으면 진행할 의미가
            # 없으므로 예외를 던진다.
            if not self.manifolds:
                raise InterferenceException(
                    "모든 부품이 제외되어 판정할 것이 없다: "
                    + "; ".join(
                        f"{solid_id}: {reason}"
                        for solid_id, reason in self.exclusion_reasons.items()
                    )
                )

    def _transformed_solids(self, solid_id: int, transformation_matrix: np.ndarray):
        """변환을 적용한 닫힌 메쉬. 이 부품이 판정에서 제외됐으면 None."""
        if solid_id in self.manifolds:
            return [_transform_manifold(self.manifolds[solid_id], transformation_matrix)]
        return None

    def world_bounds(self, solid_id: int, transformation_matrix: np.ndarray) -> np.ndarray:
        """변환을 적용한 부품의 월드 AABB (2,3)."""
        low, high = self.local_bounds[solid_id]
        corners = np.array(
            [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])],
            dtype=float,
        )
        matrix = np.asarray(transformation_matrix, dtype=float)
        moved = corners @ matrix[:3, :3].T + matrix[:3, 3]
        return np.vstack([moved.min(axis=0), moved.max(axis=0)])

    def interference_volume(
        self,
        moving_solid_id: int,
        moving_transformation: np.ndarray,
        obstacle_solid_id: int,
        obstacle_transformation: np.ndarray,
    ) -> float:
        """두 부품이 주어진 자세에서 겹치는 부피.

        떨어져 있으면 정확히 0.0 을 반환한다.
        """
        moving_bounds = self.world_bounds(moving_solid_id, moving_transformation)
        obstacle_bounds = self.world_bounds(obstacle_solid_id, obstacle_transformation)
        if np.any(moving_bounds[1] < obstacle_bounds[0]) or np.any(
            obstacle_bounds[1] < moving_bounds[0]
        ):
            return 0.0

        moving_solids = self._transformed_solids(moving_solid_id, moving_transformation)
        obstacle_solids = self._transformed_solids(obstacle_solid_id, obstacle_transformation)
        if moving_solids is None or obstacle_solids is None:
            missing = moving_solid_id if moving_solids is None else obstacle_solid_id
            raise InterferenceException(
                f"solid {missing} has no mesh backend; the general repair is expected to "
                "close every part, so this should have been caught at construction"
            )
        return _sum_pairwise_intersection_volume(moving_solids, obstacle_solids)

    def interference_length_between(
        self,
        moving_solid_id: int,
        moving_transformation: np.ndarray,
        obstacle_solid_id: int,
        obstacle_transformation: np.ndarray,
    ) -> float:
        """두 부품이 주어진 자세에서 맞물린 깊이. 떨어져 있으면 0.0.

        깊이는 교집합 형상에서 t = 2V/A 로 잰다. 부피와 같은 AABB 사전 필터를 쓰므로
        떨어져 있는 쌍에는 추가 비용이 없다.
        """
        moving_bounds = self.world_bounds(moving_solid_id, moving_transformation)
        obstacle_bounds = self.world_bounds(obstacle_solid_id, obstacle_transformation)
        if np.any(moving_bounds[1] < obstacle_bounds[0]) or np.any(
            obstacle_bounds[1] < moving_bounds[0]
        ):
            return 0.0

        moving_solids = self._transformed_solids(moving_solid_id, moving_transformation)
        obstacle_solids = self._transformed_solids(obstacle_solid_id, obstacle_transformation)
        if moving_solids is None or obstacle_solids is None:
            missing = moving_solid_id if moving_solids is None else obstacle_solid_id
            raise InterferenceException(
                f"solid {missing} has no mesh backend; the general repair is expected to "
                "close every part, so this should have been caught at construction"
            )
        volume = _sum_pairwise_intersection_volume(moving_solids, obstacle_solids)
        return interference_length(volume)

    def calibrate_baselines(
        self,
        assembled_transformations: Dict[int, np.ndarray],
        margin: float,
        worker_count: int,
    ) -> InterferenceBudget:
        """조립 상태의 쌍별 간섭 길이를 재어 허용 상한을 만든다.

        설계상의 맞물림(압입·삽입)은 그대로 허용하고 그보다 늘어난 간섭만 충돌로 본다.
        실무 STEP 은 조립 상태에서 이미 겹쳐 있으므로 이 기준선 없이는 어떤 부품도 움직일
        수 없다.

        Args:
            assembled_transformations: 부품 번호 -> 조립 상태 4x4 변환.
            margin: 기준선 대비 추가로 허용할 간섭 길이.
            worker_count: 1 보다 크면 쌍별 계산을 fork 워커로 분산한다. 순차 계산은 대형
                껍데기가 있는 조립체에서 78쌍에 415초가 걸린다.

        Returns:
            InterferenceBudget.
        """
        solid_ids = sorted(self.solid_shapes)
        baselines: Dict[int, Dict[int, float]] = {solid_id: dict() for solid_id in solid_ids}
        pairs = [
            (first, second)
            for a_index, first in enumerate(solid_ids)
            for second in solid_ids[a_index + 1 :]
        ]

        # 쌍별 계산은 서로 독립이므로 fork 워커로 분산한다. 순차 계산은 대형 껍데기가 있는
        # 조립체에서 78쌍에 415초가 걸리는데, 그 대부분이 껍데기 쌍 몇 개에 몰려 있어
        # 병렬화 효과가 크다. OCC 형상은 pickle 되지 않으므로 fork 상속을 이용한다.
        if worker_count > 1 and len(pairs) > 1:
            computed = _run_baseline_pairs_parallel(self, pairs, assembled_transformations, worker_count)
        else:
            computed = {
                (first, second): self.interference_length_between(
                    first,
                    assembled_transformations[first],
                    second,
                    assembled_transformations[second],
                )
                for first, second in pairs
            }
        for (first, second), depth in computed.items():
            baselines[first][second] = depth
            baselines[second][first] = depth
        return InterferenceBudget(baseline_lengths=baselines, margin=float(margin))

    def find_interfering_obstacles(
        self,
        moving_solid_id: int,
        moving_transformation: np.ndarray,
        obstacle_transformations: Dict[int, np.ndarray],
        budget: InterferenceBudget,
    ) -> List[Tuple[int, float, float]]:
        """상한을 넘긴 장애물들을 (번호, 간섭 깊이, 상한) 으로 돌려준다."""
        offenders: List[Tuple[int, float, float]] = list()
        for obstacle_solid_id, obstacle_transformation in obstacle_transformations.items():
            if obstacle_solid_id == moving_solid_id:
                continue
            limit = budget.limit_for(moving_solid_id, obstacle_solid_id)
            depth = self.interference_length_between(
                moving_solid_id, moving_transformation, obstacle_solid_id, obstacle_transformation
            )
            if depth > limit:
                offenders.append((obstacle_solid_id, depth, limit))
        return offenders

    def is_transformation_valid(
        self,
        moving_solid_id: int,
        moving_transformation: np.ndarray,
        obstacle_transformations: Dict[int, np.ndarray],
        budget: InterferenceBudget,
    ) -> bool:
        """이 자세가 충돌 없는가 — 어느 장애물과도 깊이 상한을 넘지 않는가."""
        for obstacle_solid_id, obstacle_transformation in obstacle_transformations.items():
            if obstacle_solid_id == moving_solid_id:
                continue
            limit = budget.limit_for(moving_solid_id, obstacle_solid_id)
            depth = self.interference_length_between(
                moving_solid_id, moving_transformation, obstacle_solid_id, obstacle_transformation
            )
            if depth > limit:
                return False
        return True


class InterferenceSession:
    """한 부품을 움직이는 동안 장애물 변환을 고정해 두는 세션.

    장애물의 변환된 Manifold 를 한 번만 만들어 재사용한다. 이동 부품만 매 질의에서
    변환하므로 라운드 내 반복 질의가 빨라진다.
    """

    def __init__(
        self,
        checker: InterferenceVolumeChecker,
        moving_solid_id: int,
        obstacle_transformations: Dict[int, np.ndarray],
        budget: InterferenceBudget,
    ):
        """세션을 만든다.

        Args:
            checker: 부품 기하를 담은 판정기.
            moving_solid_id: 움직일 부품 번호.
            obstacle_transformations: 장애물 번호 -> 고정된 4x4 변환.
            budget: 쌍별 허용 상한.
        """
        self.checker = checker
        self.moving_solid_id = moving_solid_id
        self.budget = budget
        self.obstacle_transformations = {
            solid_id: np.asarray(matrix, dtype=float)
            for solid_id, matrix in obstacle_transformations.items()
            if solid_id != moving_solid_id
        }
        if not self.obstacle_transformations:
            raise InterferenceException(
                "at least one obstacle besides the moving solid is required"
            )
        self.obstacle_world_bounds = {
            solid_id: checker.world_bounds(solid_id, matrix)
            for solid_id, matrix in self.obstacle_transformations.items()
        }
        self.transformed_obstacles: Dict[int, List[object]] = dict()
        for solid_id, matrix in self.obstacle_transformations.items():
            transformed = checker._transformed_solids(solid_id, matrix)
            if transformed is not None:
                self.transformed_obstacles[solid_id] = transformed
        self.query_count = 0
        self.boolean_count = 0

        # 속도 개선: 자세별 판정 결과 캐시. RRT* 는 같은 자세를 여러 번 검사한다.
        # 키는 이동 변환을 반올림한 값이며, 값은 is_transformation_valid 결과다.
        self.decision_cache: Dict[Tuple, bool] = dict()
        self.cache_hit_count = 0


    @staticmethod
    def _transformation_key(transformation_matrix: np.ndarray) -> Tuple:
        """캐시 키. 0.01 유닛/라디안 단위로 반올림한다(탐색 격자보다 훨씬 촘촘)."""
        return tuple(np.round(np.asarray(transformation_matrix, dtype=float).ravel(), 2))

    def interference_length_against(
        self, obstacle_solid_id: int, moving_transformation: np.ndarray
    ) -> float:
        """이동 부품이 한 장애물과 맞물린 깊이. AABB 가 떨어져 있으면 0.0.

        깊이 t = 2V/A 다. AABB 사전 필터가 먼저 걸리므로 떨어져 있는 쌍에는 면적 계산
        비용이 붙지 않는다 — 실제 탐색에서 대부분의 질의가 그 필터에서 끝난다.
        """
        self.query_count += 1
        moving_bounds = self.checker.world_bounds(self.moving_solid_id, moving_transformation)
        obstacle_bounds = self.obstacle_world_bounds[obstacle_solid_id]
        if np.any(moving_bounds[1] < obstacle_bounds[0]) or np.any(
            obstacle_bounds[1] < moving_bounds[0]
        ):
            return 0.0

        self.boolean_count += 1
        if obstacle_solid_id in self.transformed_obstacles:
            moving_solids = self.checker._transformed_solids(
                self.moving_solid_id, moving_transformation
            )
            if moving_solids is not None:
                volume = _sum_pairwise_intersection_volume(
                    moving_solids, self.transformed_obstacles[obstacle_solid_id]
                )
                return interference_length(volume)

        return self.checker.interference_length_between(
            self.moving_solid_id,
            moving_transformation,
            obstacle_solid_id,
            self.obstacle_transformations[obstacle_solid_id],
        )

    def is_transformation_valid(self, moving_transformation: np.ndarray) -> bool:
        """이 자세가 충돌 없는가. 상한을 넘긴 장애물을 만나면 즉시 False.

        자세 캐시: 같은 자세 재질의는 계산 없이 반환한다.

        판정은 전부 삼각형 메쉬 불린(manifold3d)이다. 볼록 분해 백엔드와 그 B-rep 재확인
        경로는 제거됐다 — 볼록 조각의 합집합 부피가 원본보다 커서(껍데기 실측 1.9~3.9 배)
        부풀린 baseline 이 실제 관통을 덮었기 때문이다.
        """
        cache_key = self._transformation_key(moving_transformation)
        cached = self.decision_cache.get(cache_key)
        if cached is not None:
            self.cache_hit_count += 1
            return cached

        result = True
        for obstacle_solid_id in self.obstacle_transformations:
            limit = self.budget.limit_for(self.moving_solid_id, obstacle_solid_id)
            depth = self.interference_length_against(obstacle_solid_id, moving_transformation)
            if depth > limit:
                result = False
                break

        self.decision_cache[cache_key] = result
        return result

    def is_segment_valid(
        self,
        start_transformation: np.ndarray,
        end_transformation: np.ndarray,
        interpolation_count: int,
    ) -> bool:
        """두 자세를 잇는 직선 구간이 충돌 없는가.

        회전이 섞인 구간도 다룰 수 있도록 회전 부분은 각 보간점에서 시작·끝 중
        가까운 쪽을 쓴다(순수 이동 구간에서는 정확하다). 회전 보간이 필요한 호출자는
        보간된 변환 목록을 직접 만들어 is_transformation_valid 로 검사하기를 권한다.
        """
        if interpolation_count < 2:
            raise InterferenceException("interpolation_count must be at least 2")
        start = np.asarray(start_transformation, dtype=float)
        end = np.asarray(end_transformation, dtype=float)
        for step in range(interpolation_count + 1):
            ratio = step / interpolation_count
            matrix = start.copy()
            matrix[:3, 3] = start[:3, 3] * (1.0 - ratio) + end[:3, 3] * ratio
            if ratio > 0.5:
                matrix[:3, :3] = end[:3, :3]
            if not self.is_transformation_valid(matrix):
                return False
        return True
