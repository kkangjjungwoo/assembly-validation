"""RRT* 기반 조립체 분해 경로 탐색.

STEP 조립체를 입력으로 각 부품의 분해 경로를 RRT* 로 찾아 msgpack 으로 저장한다.
저장된 경로를 뒤집으면 조립 경로가 된다.

    python main.py step_path=<입력.stp> output_path=<출력.msgpack>

`config/config.yaml` 의 모든 항목을 같은 방식으로 커맨드라인에서 덮어쓸 수 있다.

충돌 판정은 삼각형 메쉬 불린(manifold3d)의 간섭 길이 L = V^(1/3) 하나만 쓴다. 조립
상태의 간섭 길이를 기준선으로 삼아 그보다 max_interference_growth 이상 증가하면 충돌로
본다.
"""
import os
import sys
import time
import unicodedata

import hydra
import numpy as np
import trimesh
from omegaconf import DictConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.action import Action, ActionType
from core.interference import (
    InterferenceSession,
    InterferenceVolumeChecker,
    repair_mesh_geometry,
)
from core.planner import (
    PlanningException,
    PlanningResult,
    RRTStarConfig,
    RRTStarPlanner,
)
from core.state import State
from data.exporter import MsgpackTrajectorySerializer
from data.loader import STEPLoader, match_solids_to_step

def _display_width(text: str) -> int:
    """터미널에서 문자열이 차지하는 칸 수. 한글·전각 문자는 두 칸이다."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int, does_align_right: bool = False) -> str:
    """표시폭 기준으로 채운다 — str.ljust 는 한글을 한 칸으로 세어 표가 어긋난다."""
    padding = " " * max(0, width - _display_width(text))
    return padding + text if does_align_right else text + padding


def print_table(headers, rows, alignments=None) -> None:
    """열 폭을 내용에 맞춰 정한 표를 출력한다.

    Args:
        headers: 열 제목 목록.
        rows: 행마다 문자열 목록. 길이가 headers 와 같아야 한다.
        alignments: 열마다 "<"(왼쪽) 또는 ">"(오른쪽). 생략하면 전부 왼쪽.
    """
    if alignments is None:
        alignments = ["<"] * len(headers)
    widths = [max(_display_width(headers[i]),
                  *(_display_width(row[i]) for row in rows)) if rows
              else _display_width(headers[i])
              for i in range(len(headers))]
    align_right = [alignments[i] == ">" for i in range(len(headers))]
    def render(cells):
        return ("  " + "  ".join(_pad(cells[i], widths[i], align_right[i])
                                 for i in range(len(headers)))).rstrip()

    print(render(headers), flush=True)
    print(render(["─" * width for width in widths]), flush=True)
    for row in rows:
        print(render(row), flush=True)


def print_section(title: str) -> None:
    """구획 제목을 출력한다."""
    print(f"\n── {title} " + "─" * max(0, 74 - _display_width(title)), flush=True)


def print_msgpack_structure(document, output_path: str) -> None:
    """저장된 msgpack 의 구조와 분해 경로를 출력한다.

    메모리의 값이 아니라 재판독한 문서를 읽는다 — 소비자가 실제로 보게 되는 것이
    무엇인지 보여야 하고, 직렬화·재판독을 거쳐 값이 바뀌지 않았음도 함께 확인된다.

    Args:
        document: read_from_file 로 재판독한 문서.
        output_path: 저장 경로(크기 표시에 쓴다).
    """
    metadata = document["metadata"]
    solids = document["solids"]
    trajectories = document["trajectories"]

    print_section("산출물 구조")
    print(f"  {os.path.basename(output_path)}  "
          f"({os.path.getsize(output_path) / 1e6:.1f} MB)", flush=True)
    vertex_total = sum(len(solids[key]["mesh"]["vertices"]) for key in solids)
    face_total = sum(len(solids[key]["mesh"]["faces"]) for key in solids)
    sample_key = next(iter(sorted(solids)))
    sample_step = next(iter(sorted(trajectories))) if trajectories else None
    conversion_counts = dict()
    for key in solids:
        result = str(solids[key].get("conversion", "—"))
        conversion_counts[result] = conversion_counts.get(result, 0) + 1
    lines = [
        ("metadata", ""),
        ("  step_path", str(metadata["step_path"])),
        ("  global_bbox", f"min {[round(v, 1) for v in metadata['global_bbox']['min']]}  "
                          f"max {[round(v, 1) for v in metadata['global_bbox']['max']]}"),
        ("  excluded_bodies", f"{len(metadata.get('excluded_bodies', []))}개 "
                             "(형상조차 얻지 못해 판정에서 뺀 몸체)"),
        ("  unwatertight_bodies", f"{len(metadata.get('unwatertight_bodies', []))}개 "
                                 "(형상은 담고 판정에서만 뺀 몸체)"),
        ("solids", f"{len(solids)}개  ·  정점 {vertex_total:,}  ·  삼각형 {face_total:,}"),
        (f"  [{sample_key}].name", str(solids[sample_key].get("name", "—"))),
        (f"  [{sample_key}].conversion",
         str(solids[sample_key].get("conversion", "—"))
         + "  (전체: " + ", ".join(f"{key} {value}개" for key, value
                                   in sorted(conversion_counts.items())) + ")"),
        (f"  [{sample_key}].mesh", f"vertices {len(solids[sample_key]['mesh']['vertices']):,} x 3  ·  "
                                   f"faces {len(solids[sample_key]['mesh']['faces']):,} x 3"),
        (f"  [{sample_key}].state", f"position {solids[sample_key]['state']['position']}  ·  "
                                     f"rotation {solids[sample_key]['state']['rotation']}  "
                                     "(조립 상태)"),
        ("trajectories", f"{len(trajectories)} 스텝  (분해 순서 — 뒤집으면 조립 순서)"),
    ]
    if sample_step is not None:
        entry = trajectories[sample_step]
        lines.append((f"  [{sample_step}]", f"solid {entry['solid']}  ·  "
                                            f"action {entry['action']['type']}  ·  "
                                            "state = 도달 위치·자세"))
    label_width = max(_display_width(label) for label, _ in lines)
    for label, value in lines:
        print(("  " + _pad(label, label_width)
               + ("  " + value if value else "")).rstrip(), flush=True)

    if not trajectories:
        return

    # 스텝을 부품 단위 구간으로 묶는다 — 161 스텝을 그대로 찍으면 읽을 수 없고, 웨이포인트
    # 분할로 쪼갠 조각은 한 부품의 한 동작이므로 구간이 의미 단위다.
    segments = []
    for key in sorted(trajectories):
        entry = trajectories[key]
        solid_id = entry["solid"]
        if segments and segments[-1]["solid"] == solid_id:
            segments[-1]["steps"].append(entry)
        else:
            segments.append(dict(solid=solid_id, steps=[entry]))

    names_by_id = {int(key): str(value.get("name", "")) for key, value in solids.items()}

    # '순서' 는 궤적에 나타나는 구간 순서이고 그것이 곧 제거 순서다 — 그리디 라운드가
    # 라운드마다 빠진 부품의 궤적을 순서대로 이어 붙이기 때문이다. 궤적을 뒤집으면 조립
    # 순서가 된다는 것이 이 성질에 의존한다.
    print_section("분해 경로  (역순이 조립 경로)")
    rows = []
    for order, segment in enumerate(segments, 1):
        solid_id = int(segment["solid"])
        steps = segment["steps"]
        start = np.asarray(solids[solid_id]["state"]["position"], dtype=float)
        finish = np.asarray(steps[-1]["state"]["position"], dtype=float)
        displacement = finish - start
        axis = int(np.argmax(np.abs(displacement)))
        rotation_count = sum(1 for step in steps if step["action"]["type"] != "translation")
        rows.append([
            str(order),
            str(solid_id),
            names_by_id.get(solid_id, "")[:16],
            f"{len(steps)}",
            f"{'XYZ'[axis]}{'+' if displacement[axis] >= 0 else '-'}",
            f"{float(np.linalg.norm(displacement)):.1f}",
            f"{rotation_count}" if rotation_count else "—",
            f"{steps[-1]['state']['rotation']}",
        ])
    print_table(
        ["순서", "번호", "이름", "스텝", "축", "이동거리", "회전", "최종 자세"],
        rows,
        [">", ">", "<", ">", "<", ">", ">", "<"],
    )
    print(f"\n  조립 경로는 위 {len(segments)}개 구간을 역순으로, 각 구간의 스텝도 역순으로,"
          f" 각 동작의 값을 반대로 적용해 얻는다.", flush=True)


# 목표 도달 후 분리 축으로 더 밀어내는 거리. RRT* 는 AABB 가 removal_clearance 만큼만
# 분리되면 멈추므로(실측 13.4 유닛) 분해도로 쓰기엔 조립체에 붙어 있다. AABB 가 이미
# 분리된 축으로 더 가는 것은 간섭을 늘리지 않는다.
ESCAPE_DISTANCE = 400.0

# 긴 병진을 쪼개는 웨이포인트 간격. 시각화에서 부품이 순간이동하지 않도록 한다.
TRAJECTORY_WAYPOINT_STEP = 50.0


def execute_disassembly_search(
    step_path: str,
    output_path: str,
    worker_count: int,
    iteration_count: int,
    does_sample_rotation: bool,
    random_seed: int,
    time_budget_seconds: float,
    max_interference_growth: float,
):
    """조립체 하나를 분해 탐색하고 결과를 msgpack 으로 저장한다.

    Args:
        step_path: 입력 STEP 파일 경로.
        output_path: 출력 msgpack 파일 경로.
        worker_count: 기준선 보정 병렬 워커 수.
        iteration_count: 부품 하나당 RRT* 최대 반복 수.
        does_sample_rotation: 참이면 24 가지 이산 자세를 함께 탐색한다.
        random_seed: 난수 시드. 같은 시드는 같은 결과를 준다.
        time_budget_seconds: 부품 하나당 탐색 시간 상한.
        max_interference_growth: 조립 상태 간섭 길이 대비 추가로 허용할 길이.

    Returns:
        {"removable", "total", "trajectory_step_count", "output_path", "seconds"}
    """
    search_started = time.time()
    output_directory = os.path.dirname(os.path.abspath(output_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    def elapsed() -> str:
        return f"{time.time() - search_started:6.1f}s"

    print_section(f"입력  {os.path.basename(step_path)}")

    # ---------- 1. 로드 및 메쉬 변환 + watertight 복구 ----------
    loader = STEPLoader(step_path, face_tolerance=0.02, angle_tolerance=0.5, max_workers=8)
    meshes, signatures, seen_signatures, repair_labels, names = {}, {}, set(), {}, {}
    name_sources, hash_named_ids = {}, []
    # 판정에서 뺀 몸체는 두 갈래다. display_* 는 형상은 얻었으나 봉합이 안 된 몸체로,
    # msgpack 에 형상을 담고 탐색·장애물에서만 뺀다. excluded_bodies 는 삼각분할이 아무
    # 면도 내지 못해 형상조차 없는 몸체로, 사유만 남긴다.
    display_meshes, display_names, display_signatures, display_reasons = {}, {}, {}, {}
    excluded_bodies = []
    for mesh in loader.load_all():
        signature = STEPLoader.mesh_signature(mesh.vertices)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        repaired_vertices, repaired_faces, manifold, repair_label = repair_mesh_geometry(
            np.asarray(mesh.vertices, dtype=float),
            np.asarray(mesh.faces, dtype=np.int64),
            1e-9,
        )
        if manifold is None:
            display_id = len(display_meshes)
            display_meshes[display_id] = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices, float),
                faces=np.asarray(mesh.faces, np.int64),
                process=False,
            )
            display_names[display_id] = str(mesh.metadata["name"])
            display_signatures[display_id] = signature
            display_reasons[display_id] = f"복구 후에도 닫히지 않음: {repair_label}"
            continue
        solid_id = len(meshes)
        meshes[solid_id] = trimesh.Trimesh(
            vertices=repaired_vertices, faces=repaired_faces, process=False
        )
        signatures[solid_id] = signature
        names[solid_id] = str(mesh.metadata["name"])
        # 이름의 출처: product(STEP PRODUCT 체인) | occ(pythonocc) | hash(형상 해시 폴백).
        # 'SOLID#3' 같은 무의미한 이름도 STEP 에서 올 수 있어 문자열만으로는 신뢰도를 모른다.
        name_sources[solid_id] = str(mesh.metadata.get("name_source", "occ"))
        if name_sources[solid_id] == "hash":
            hash_named_ids.append(solid_id)
        if repair_label != "직접":
            repair_labels[solid_id] = repair_label
    for skipped_name, skipped_reason in getattr(loader, "skipped_bodies", []):
        excluded_bodies.append(("", str(skipped_name), str(skipped_reason)))

    solid_ids = sorted(meshes)
    if not solid_ids:
        raise RuntimeError(
            "모든 몸체가 제외되어 판정할 것이 없다: "
            + "; ".join(f"{name}: {reason}" for _, name, reason in excluded_bodies)
        )
    step_named_ids = [i for i in solid_ids if i not in hash_named_ids]

    extents = {i: np.asarray(meshes[i].bounds[1]) - np.asarray(meshes[i].bounds[0])
               for i in solid_ids}
    vertex_arrays = {i: np.asarray(meshes[i].vertices, float) for i in solid_ids}
    # 대응 오차 상한. 이 값을 넘으면 메쉬-B-rep 짝짓기를 신뢰할 수 없어 예외를 던진다.
    shapes = match_solids_to_step(vertex_arrays, step_path, maximum_error=2.0)
    identity = np.eye(4)
    assembled_states = {i: State((0.0, 0.0, 0.0), (0, 0, 0)) for i in solid_ids}

    smallest_extent = min(float(np.min(extents[i])) for i in solid_ids)
    spacing = smallest_extent / 2.0
    global_low = np.min([np.asarray(meshes[i].bounds[0]) for i in solid_ids], axis=0)
    global_high = np.max([np.asarray(meshes[i].bounds[1]) for i in solid_ids], axis=0)
    assembly_span = float(np.max(global_high - global_low))

    print(f"  부품 {len(solid_ids)}개  ·  B-rep 대응 {len(shapes)}/{len(solid_ids)}"
          f"  ·  이름 {len(step_named_ids)}/{len(solid_ids)} STEP"
          + (f" + {len(hash_named_ids)} 해시" if hash_named_ids else "")
          + f"  ·  {elapsed()}", flush=True)
    print(f"  조립체 span {assembly_span:.1f}  ·  최소 부품 두께 {smallest_extent:.2f}"
          f"  ·  탐색 스텝 {spacing:.2f}", flush=True)
    if repair_labels:
        print(f"  메쉬 복구 {len(repair_labels)}/{len(solid_ids)}: "
              + ", ".join(f"{signatures[i]}({repair_labels[i]})"
                          for i in sorted(repair_labels)), flush=True)
    if excluded_bodies:
        print(f"  제외 {len(excluded_bodies)}개: "
              + ", ".join(f"{name}({reason[:36]})" for _, name, reason in excluded_bodies),
              flush=True)

    print_section("부품")
    print_table(
        ["서명", "이름", "부피", "AABB (x·y·z)"],
        [[signatures[i], str(names.get(i, "")),
          f"{float(meshes[i].volume):,.0f}" if meshes[i].is_volume else "—",
          " · ".join(f"{value:.1f}" for value in extents[i])]
         for i in sorted(solid_ids, key=lambda k: -float(np.prod(extents[k])))],
        ["<", "<", ">", "<"],
    )

    # ---------- 2. 판정기 구성 및 기준선 보정 ----------
    preparation_started = time.time()
    checker = InterferenceVolumeChecker(shapes)
    # checker 는 자체 삼각분할·복구 경로를 쓰므로 로더가 통과시킨 부품을 여기서 또 제외할
    # 수 있다. 그 제외를 흡수하지 않으면 판정 도중 KeyError 가 난다.
    for excluded_id in getattr(checker, "excluded_solid_ids", []):
        reason = checker.exclusion_reasons.get(excluded_id, "판정기 구성 중 제외")
        excluded_bodies.append((signatures.get(excluded_id, ""),
                                names.get(excluded_id, str(excluded_id)),
                                f"판정기: {reason}"))
        for table in (meshes, shapes, extents, vertex_arrays, assembled_states,
                      signatures, names):
            table.pop(excluded_id, None)
    solid_ids = sorted(meshes)
    if not solid_ids:
        raise RuntimeError(
            "판정기 구성 후 남은 부품이 없다: "
            + "; ".join(f"{name}: {reason}" for _, name, reason in excluded_bodies)
        )
    if getattr(checker, "excluded_solid_ids", []):
        print(f"  판정기 추가 제외 {len(checker.excluded_solid_ids)}개 "
              f"-> 남은 부품 {len(solid_ids)}개", flush=True)

    budget = checker.calibrate_baselines({i: identity for i in solid_ids},
                                         margin=float(max_interference_growth),
                                         worker_count=worker_count)
    # 조립 상태 간섭 길이의 분포를 찍는다 — 허용값이 이 분포에 비해 어느 규모인지 보이지
    # 않으면 값을 고를 수 없다.
    # 쌍은 명시적으로 센다 — baseline_lengths 는 (i,j)·(j,i) 양방향을 담으므로 원소 수를
    # 2 로 나누면 개수가 어긋난다. 문턱을 두는 것은 실측 때문이다: 겹침이 스치는 정도인
    # 쌍의 길이가 1e-05 와 0 사이에서 실행마다 흔들려(as1 3회에서 양수 쌍 25·26·27) 0 초과
    # 로 세면 보고 값이 재현되지 않는다. 이 문턱은 표시에만 쓰고 판정에는 관여하지 않는다.
    CONTACT_REPORT_THRESHOLD = 1e-3
    contact_pairs = {
        frozenset((first, second)): length
        for first, obstacles in budget.baseline_lengths.items()
        for second, length in obstacles.items()
        if length > CONTACT_REPORT_THRESHOLD and first != second
    }
    contact_lengths = sorted(contact_pairs.values())
    print_section("충돌 판정")
    print(f"  방식      삼각형 메쉬 불린(manifold3d), 간섭 길이 L = V^(1/3)", flush=True)
    print(f"  허용 증가 {max_interference_growth:g}  "
          f"(쌍별 상한 = 조립 상태 간섭 길이 + 이 값)", flush=True)
    if contact_lengths:
        print(f"  조립 간섭 쌍 {len(contact_lengths)}개  ·  길이 최대 "
              f"{contact_lengths[-1]:.3f} / 중앙 "
              f"{contact_lengths[len(contact_lengths) // 2]:.3f}"
              f"  (길이 {CONTACT_REPORT_THRESHOLD:g} 이하는 세지 않음)", flush=True)
    else:
        print(f"  조립 간섭 없음 (모든 쌍 길이 {CONTACT_REPORT_THRESHOLD:g} 이하)",
              flush=True)
    print(f"  manifold {len(checker.manifolds)}/{len(solid_ids)}  ·  준비 "
          f"{time.time() - preparation_started:.1f}s  ·  {elapsed()}", flush=True)

    # ---------- 3. RRT* 설정 ----------
    # 모든 값이 spacing(가장 얇은 부품 두께의 절반)에서 유도된다. 더 크면 좁은 통로를
    # 건너뛰고 더 작으면 반복이 낭비된다. neighbor_radius 는 부모 재선택이 의미를 가지도록
    # 스텝의 3배, sampling_margin_ratio 1.0 은 조립체 밖까지 샘플링하기 위한 값이다.
    config = RRTStarConfig(
        max_iteration_count=iteration_count,
        translation_step_size=spacing,
        neighbor_radius=spacing * 3.0,
        goal_sample_rate=0.3,
        rotation_distance_weight=spacing,
        translation_interpolation_count=8,
        rotation_interpolation_count=8,
        removal_clearance=spacing,
        sampling_margin_ratio=1.0,
        does_sample_rotation=does_sample_rotation,
        maximum_extension_step_count=int(np.ceil(assembly_span * 2.0 / spacing)) + 2,
        stops_at_first_feasible_path=True,
        random_seed=random_seed,
    )
    print_section("RRT* 설정")
    print_table(
        ["항목", "값", "유도"],
        [["최대 반복", f"{iteration_count}", "config.search.iteration_count"],
         ["병진 스텝", f"{spacing:.2f}", "최소 부품 두께 / 2"],
         ["이웃 반경", f"{config.neighbor_radius:.2f}", "스텝 x 3"],
         ["회전 가중", f"{config.rotation_distance_weight:.2f}", "스텝 (전이 1회 = 1스텝)"],
         ["목표 여유", f"{config.removal_clearance:.2f}", "스텝"],
         ["확장 한계", f"{config.maximum_extension_step_count}", "조립체 span x 2 / 스텝"],
         ["회전 탐색", "24 자세" if does_sample_rotation else "고정", "config"],
         ["시드", f"{random_seed}", "config"]],
        ["<", ">", "<"],
    )

    def extend_to_escape(planner, result):
        """도달 지점에서 분리 축으로 ESCAPE_DISTANCE 까지 더 밀어낸다.

        분리된 축으로 더 가는 것은 간섭을 늘리지 않지만, 연장 구간도 스윕 검사한다.
        실패하면 연장 없이 원래 결과를 돌려준다.
        """
        final_state = result.states[-1]
        start_position = np.asarray(planner.start_state.position, dtype=float)
        final_position = np.asarray(final_state.position, dtype=float)
        displacement = final_position - start_position

        axis = int(np.argmax(np.abs(displacement)))
        sign = 1.0 if displacement[axis] >= 0.0 else -1.0
        remaining = ESCAPE_DISTANCE - abs(displacement[axis])
        if remaining <= 0.0:
            return result

        escape_position = final_position.copy()
        escape_position[axis] += sign * remaining
        escape_state = State(position=tuple(escape_position), rotation=final_state.rotation)

        if not planner.collision_session.is_valid_state(escape_state):
            return result
        if not planner._segment_valid(
            final_state, escape_state,
            planner._interpolation_count_for(final_state, escape_state),
        ):
            return result

        escape_action = Action(
            action_type=ActionType.TRANSLATION,
            value=tuple(escape_position - final_position),
        )
        return PlanningResult(
            is_success=True,
            states=result.states + (escape_state,),
            actions=result.actions + (escape_action,),
            cost=result.cost + float(remaining),
            iteration_count=result.iteration_count,
        )

    def plan_one_solid(moving_solid_id, present_ids):
        """한 부품을 RRT* 로 탐색한다. 성공하면 PlanningResult, 실패하면 None."""
        obstacle_ids = [i for i in present_ids if i != moving_solid_id]
        if not obstacle_ids:
            return None
        # 부품별 허용값 조정은 없다 — 길이는 부품 크기에 대해 완만하게 변하므로 하나의
        # 허용값이 모든 부품에 같은 의미를 갖는다.
        session = InterferenceSession(
            checker, moving_solid_id, {i: identity for i in obstacle_ids}, budget
        )
        planner = RRTStarPlanner(
            moving_solid_id=moving_solid_id,
            solid_meshes={i: meshes[i] for i in present_ids},
            assembled_states={i: assembled_states[i] for i in present_ids},
            config=config,
            interference_session=session,
        )
        result = planner.execute_search()
        if not result.is_success:
            return None
        return extend_to_escape(planner, result)

    # ---------- 4. 그리디 라운드 — 라운드마다 빠지는 부품 하나를 찾아 제거 ----------
    print_section("분해 탐색")
    search_header = ["#", "부품", "이름", "계층", "스텝", "비용", "남은", "누적"]
    search_alignments = [">", "<", "<", "<", ">", ">", ">", ">"]
    search_widths = [3, 6, 16, 16, 4, 7, 4, 6]
    def print_search_row(cells) -> None:
        print(("  " + "  ".join(_pad(cells[i], search_widths[i],
                                     search_alignments[i] == ">")
                                for i in range(len(cells)))).rstrip(), flush=True)

    print_search_row(search_header)
    print_search_row(["─" * width for width in search_widths])

    search_loop_started = time.time()
    active_ids = list(solid_ids)
    removal_results = []
    resolved_tiers = {}
    while active_ids:
        if len(active_ids) == 1:
            # 장애물이 없으면 자명하게 분해된다(planner 는 장애물 0 개에서 예외를 던진다).
            # 궤적을 비우면 분해도에서 이 부품만 제자리에 남으므로 이탈 궤적을 만들어 준다.
            last_id = active_ids[0]
            escape_position = np.zeros(3)
            escape_position[0] = ESCAPE_DISTANCE
            escape_state = State(position=tuple(escape_position), rotation=(0, 0, 0))
            escape_action = Action(action_type=ActionType.TRANSLATION,
                                   value=tuple(escape_position))
            removal_results.append((last_id, PlanningResult(
                is_success=True,
                states=(assembled_states[last_id], escape_state),
                actions=(escape_action,),
                cost=float(ESCAPE_DISTANCE),
                iteration_count=0,
            )))
            resolved_tiers[last_id] = "자명(마지막)"
            print_search_row([str(len(removal_results)), signatures[last_id],
                               str(names.get(last_id, ""))[:16], "자명(마지막)",
                               "1", f"{ESCAPE_DISTANCE:.1f}", "0", elapsed()])
            active_ids = []
            break

        picked = None
        # 부피 오름차순 — 작은 부품이 빠질 가능성이 높아 조기 종료 이득이 크다.
        for moving_solid_id in sorted(active_ids, key=lambda i: float(np.prod(extents[i]))):
            try:
                result = plan_one_solid(moving_solid_id, active_ids)
            except PlanningException as error:
                print(f"    ! {signatures[moving_solid_id]}  예외: {error}", flush=True)
                continue
            if result is not None:
                picked = (moving_solid_id, result)
                break
            if time.time() - search_loop_started > time_budget_seconds * len(solid_ids):
                print("    ! 시간 상한 도달 — 이 라운드에서 중단", flush=True)
                break
        if picked is None:
            break

        moving_solid_id, result = picked
        removal_results.append((moving_solid_id, result))
        active_ids.remove(moving_solid_id)
        # 계층 판정은 iteration_count 로만 한다. len(actions) 는 쓸 수 없다 —
        # extend_to_escape 가 동작을 붙이면 Tier1 직선도 2동작이 되어 오분류된다.
        tier = ("Tier1 직선" if result.iteration_count == 0
                else f"Tier2 RRT* {result.iteration_count}회")
        resolved_tiers[moving_solid_id] = tier
        print_search_row([str(len(removal_results)), signatures[moving_solid_id],
                          str(names.get(moving_solid_id, ""))[:16], tier,
                          str(len(result.actions)), f"{result.cost:.1f}",
                          str(len(active_ids)), elapsed()])

    search_failure_ids = list(active_ids)
    print(f"\n  분해 {len(removal_results)}/{len(solid_ids)}"
          + (f"  ·  실패 {', '.join(signatures[i] for i in active_ids)}"
             if active_ids else "")
          + f"  ·  탐색 {time.time() - search_loop_started:.0f}s", flush=True)

    # ---------- 5. 궤적 구성 (분해 경로 — 뒤집으면 조립 경로) ----------
    # 각 스텝은 (부품, 도달 State, 적용 Action) 이다. 긴 병진은 웨이포인트로 쪼갠다 —
    # 판정은 이미 스윕으로 검사했으므로 쪼개기가 안전성을 바꾸지 않는다.
    trajectory_steps = []
    for moving_solid_id, result in removal_results:
        for previous_state, state, action in zip(
            result.states[:-1], result.states[1:], result.actions
        ):
            if action.action_type != ActionType.TRANSLATION:
                trajectory_steps.append((moving_solid_id, state, action))
                continue
            displacement = np.asarray(action.value, dtype=float)
            distance = float(np.linalg.norm(displacement))
            piece_count = max(1, int(np.ceil(distance / TRAJECTORY_WAYPOINT_STEP)))
            piece = displacement / piece_count
            position = np.asarray(previous_state.position, dtype=float)
            for piece_index in range(piece_count):
                position = position + piece
                # 마지막 조각은 누적 오차 없이 목표 위치를 그대로 쓴다.
                if piece_index == piece_count - 1:
                    position = np.asarray(state.position, dtype=float)
                trajectory_steps.append((
                    moving_solid_id,
                    State(position=tuple(position), rotation=state.rotation),
                    Action(action_type=ActionType.TRANSLATION, value=tuple(piece)),
                ))
    rotation_step_count = sum(1 for _, _, action in trajectory_steps
                              if action.action_type != ActionType.TRANSLATION)

    # ---------- 6. 부품별 상태 메타데이터 ----------
    removal_orders = {solid_id: index + 1
                      for index, (solid_id, _) in enumerate(removal_results)}
    results_by_id = dict(removal_results)
    parts_metadata = {}
    for solid_id in solid_ids:
        entry = {
            "signature": signatures[solid_id],
            "name": names.get(solid_id),
            "name_source": name_sources.get(solid_id, "occ"),
        }
        if solid_id in removal_orders:
            result = results_by_id[solid_id]
            entry.update({
                "status": "분해 가능",
                "is_removable": True,
                "removal_order": removal_orders[solid_id],
                "path_step_count": len(result.actions),
                "path_cost": float(result.cost),
                "rrt_iteration_count": int(result.iteration_count),
                "resolved_by": resolved_tiers.get(solid_id),
                "failure_reason": None,
            })
        else:
            entry.update({
                "status": "분해 불가능",
                "is_removable": False,
                "removal_order": None,
                "path_step_count": None,
                "path_cost": None,
                "rrt_iteration_count": None,
                "resolved_by": None,
                "failure_reason": ("RRT* 가 이 설정에서 목표 도달 경로를 찾지 못했다. "
                                   "반복 수·시간 상한·회전 허용 여부를 바꾸면 달라질 수 있다."),
            })
        parts_metadata[int(solid_id)] = entry

    # parts_metadata 는 화면 출력 전용이다 — msgpack 에는 넣지 않는다.
    metadata = {
        # 형상조차 얻지 못해 판정에서 뺀 몸체. 장애물로도 참여하지 않으므로 남은 부품의
        # '분해 가능' 판정은 이 몸체가 없는 조립체에 대한 결과다.
        "excluded_bodies": [
            {"signature": signature, "name": name, "reason": reason}
            for signature, name, reason in excluded_bodies
        ],
    }

    # ---------- 7. msgpack 저장 ----------
    # 봉합 실패 몸체를 여기서 합친다. 탐색·장애물에서는 빠졌지만 형상은 저장한다 —
    # 시각화에서 부품이 사라지면 조립체가 실제와 달라 보인다.
    export_meshes = dict(meshes)
    export_names = dict(names)
    # 판정에 쓰인 부품은 봉합에 성공한 것들이다. 실패 몸체는 아래에서 형상만 덧붙인다.
    conversions = {int(solid_id): "성공" for solid_id in solid_ids}
    for display_id in sorted(display_meshes):
        export_id = max(export_meshes) + 1 if export_meshes else 0
        export_meshes[export_id] = display_meshes[display_id]
        export_names[export_id] = display_names[display_id]
        assembled_states[export_id] = State((0.0, 0.0, 0.0), (0, 0, 0))
        conversions[int(export_id)] = "실패"
        parts_metadata[int(export_id)] = {
            "signature": display_signatures[display_id],
            "name": display_names[display_id],
            "status": "메쉬 변환 실패",
            "removal_order": None,
            "path_step_count": None,
            "resolved_by": None,
        }
    # excluded_bodies(형상조차 없음)와 달리 이 몸체들은 solids 에 형상이 들어 있다.
    metadata["unwatertight_bodies"] = [
        {"signature": display_signatures[display_id], "name": display_names[display_id],
         "reason": display_reasons[display_id]}
        for display_id in sorted(display_meshes)
    ]

    serializer = MsgpackTrajectorySerializer(step_path, export_meshes, assembled_states,
                                             extra_metadata=metadata,
                                             solid_names=export_names,
                                             conversion_results=conversions)
    serializer.write_to_file(trajectory_steps, output_path)
    read_back = MsgpackTrajectorySerializer.read_from_file(output_path)

    print_section("결과")
    print_table(
        ["서명", "이름", "상태", "순서", "스텝", "해소"],
        [[entry["signature"], str(entry["name"] or "")[:18], entry["status"],
          str(entry["removal_order"] or "—"), str(entry["path_step_count"] or "—"),
          str(entry["resolved_by"] or "—")]
         for entry in sorted(
             (parts_metadata[int(i)] for i in solid_ids),
             key=lambda e: (e["removal_order"] is None, e["removal_order"] or 0))],
        ["<", "<", "<", ">", ">", "<"],
    )

    seconds = time.time() - search_started
    print_section("요약")
    print(f"  분해        {len(removal_results)}/{len(solid_ids)} 부품"
          + (f"  (실패 {len(search_failure_ids)})" if search_failure_ids else "")
          + (f"  (메쉬 실패 {len(display_meshes)})" if display_meshes else ""), flush=True)
    print(f"  분해 궤적   {len(trajectory_steps)} 스텝  "
          f"(병진 {len(trajectory_steps) - rotation_step_count} · "
          f"회전 {rotation_step_count} · 웨이포인트 간격 "
          f"{TRAJECTORY_WAYPOINT_STEP:.0f})", flush=True)
    print(f"  조립 궤적   위 궤적을 뒤집으면 조립 순서가 된다", flush=True)
    print(f"  소요        {seconds:.0f}초", flush=True)
    print(f"  저장        {output_path}", flush=True)

    print_msgpack_structure(read_back, output_path)
    print("", flush=True)
    return {
        "removable": len(removal_results),
        "total": len(solid_ids),
        "trajectory_step_count": len(trajectory_steps),
        "output_path": output_path,
        "seconds": seconds,
    }


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: DictConfig):
    """설정을 검증하고 탐색을 실행한다."""
    usage = ("예: python main.py step_path=/path/to/assembly.stp "
             "output_path=/path/to/result.msgpack")
    if not config.step_path:
        raise SystemExit(f"step_path 가 비어 있다. 입력 STEP 경로를 지정하시오 — {usage}")
    if not config.output_path:
        raise SystemExit(f"output_path 가 비어 있다. 출력 msgpack 경로를 지정하시오 — {usage}")
    if not os.path.isfile(config.step_path):
        raise SystemExit(f"입력 STEP 파일이 없다: {config.step_path}")

    summary = execute_disassembly_search(
        step_path=config.step_path,
        output_path=config.output_path,
        worker_count=config.search.worker_count,
        iteration_count=config.search.iteration_count,
        does_sample_rotation=config.search.does_sample_rotation,
        random_seed=config.search.random_seed,
        time_budget_seconds=config.search.time_budget_seconds,
        max_interference_growth=config.max_interference_growth,
    )
    # 요약은 execute_disassembly_search 가 '요약' 구획으로 이미 출력한다.
    return summary


if __name__ == "__main__":
    main()
