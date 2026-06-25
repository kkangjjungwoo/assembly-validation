# Instructions
## Naming Convention
### Variables
General & Local Variable: 단일 알파벳 변수명(루프 인덱스 등은 제외) 및 축약 금지
* Good: `transformed_vertices`, `iteration_count`, `bounding_box`
* Bad: `v_tf`, `it_cnt`, `bbox`

Boolean Variable: 참 또는 거짓임을 직관적으로 파악할 수 있도록 의문형 접두어 명시
* Good: `is_watertight`, `has_intersection`, `can_extract_part`
* Bad: `collision_flag`, `watertight_check`, `success`

Lists, Dicts, Sets: 단어 자체를 복수형으로 표기하거나 데이터 구조를 유추할 수 있는 접미어 명시
* Good: `collision_objects`, `part_id_set`, `trajectory_frames`
* Bad: `part_group`, `data_array`

### Methods
Actions: 동사 원형 명시
* Good: `execute_search()`, `update_transformation_matrix()`, `serialize_to_binary()`
* Bad: `search_run()`, `matrix_modifier()`, `msgpack_save()`

Getters: `get_` 또는 `to_` 접두어 명시
* Good: `get_nearest_node()`, `get_minimum_clearance()`, `to_quaternion_array()`
* Bad: `fetch_node()`, `clearance_value()`, `quaternion_convert()`

Predicates: 논리값을 반환하는 메소드는 `is_`, `has_`, `check_`와 같은 접두어 명시
* Good: `is_valid_state()`, `has_touching_contact()`, `check_mesh_intersection()`
* Bad: `validate_state()`, `contact_evaluation()`, `intersection_test()`

### Classes
General: 명사 형태로 작성하고 해당 클래스의 정체성을 나타내는 접미어 명시
* Good: `RRTStarPlanner`, `AssemblyCollisionChecker`, `MsgpackTrajectorySerializer`
* Bad: `RunRRTStar`, `CollisionCheckingEngine`, `DataPacker`

Exception: `Exception` 접미어 명시
* Good: `MeshLoadException`, `InvalidMatrixException`
* Bad: `MeshLoadError`, `TransformationMatrixInverseError`, `PathNotFoundError`