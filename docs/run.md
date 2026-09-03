# 화면과 조작

설치와 실행은 [빠른 시작](getting-started.md)을 참고한다.

전방 3단 선반에서 물체를 집어 좌우 테이블의 색상 박스로 옮기려면
[3단 선반 색상 분류](shelf-sort-task.md)를 참고한다.

## 화면

- **MuJoCo main**: 3D scene, target marker, gizmo, 상태 창
- **Control Center**: Target, Task Space, 양팔, Pose Graph, IK Solver, Robot/Grasp
- **Diagnostics**: Kinematic Tree, Joint Monitor

도구 창은 drag/resize/close할 수 있다. 닫은 창은 **Status & Windows**에서 다시 열고,
**Detach tools outside** / **Return tools to main**으로 배치를 바꾼다.

## 키보드와 마우스

| 입력 | 기능 |
|---|---|
| Mouse left/right drag, wheel | orbit, pan, zoom |
| `Up` / `Down` | base 전진 / 후진 |
| `Left` / `Right` | base yaw |
| `[` / `]` | base strafe |
| `Q` / `E` | lift 하강 / 상승 |
| `R` | can reset |
| `G` | contact point/force 표시 |
| `V` | collision geometry/CBF 표시 |
| `C` | camera preset 전환 |

ImGui 입력 필드에 focus가 있으면 drive key가 전달되지 않는다. 3D scene을 클릭해 focus를
돌린다.

## 상태 창

| 표시 | 의미 |
|---|---|
| controller / marker | 현재 target controller와 조작 대상 |
| sim / wall / Hz | simulation·실제 시간과 loop 주파수 |
| IK err L/R | 실제 손과 target의 위치 오차; FK 팔은 `FK` |
| Base x/y/yaw | 실제 base pose |
| Whole-body IK | base/lift 자동 참여 상태 |
| body cmd | swerve에 전달되는 최종 body twist |
| Collision CBF | 활성 pair, 최소 거리, 남은 위반량 |

## Target

| Controller | 동작 |
|---|---|
| MoveL | 오른손과 왼손 target을 독립 이동 |
| Bimanual MoveL | Capture 뒤 virtual object로 양손을 함께 이동 |

선택 marker를 jog 버튼이나 3D gizmo로 움직인다. 기본 step은 위치 0.005 m, 회전 2°다.

## Task Space

오른손 또는 왼손의 MuJoCo world-frame 절대 pose를 숫자로 입력한다.

1. 손을 선택한다.
2. `X/Y/Z (m)`, `Roll/Pitch/Yaw (deg)`를 입력한다.
3. **Apply Target**을 누른다.

**Load Current Pose**는 측정 pose, **Load Target Pose**는 활성 목표를 입력 칸에 복사한다.
적용 시 해당 팔은 IK로 전환된다. Captured Bimanual 상태라면 독립 손 명령을 위해 먼저
release한다.

## Right Arm / Left Arm

| 모드 | 입력 |
|---|---|
| IK | home 기준 XYZ offset과 local RPY delta |
| FK | J1~J7 관절각(deg) |

IK→FK는 현재 관절 목표를 복사하고, FK→IK는 현재 측정 손 pose를 target으로 잡아 전환
점프를 막는다.

## Robot / Grasp

- **Grab/Release**: grasp와 thumb target을 ramp
- **grasp/thumb slider**: finger synergy 직접 조절
- **Whole-body ON**: base + lift + IK 팔 자동 참여
- **Whole-body OFF**: IK 팔만 참여; keyboard base와 수동 lift는 계속 사용 가능
- **Lift target**: -0.5~0.0 m
- **Reset / Contact / Collision / Camera**: `R/G/V/C`와 동일

캔은 손에 강제로 붙지 않는다. 접촉력과 마찰이 실제 물체 이동을 만든다.

## Collision 표시

| 색 | signed distance |
|---|---:|
| 노랑 | 1~3 cm |
| 주황 | 0~1 cm |
| 빨강 | < 0 cm |

`slack`은 남은 최대 CBF 위반 속도다.

## 기본 작업 순서

### 한 손

1. MoveL, 해당 팔 IK, Whole-body OFF
2. 작은 jog 또는 Task Space target 적용
3. IK error와 collision 표시 확인

### 양손

1. 양팔 IK, Bimanual MoveL
2. Capture Grasp, Whole-body ON
3. virtual object 이동
4. base command, 양손 error와 collision 표시 확인

수동 base 이동 후에는 키를 놓고 물리 제동이 끝난 다음 새 target을 입력한다.

## 검증

```bash
python3 tests/test_phase_6.py
python3 tests/test_whole_body.py
mkdocs build --strict
```

증상별 확인 순서는 [문제 해결](troubleshooting.md)에 있다.
