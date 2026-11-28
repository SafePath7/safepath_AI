"""
경로 이탈 상태 (백엔드 → AI, POST /api/ai/route-status).

백엔드가 귀가 경로와 현재 위치로 이탈을 판단하고, 상태가 바뀔 때만 알려 준다.
    - 신호를 받은 적 없는 세션은 이탈 안 함(기본).
    - 이탈 신호 뒤 다른 신호가 없으면 계속 이탈로 본다.
    - 경로로 돌아오거나 경로를 수정하면 deviated=false (재진입)로 다시 알려 준다.
AI는 세션별로 메모리에만 둔다. 그 세션 요청(분석·경로 신호)이 ROUTE_IDLE_SEC(기본 10분) 동안 없으면 지운다.
AI 서버가 다시 시작되면 상태를 잊고 이탈 안 함으로 시작한다(백엔드가 이탈 중인 세션을 다시 보내 주면 복구된다).

위험도에는 보조 신호로만 쓴다 (app/risk.py): 이탈만으로는 위험이 아니고, 다른 신호가 애매할 때 조금 올린다.
"""

import threading
import time

from app import config


def empty_route(status: str = "on_route") -> dict:
    """분석 응답의 route 형식. status: on_route(이탈 안 함·신호 없음) / deviated."""
    return {"status": status, "deviated": status == "deviated", "since": None, "distance_m": None, "applied": 0.0}


class RouteStateStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._states: dict[str, dict] = {}

    def _cleanup(self, now: float) -> None:
        for k in [k for k, s in self._states.items() if now - s["seen"] > config.ROUTE_IDLE_SEC]:
            del self._states[k]

    def set(self, session_id: str, deviated: bool, at_ms: int | None = None, distance_m: float | None = None) -> dict:
        """상태를 바꾼다. at_ms가 지금 가진 상태보다 이전이면(늦게 도착한 재전송) 무시한다. 현재 상태를 돌려준다."""
        now = time.monotonic()
        with self._lock:
            self._cleanup(now)
            cur = self._states.get(session_id)
            if cur and at_ms is not None and cur["at_ms"] is not None and at_ms < cur["at_ms"]:
                cur["seen"] = now
                return self._public(cur)
            if cur and cur["deviated"] == deviated:
                since = cur["since"]  # 같은 상태 재전송: 시작 시각은 그대로
            else:
                since = at_ms if at_ms is not None else int(time.time() * 1000)
            state = {"deviated": deviated, "at_ms": at_ms, "since": since, "distance_m": distance_m, "seen": now}
            self._states[session_id] = state
            return self._public(state)

    def get(self, session_id: str | None) -> dict:
        """분석 때 부른다. 그 세션이 살아 있다는 표시도 겸한다."""
        if not session_id:
            return empty_route()
        now = time.monotonic()
        with self._lock:
            self._cleanup(now)
            state = self._states.get(session_id)
            if state is None:
                return empty_route()
            state["seen"] = now
            return self._public(state)

    @staticmethod
    def _public(state: dict) -> dict:
        r = empty_route("deviated" if state["deviated"] else "on_route")
        if state["deviated"]:
            r.update(since=state["since"], distance_m=state["distance_m"])
        return r

    def clear(self) -> None:
        with self._lock:
            self._states.clear()


store = RouteStateStore()
