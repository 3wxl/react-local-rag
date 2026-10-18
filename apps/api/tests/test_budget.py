"""预算守卫（M3 验收：超预算请求被拒绝且说明原因）。

三层断言：
1. **预检拒绝**（HTTP 422 budget_exceeded）：超标的请求连一次模型都不调；
2. **运行中拒绝**（图内 charge）：真实用量超限时抛 BudgetExceeded，
   且它必须能**穿过 LangGraph** 原样冒到调用方（否则路由映射不到 422）；
3. **归因**：detail.budget.exceeded 指出超的是 tokens / steps / cost_usd 哪一道。
"""

from __future__ import annotations

import dataclasses

import pytest
from fastapi.testclient import TestClient

from .test_qa_graph import CORPUS_DOCS, POSITIVE_Q, _answer, _ingest


@pytest.fixture(scope="module")
def client() -> TestClient:
    """共享 app + 一份语料。

    语料是必需的：路由先校验"索引已就绪"（没有文档直接 503），预检发生在之后。
    断言只针对预检结果（与检索统计无关），所以语料被别处增删不影响本文件。
    """
    from app.main import app

    with TestClient(app) as test_client:
        _ingest(test_client, CORPUS_DOCS)
        yield test_client


def test_preflight_rejects_over_token_budget(client: TestClient, monkeypatch) -> None:
    ctx = client.app.state.ctx
    monkeypatch.setattr(ctx.settings, "request_token_budget", 8)

    response = client.post("/v1/answer", json={"question": POSITIVE_Q})

    assert response.status_code == 422, response.text
    body = response.json()["error"]
    assert body["code"] == "budget_exceeded"
    budget = body["detail"]["budget"]
    assert budget["exceeded"] == "tokens"
    assert budget["stage"] == "preflight"
    assert budget["estimated"] is True
    assert budget["limit"]["tokens"] == 8
    assert budget["used"]["tokens"] > 8
    # 提示要能指导下一步动作（而不是只报一个数字）
    assert "REQUEST_TOKEN_BUDGET" in body["message"]


def test_preflight_rejects_over_cost_budget(client: TestClient, monkeypatch) -> None:
    """配了价目（接云端通道的口径）后，成本上限同样在预检生效。"""
    ctx = client.app.state.ctx
    monkeypatch.setattr(ctx.settings, "cost_per_1k_prompt_usd", 1.0)
    monkeypatch.setattr(ctx.settings, "cost_per_1k_completion_usd", 1.0)
    monkeypatch.setattr(ctx.settings, "request_cost_limit_usd", 0.000001)

    response = client.post("/v1/answer", json={"question": POSITIVE_Q})

    assert response.status_code == 422, response.text
    budget = response.json()["error"]["detail"]["budget"]
    assert budget["exceeded"] == "cost_usd"
    assert budget["estimated"] is True


async def test_runtime_token_budget_raises_through_graph(qa_env) -> None:
    """真实用量超限：charge 抛出 → 穿过 LangGraph 原样冒到调用方。"""
    from app.generation.base import Completion
    from app.graphs.budget import BudgetExceeded

    class _HugeTokenProvider:
        name = "stub"
        model = "stub-huge"

        def available(self) -> bool:
            return True

        def complete(self, prompt: str, *, max_new_tokens: int = 512) -> Completion:
            # 报告真实用量（远超预检估的上界）：预检放行，记账时拒绝
            return Completion(
                text="答案[1]。",
                prompt_tokens=50_000,
                completion_tokens=50_000,
                model=self.model,
                provider=self.name,
                sampling={},
            )

    env = qa_env
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(env.settings, "request_token_budget", 4096)
    try:
        with pytest.raises(BudgetExceeded) as excinfo:
            await _answer(env, POSITIVE_Q, provider=_HugeTokenProvider())
    finally:
        monkeypatch.undo()

    detail = excinfo.value.detail["budget"]
    assert detail["exceeded"] == "tokens"
    assert detail["stage"] == "generate"
    assert detail["estimated"] is False
    assert detail["used"]["tokens"] == 100_000
    assert "已用" in str(excinfo.value)


async def test_runtime_step_cap(qa_env) -> None:
    """步数上限：两次模型调用（改写 + 生成）在 GRAPH_MAX_STEPS=1 时被拒。"""
    from app.generation.fake import FakeProvider
    from app.graphs.budget import BudgetExceeded

    env = qa_env
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(env.settings, "graph_max_steps", 1)
    try:
        # 预检不看步数，所以这里必须让真实调用走到第二步
        deps = dataclasses.replace(env.deps, provider=FakeProvider())
        with pytest.raises(BudgetExceeded) as excinfo:
            from app.graphs.qa_graph import build_qa_graph
            from app.services.answer_service import AnswerService
            from app.store import Filters

            service = AnswerService(build_qa_graph(deps, None), settings=env.settings)
            await service.answer(
                POSITIVE_Q,
                version=env.version,
                top_k=3,
                filters=Filters(tenant_id="default"),
                rewrite=True,
            )
    finally:
        monkeypatch.undo()

    detail = excinfo.value.detail["budget"]
    assert detail["exceeded"] == "steps"
    assert detail["limit"]["steps"] == 1
    assert detail["used"]["steps"] == 2
