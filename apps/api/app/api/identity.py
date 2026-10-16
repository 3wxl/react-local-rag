"""请求身份（ACL 的输入）。

M1/M2 阶段是**单租户硬编码**：契约 §5 已写明这个接缝（"M1 单租户硬编码 tenant_id；
M2 起从 token 解出"），本模块就是那个"解出身份"的位置 —— 换 JWT/OIDC 时只改 `get_identity`，
路由与存储层不动。

为什么身份要在这里、而不是让客户端传：
客户端能传的只有**收紧**条件（doc_ids / tags），越权参数一律被忽略。
如果身份来自请求体，等于把授权交给调用方自己声明。
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

#: 密级取值（数值越大越敏感）：与 documents.classification 同口径
CLASSIFICATION_PUBLIC = 0
CLASSIFICATION_INTERNAL = 1
CLASSIFICATION_CONFIDENTIAL = 2


@dataclass(frozen=True)
class Identity:
    """调用方身份：决定**看得见哪些文档**（在召回层生效）。"""

    tenant_id: str
    #: 所属部门；空 = 不受部门限制（管理员/公司级视图）
    department: str = ""
    #: 可访问的最大密级
    clearance: int = CLASSIFICATION_CONFIDENTIAL
    #: 预留：审计与配额要用（M2 只记录，不做判定）
    subject: str = "local"


#: 单租户默认身份（无认证时的兜底）：能看全部密级、不受部门限制
DEFAULT_IDENTITY = Identity(tenant_id="default", department="", clearance=CLASSIFICATION_CONFIDENTIAL)


def get_identity(request: Request) -> Identity:
    """从请求解析身份。

    当前实现返回默认身份（单租户、无认证）——但**签名与位置已经对了**：
    换成 token 解析时，只需在这里读 Authorization 头并构造 Identity，
    路由与 SQL 都不用改（它们只认 Identity 这个结构）。
    """
    override = getattr(request.state, "identity", None)
    if isinstance(override, Identity):
        return override
    return DEFAULT_IDENTITY
