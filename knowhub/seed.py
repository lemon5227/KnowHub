"""Idempotent, explicitly synthetic demonstration records and SOPs."""

from pathlib import Path


def seed(store, retriever):
    with store.db() as conn:
        for row in [
            (
                "ORD-1001",
                "示例客户 林女士",
                "delayed",
                "演示承运商：星桥物流。预计延迟48小时，尚未送达；需要物流专员跟进。",
            ),
            (
                "ORD-1002",
                "示例客户 陈先生",
                "paid",
                "订单已付款，仓库尚未发货；客户申请取消，但尚无退款审批结果。",
            ),
            (
                "ORD-1003",
                "示例客户 周女士",
                "delivered",
                "已签收10天，客户称商品质量问题；尚未提供照片或购买凭证。",
            ),
        ]:
            conn.execute("INSERT INTO orders VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING", row)
        for row in [
            (
                "TK-1042",
                "订单延迟，需要核实与跟进",
                "示例客户 林女士",
                "ORD-1001",
                "high",
                "客户表示订单未按时到达，请查询实际状态并提供处理建议。",
            ),
            (
                "TK-1043",
                "未发货订单取消咨询",
                "示例客户 陈先生",
                "ORD-1002",
                "normal",
                "客户咨询订单能否取消，以及退款何时到账。",
            ),
            (
                "TK-1044",
                "特殊运输赔付条款咨询",
                "示例客户 赵先生",
                None,
                "normal",
                "客户询问公司火星运输赔付政策，暂无订单号及可用政策。",
            ),
        ]:
            conn.execute(
                """INSERT INTO tickets(id,title,customer,order_id,priority,description)
                VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                row,
            )
    existing = {d["source"] for d in store.documents()}
    for path in sorted((Path(__file__).resolve().parent.parent / "examples/knowledge").glob("*.md")):
        source = "demo/" + path.name
        if source not in existing:
            retriever.ingest(path.stem, path.read_text(), source)
