from app.rag_query_expansion import build_retrieval_query


def test_build_retrieval_query_adds_company_guarantee_hints():
    query = build_retrieval_query("法定代表人超越内部授权签订担保合同，公司能否拒绝承担责任？", "商事", [])

    assert "公司法 第十六条" in query
    assert "民法典 第五百零四条" in query
    assert "民法典 第六百八十六条" in query


def test_build_retrieval_query_adds_minor_recharge_hints():
    query = build_retrieval_query("未成年人充值游戏花了很多钱，家长能要求平台退款吗？", "未成年人", [])

    assert "民法典 第十九条" in query
    assert "民法典 第二十条" in query
    assert "未成年人保护法 第七十四条" in query
    assert "未成年人保护法 第七十五条" in query


def test_build_retrieval_query_adds_evidence_hints_for_oral_contract():
    query = build_retrieval_query("口头约定合伙做生意没有书面合同，亏损后对方不认账怎么办？", "商事", [])

    assert "民法典 第四百六十九条" in query
    assert "民事诉讼法 第六十七条" in query
    assert "民法典 第九百七十二条" in query
    assert "民法典 第九百七十三条" in query


def test_build_retrieval_query_adds_labor_renewal_hints():
    query = build_retrieval_query("劳动合同到期后继续上班两个月，公司没续签书面合同怎么办？", "劳动", [])

    assert "劳动合同法 第十四条" in query
    assert "劳动合同法 第八十二条" in query


def test_build_retrieval_query_adds_telefraud_account_hints():
    query = build_retrieval_query("银行卡被用于跑分后被冻结，公安让我说明资金来源，我该准备什么证据？", "刑事", [])

    assert "反电信网络诈骗法" in query
    assert "银行账户" in query
    assert "刑事诉讼法" in query
    assert "调取证据" in query


def test_build_retrieval_query_adds_enforcement_transfer_hints():
    query = build_retrieval_query("申请强制执行后发现被执行人转移财产，可以要求法院采取哪些措施？", "民事诉讼", [])

    assert "民事诉讼法 第二百五十二条" in query
    assert "民事诉讼法 第二百六十五条" in query
    assert "民事诉讼法 第二百六十六条" in query


def test_build_retrieval_query_leaves_unmatched_question_unchanged():
    question = "诉讼时效一般是多久？"

    assert build_retrieval_query(question, "综合", []) == question
