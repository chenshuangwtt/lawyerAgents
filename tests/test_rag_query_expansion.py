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


def test_build_retrieval_query_adds_trade_secret_hints_for_client_list():
    query = build_retrieval_query("离职高管带走客户名单另开公司，原公司可以追责吗？", "商事", [])

    assert "反不正当竞争法 第十条" in query
    assert "劳动合同法 第二十三条" in query
    assert "劳动合同法 第二十四条" in query
    assert "公司法 第一百八十条" in query
    assert "公司法 第一百八十一条" in query


def test_build_retrieval_query_adds_trade_secret_hints_for_departing_employee():
    query = build_retrieval_query("公司离职员工带走客户个人信息另行营销，涉及哪些法律责任？", "网络与数据", [])

    assert "反不正当竞争法 第十条" in query
    assert "劳动合同法 第二十三条" in query
    assert "劳动合同法 第二十四条" in query
    assert "个人信息保护法 第二条" in query
    assert "个人信息保护法 第六十六条" in query
    assert "公司法 第一百八十条" not in query
    assert "公司法 第一百八十一条" not in query


def test_build_retrieval_query_adds_civil_code_hints_for_personal_info():
    query = build_retrieval_query("App未经同意读取通讯录并向第三方推送广告，用户可以要求删除和赔偿吗？", "网络与数据", [])

    assert "民法典 第一千零三十四条" in query
    assert "民法典 第一千零三十五条" in query
    assert "民法典 第一千零三十六条" in query
    assert "个人信息保护法 第四条" in query


def test_build_retrieval_query_adds_criminal_data_hints_for_deleted_system_data():
    query = build_retrieval_query("员工离职后带走客户资料并删除公司系统数据，公司能要求赔偿并报警吗？", "商事", [])

    assert "反不正当竞争法 第十条" in query
    assert "刑法 第二百五十三条之一" in query
    assert "刑法 第二百八十五条" in query
    assert "刑法 第二百八十六条" in query


def test_build_retrieval_query_does_not_trigger_hints_from_law_names_only():
    query = build_retrieval_query(
        "法定代表人超越内部授权签订担保合同，公司能否拒绝承担责任？",
        "商事",
        ["中华人民共和国个人信息保护法", "中华人民共和国刑法"],
    )

    assert "公司法 第十六条" in query
    assert "个人信息保护法 第二条" not in query
    assert "刑法 第二百八十五条" not in query


def test_build_retrieval_query_adds_enforcement_transfer_hints():
    query = build_retrieval_query("申请强制执行后发现被执行人转移财产，可以要求法院采取哪些措施？", "民事诉讼", [])

    assert "民事诉讼法 第二百五十二条" in query
    assert "民事诉讼法 第二百六十五条" in query
    assert "民事诉讼法 第二百六十六条" in query


def test_build_retrieval_query_leaves_unmatched_question_unchanged():
    question = "诉讼时效一般是多久？"

    assert build_retrieval_query(question, "综合", []) == question
