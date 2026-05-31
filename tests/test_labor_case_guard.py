from app.labor_case_guard import is_labor_case_context


def test_labor_primary_domain_allows_action():
    assert is_labor_case_context({"primary_domain": "劳动"})


def test_labor_domain_alias_allows_action():
    assert is_labor_case_context({"domains": ["labor"], "case_type": "劳动争议"})


def test_labor_keywords_allow_action_without_structured_domain():
    case_state = {"raw_input": "老板口头辞退我，还拖欠工资两个月。"}
    assert is_labor_case_context(case_state)


def test_non_labor_domain_blocks_broad_company_keyword():
    case_state = {
        "primary_domain": "公司",
        "case_type": "公司纠纷",
        "raw_input": "公司股东之间发生出资纠纷，想了解赔偿金风险。",
    }
    assert not is_labor_case_context(case_state)


def test_marriage_salary_question_does_not_show_labor_action():
    case_state = {
        "primary_domain": "婚姻",
        "case_type": "婚姻家庭",
        "raw_input": "离婚时夫妻一方工资收入是否属于夫妻共同财产？",
    }
    assert not is_labor_case_context(case_state)


def test_misclassified_contract_with_labor_anchor_allows_action():
    case_state = {
        "primary_domain": "合同",
        "case_type": "合同纠纷",
        "raw_input": "公司一直没有签劳动合同，现在把我辞退了。",
    }
    assert is_labor_case_context(case_state)

