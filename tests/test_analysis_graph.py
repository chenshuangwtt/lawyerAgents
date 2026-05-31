"""测试案情分析图中的法律过滤与推荐逻辑。"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis_graph import (
    filter_law_names_for_case,
    infer_law_hints,
    is_law_relevant_to_case,
    _get_doc_title,
)

# ─── filter_law_names_for_case ─────────────────────────────────

CLIENT_LIST_INPUT = "离职员工把客户名单卖给竞争对手，公司怎么起诉要求赔偿？"

IRRELEVANT_LAWS = [
    "最高人民法院关于适用《中华人民共和国民法典》婚姻家庭编的解释（一）",
    "中华人民共和国监察法",
    "最高人民法院关于产品侵权案件的受害人能否以产品的商标所有人为被告提起民事诉讼的批复",
]

RELEVANT_LAWS = [
    "中华人民共和国反不正当竞争法",
    "最高人民法院关于适用《中华人民共和国反不正当竞争法》若干问题的解释",
    "中华人民共和国民法典",
    "中华人民共和国劳动合同法",
]


class TestFilterLawNamesForCase:
    """测试 filter_law_names_for_case 逻辑"""

    def test_client_list_case_filters_marriage_law(self):
        """客户名单泄露案 → 婚姻家庭编解释应被过滤"""
        filtered = filter_law_names_for_case(CLIENT_LIST_INPUT, IRRELEVANT_LAWS)
        assert "最高人民法院关于适用《中华人民共和国民法典》婚姻家庭编的解释（一）" not in filtered

    def test_client_list_case_filters_supervision_law(self):
        """客户名单泄露案 → 监察法应被过滤"""
        filtered = filter_law_names_for_case(CLIENT_LIST_INPUT, IRRELEVANT_LAWS)
        assert "中华人民共和国监察法" not in filtered

    def test_client_list_case_filters_product_infringement(self):
        """客户名单泄露案 → 产品侵权批复应被过滤"""
        filtered = filter_law_names_for_case(CLIENT_LIST_INPUT, IRRELEVANT_LAWS)
        assert "最高人民法院关于产品侵权案件的受害人能否以产品的商标所有人为被告提起民事诉讼的批复" not in filtered

    def test_client_list_case_keeps_relevant_laws(self):
        """客户名单泄露案 → 相关法律应保留"""
        filtered = filter_law_names_for_case(CLIENT_LIST_INPUT, RELEVANT_LAWS)
        for law in RELEVANT_LAWS:
            assert law in filtered

    def test_marriage_case_keeps_marriage_law(self):
        """真正婚姻案 → 婚姻家庭编解释不应被过滤"""
        input_text = "丈夫出轨，我要离婚，孩子抚养权怎么争取？"
        filtered = filter_law_names_for_case(input_text, IRRELEVANT_LAWS)
        assert "最高人民法院关于适用《中华人民共和国民法典》婚姻家庭编的解释（一）" in filtered

    def test_marriage_case_filters_supervision_law(self):
        """真正婚姻案 → 监察法仍应被过滤"""
        input_text = "丈夫出轨，我要离婚，孩子抚养权怎么争取？"
        filtered = filter_law_names_for_case(input_text, IRRELEVANT_LAWS)
        assert "中华人民共和国监察法" not in filtered

    def test_supervision_case_keeps_supervision_law(self):
        """真正监察案 → 监察法不应被过滤"""
        input_text = "公职人员利用职务之便收受贿赂，监察委如何立案调查？"
        filtered = filter_law_names_for_case(input_text, IRRELEVANT_LAWS)
        assert "中华人民共和国监察法" in filtered

    def test_supervision_case_filters_marriage_law(self):
        """真正监察案 → 婚姻家庭编解释应被过滤"""
        input_text = "公职人员利用职务之便收受贿赂，监察委如何立案调查？"
        filtered = filter_law_names_for_case(input_text, IRRELEVANT_LAWS)
        assert "最高人民法院关于适用《中华人民共和国民法典》婚姻家庭编的解释（一）" not in filtered

    def test_empty_input_returns_empty(self):
        """空输入时返回空列表"""
        assert filter_law_names_for_case("", []) == []
        assert filter_law_names_for_case(None, None) == []

    def test_no_trigger_words_passes_through(self):
        """不含任何触发词的案情 + 无关法律 → 无关法律被过滤"""
        filtered = filter_law_names_for_case("今天天气怎么样", IRRELEVANT_LAWS)
        assert filtered == []


# ─── infer_law_hints ───────────────────────────────────────────

class TestInferLawHints:
    """测试 infer_law_hints 逻辑"""

    def test_client_list_case_hints(self):
        """客户名单泄露案 → 应包含反不正当竞争法、民法典"""
        hints = infer_law_hints(CLIENT_LIST_INPUT)
        assert "中华人民共和国反不正当竞争法" in hints
        assert "最高人民法院关于适用《中华人民共和国反不正当竞争法》若干问题的解释" in hints
        assert "中华人民共和国民法典" in hints

    def test_client_list_case_hints_labor_law(self):
        """含"离职员工"关键词 → 应包含劳动合同法"""
        hints = infer_law_hints(CLIENT_LIST_INPUT)
        assert "中华人民共和国劳动合同法" in hints

    def test_customer_data_hints(self):
        """含"客户资料"关键词 → 应包含反不正当竞争法"""
        hints = infer_law_hints("员工把客户资料拷贝走了，公司怎么处理？")
        assert "中华人民共和国反不正当竞争法" in hints
        assert "中华人民共和国民法典" in hints

    def test_confidentiality_agreement_hints(self):
        """含"保密协议"关键词 → 应包含劳动合同法"""
        hints = infer_law_hints("员工违反保密协议，需要承担什么责任？")
        assert "中华人民共和国劳动合同法" in hints

    def test_no_hints_for_general_input(self):
        """普通民事纠纷 → 不应触发任何 hints"""
        hints = infer_law_hints("欠钱不还怎么办")
        assert hints == []


# ─── Mock Document ─────────────────────────────────────────────

class MockDoc:
    """模拟 LangChain Document 对象"""
    def __init__(self, metadata: dict):
        self.metadata = metadata


class TestGetDocTitle:
    """测试 _get_doc_title 逻辑"""

    def test_law_name_priority(self):
        """law_name 优先于 title 和 source"""
        doc = MockDoc({"law_name": "民法典", "title": "民法通则", "source": "民法"})
        assert _get_doc_title(doc) == "民法典"

    def test_fallback_to_title(self):
        """无 law_name 时使用 title"""
        doc = MockDoc({"title": "合同法", "source": "合同"})
        assert _get_doc_title(doc) == "合同法"

    def test_fallback_to_source(self):
        """无 law_name 和 title 时使用 source"""
        doc = MockDoc({"source": "侵权责任法"})
        assert _get_doc_title(doc) == "侵权责任法"

    def test_empty_metadata(self):
        """空 metadata 返回空字符串"""
        doc = MockDoc({})
        assert _get_doc_title(doc) == ""

    def test_none_metadata(self):
        """metadata 为 None 时返回空字符串"""
        doc = MockDoc(None)
        assert _get_doc_title(doc) == ""


# ─── is_law_relevant_to_case ───────────────────────────────────

class TestIsLawRelevantToCase:
    """测试 is_law_relevant_to_case 逻辑"""

    def test_marriage_doc_filtered_for_client_case(self):
        """客户名单泄露案 → 婚姻家庭编文档应返回 False"""
        doc = MockDoc({"law_name": "最高人民法院关于适用《中华人民共和国民法典》婚姻家庭编的解释（一）"})
        assert is_law_relevant_to_case(CLIENT_LIST_INPUT, doc) is False

    def test_supervision_doc_filtered_for_client_case(self):
        """客户名单泄露案 → 监察法文档应返回 False"""
        doc = MockDoc({"law_name": "中华人民共和国监察法"})
        assert is_law_relevant_to_case(CLIENT_LIST_INPUT, doc) is False

    def test_product_infringement_doc_filtered_for_client_case(self):
        """客户名单泄露案 → 产品侵权批复文档应返回 False"""
        doc = MockDoc({"law_name": "最高人民法院关于产品侵权案件的受害人能否以产品的商标所有人为被告提起民事诉讼的批复"})
        assert is_law_relevant_to_case(CLIENT_LIST_INPUT, doc) is False

    def test_competition_law_doc_allowed_for_client_case(self):
        """客户名单泄露案 → 反不正当竞争法文档应返回 True"""
        doc = MockDoc({"law_name": "中华人民共和国反不正当竞争法"})
        assert is_law_relevant_to_case(CLIENT_LIST_INPUT, doc) is True

    def test_civil_code_doc_allowed_for_client_case(self):
        """客户名单泄露案 → 民法典文档应返回 True"""
        doc = MockDoc({"law_name": "中华人民共和国民法典"})
        assert is_law_relevant_to_case(CLIENT_LIST_INPUT, doc) is True

    def test_labor_law_doc_allowed_for_client_case(self):
        """客户名单泄露案 → 劳动合同法文档应返回 True"""
        doc = MockDoc({"law_name": "中华人民共和国劳动合同法"})
        assert is_law_relevant_to_case(CLIENT_LIST_INPUT, doc) is True

    def test_marriage_doc_allowed_for_marriage_case(self):
        """真正离婚案 → 婚姻家庭编文档应返回 True"""
        doc = MockDoc({"law_name": "最高人民法院关于适用《中华人民共和国民法典》婚姻家庭编的解释（一）"})
        assert is_law_relevant_to_case("我要离婚，孩子归谁", doc) is True

    def test_supervision_doc_allowed_for_supervision_case(self):
        """真正监察案 → 监察法文档应返回 True"""
        doc = MockDoc({"law_name": "中华人民共和国监察法"})
        assert is_law_relevant_to_case("公职人员受贿怎么调查", doc) is True

    def test_unknown_law_allowed(self):
        """未知法律名称 → 默认返回 True"""
        doc = MockDoc({"law_name": "中华人民共和国刑法"})
        assert is_law_relevant_to_case(CLIENT_LIST_INPUT, doc) is True

    def test_empty_user_input_defaults_true(self):
        """空用户输入 → 默认返回 True（不误杀）"""
        doc = MockDoc({"law_name": "中华人民共和国刑法"})
        assert is_law_relevant_to_case("", doc) is True


# ─── Prompt content verification ──────────────────────────────

from app.analysis_graph import (
    DECOMPOSE_PROMPT,
    REPORT_PROMPT,
    _default_analysis_section,
    ANALYSIS_SECTION_TITLES,
)


class TestDECOMPOSEPromptRules:
    """验证 DECOMPOSE_PROMPT 中的案件类型规则"""

    def test_contains_compound_case_rule(self):
        """应包含复合型纠纷识别规则"""
        assert "复合型纠纷" in DECOMPOSE_PROMPT

    def test_contains_legal_relationships_example(self):
        """legal_relationships 字段说明应包含复合表达示例"""
        assert "劳动合同/保密协议违约 + 侵犯商业秘密/不正当竞争" in DECOMPOSE_PROMPT

    def test_contains_not_only_labor_dispute(self):
        """不应仅将案件归为普通劳动争议"""
        assert "普通劳动争议" in DECOMPOSE_PROMPT

    def test_contains_labor_rights_caveat(self):
        """应说明仅劳动权益问题才归为普通劳动争议"""
        assert "劳动权益问题" in DECOMPOSE_PROMPT

    def test_labor_domain_still_allowed(self):
        """劳动领域仍应在可选领域中"""
        assert "劳动" in DECOMPOSE_PROMPT

    def test_pure_labor_case_not_forced_to_trade_secret(self):
        """普通工资纠纷不应强行归为商业秘密"""
        # DECOMPOSE_PROMPT 中的关键词触发规则基于特定关键词
        assert "只有当案情核心是工资" in DECOMPOSE_PROMPT


class TestREPORTPromptRules:
    """验证 REPORT_PROMPT 中的案件类型、法律关系、处理路径规则"""

    def test_contains_compound_case_type(self):
        """应包含复合型纠纷案件类型规则"""
        assert "复合型纠纷" in REPORT_PROMPT

    def test_contains_no_labor_only_case_type(self):
        """不得仅输出"劳动争议 - 劳动合同纠纷" """
        assert "劳动争议 - 劳动合同纠纷" in REPORT_PROMPT  # 在否定规则中出现
        assert "不得仅输出" in REPORT_PROMPT  # 确认是否定形式

    def test_legal_relationships_distinction(self):
        """涉及法律关系应区分内部与外部"""
        assert "内部劳动/合同关系" in REPORT_PROMPT
        assert "外部侵权/竞争关系" in REPORT_PROMPT

    def test_legal_relationships_fields(self):
        """客户名单案件的主要领域应包括商业秘密保护等"""
        assert "商业秘密保护" in REPORT_PROMPT
        assert "不正当竞争" in REPORT_PROMPT
        assert "劳动合同/保密协议违约" in REPORT_PROMPT

    def test_path_distinction(self):
        """处理路径应区分劳动仲裁和法院诉讼"""
        assert "劳动仲裁" in REPORT_PROMPT
        assert "侵犯商业秘密/不正当竞争民事诉讼" in REPORT_PROMPT
        assert "证据保全" in REPORT_PROMPT
        assert "行为保全" in REPORT_PROMPT

    def test_path_not_only_arbitration(self):
        """不能以劳动仲裁为唯一处理路径"""
        assert "唯一处理路径" in REPORT_PROMPT

    def test_disclaimer_generic(self):
        """免责声明应为通用版，不写死劳动争议"""
        assert "劳动争议案件受证据" not in REPORT_PROMPT
        assert "申请仲裁、提起诉讼、申请保全或报案" in REPORT_PROMPT
        assert "合同约定" in REPORT_PROMPT
        assert "保密措施" in REPORT_PROMPT


class TestDefaultAnalysisSection:
    """验证 _default_analysis_section 兜底内容"""

    CLIENT_LIST_INPUT = "离职员工把客户名单卖给竞争对手，公司怎么起诉要求赔偿？"

    def test_case_summary_not_labor_specific(self):
        """案情摘要不应包含劳动仲裁、未签劳动合同等劳动专属内容"""
        text = _default_analysis_section("🧾 案情摘要", self.CLIENT_LIST_INPUT)
        assert "合同责任" in text
        assert "侵权责任" in text
        # 不应硬编码劳动争议
        assert "未签劳动合同" not in text
        assert "违法解除" not in text
        assert "经济补偿" not in text

    def test_legal_relationships_generic(self):
        """涉及法律关系应为通用版"""
        text = _default_analysis_section("🏷️ 涉及法律关系", self.CLIENT_LIST_INPUT)
        assert "合同责任" in text
        assert "侵权责任" in text
        assert "行政监管" in text
        assert "劳动人事争议仲裁" not in text

    def test_treatment_path_generic(self):
        """处理路径应为通用版"""
        text = _default_analysis_section("🛠️ 处理路径", self.CLIENT_LIST_INPUT)
        assert "固定证据" in text
        assert "协商" in text
        assert "投诉举报" in text
        assert "仲裁或诉讼" in text
        assert "证据保全" in text
        assert "行为保全" in text
        # 不应只写劳动仲裁
        assert "劳动人事争议仲裁" not in text

    def test_evidence_list_generic(self):
        """证据清单应为通用版"""
        text = _default_analysis_section("📌 证据清单", self.CLIENT_LIST_INPUT)
        assert "合同" in text
        assert "协议" in text
        assert "聊天记录" in text
        assert "邮件" in text
        assert "损失证明" in text

    def test_next_steps_generic(self):
        """下一步建议应为通用版"""
        text = _default_analysis_section("📝 下一步建议", self.CLIENT_LIST_INPUT)
        assert "停止侵害" in text
        assert "赔偿损失" in text
        assert "仲裁" in text
        assert "诉讼" in text
        assert "举报" in text
        assert "报案" in text

    def test_disclaimer_generic(self):
        """免责声明应为通用版"""
        text = _default_analysis_section("📜 免责声明", self.CLIENT_LIST_INPUT)
        assert "劳动争议" not in text
        assert "劳动仲裁" not in text
        assert "合同约定" in text
        assert "地区裁判口径" in text
        assert "建议采取法律行动前咨询专业律师" in text

    def test_needs_more_info_generic(self):
        """需要补充的信息应为通用版"""
        text = _default_analysis_section("❓ 需要补充的信息", self.CLIENT_LIST_INPUT)
        assert "合同" in text
        assert "证据" in text
        assert "损失" in text


# ─── 食品安全法过滤 ─────────────────────────────────────────


class TestFilterFoodSafetyLaw:
    """测试 食品安全法 过滤逻辑"""

    CLIENT_LIST_INPUT = "离职员工把客户名单卖给竞争对手，公司怎么起诉要求赔偿？"
    FOOD_SAFETY_LAW = "中华人民共和国食品安全法"
    RELEVANT_LAWS = [
        "中华人民共和国反不正当竞争法",
        "最高人民法院关于适用《中华人民共和国反不正当竞争法》若干问题的解释",
        "中华人民共和国民法典",
        "中华人民共和国劳动合同法",
    ]

    def test_food_safety_filtered_for_client_list_case(self):
        """客户名单泄露案 → 食品安全法应被过滤"""
        law_names = self.RELEVANT_LAWS + [self.FOOD_SAFETY_LAW]
        filtered = filter_law_names_for_case(self.CLIENT_LIST_INPUT, law_names)
        assert self.FOOD_SAFETY_LAW not in filtered

    def test_relevant_laws_kept_when_filtering_food_safety(self):
        """客户名单泄露案 → 反不正当竞争法、民法典、劳动合同法仍保留"""
        law_names = self.RELEVANT_LAWS + [self.FOOD_SAFETY_LAW]
        filtered = filter_law_names_for_case(self.CLIENT_LIST_INPUT, law_names)
        for law in self.RELEVANT_LAWS:
            assert law in filtered

    def test_food_safety_kept_when_food_related(self):
        """食品相关案 → 食品安全法应保留"""
        input_text = "食品公司离职员工泄露食品经销客户名单并造成食品安全监管风险怎么办？"
        law_names = [self.FOOD_SAFETY_LAW]
        filtered = filter_law_names_for_case(input_text, law_names)
        assert self.FOOD_SAFETY_LAW in filtered

    def test_food_safety_filtered_no_food_keywords(self):
        """不含食品关键词 → 食品安全法应被过滤"""
        filtered = filter_law_names_for_case("员工把客户名单卖了怎么办", [self.FOOD_SAFETY_LAW])
        assert self.FOOD_SAFETY_LAW not in filtered

    def test_food_safety_kept_with_food_keyword(self):
        """含"食品"关键词 → 食品安全法应保留"""
        filtered = filter_law_names_for_case("食品公司产品质量问题怎么索赔", [self.FOOD_SAFETY_LAW])
        assert self.FOOD_SAFETY_LAW in filtered

    def test_food_safety_kept_with_catering_keyword(self):
        """含"餐饮"关键词 → 食品安全法应保留"""
        filtered = filter_law_names_for_case("餐饮店食物中毒怎么赔偿", [self.FOOD_SAFETY_LAW])
        assert self.FOOD_SAFETY_LAW in filtered


class TestIsLawRelevantFoodSafety:
    """测试 is_law_relevant_to_case 的食品安全法过滤"""

    CLIENT_LIST_INPUT = "离职员工把客户名单卖给竞争对手，公司怎么起诉要求赔偿？"

    def test_food_safety_doc_filtered_for_client_case(self):
        """客户名单泄露案 → 食品安全法 doc 应返回 False"""
        doc = MockDoc({"law_name": "中华人民共和国食品安全法"})
        assert is_law_relevant_to_case(self.CLIENT_LIST_INPUT, doc) is False

    def test_food_safety_doc_allowed_for_food_case(self):
        """食品相关案 → 食品安全法 doc 应返回 True"""
        doc = MockDoc({"law_name": "中华人民共和国食品安全法"})
        assert is_law_relevant_to_case("食品公司员工泄露客户名单造成食品安全风险", doc) is True

    def test_food_safety_doc_allowed_with_catering(self):
        """含"餐饮"关键词 → 食品安全法 doc 应返回 True"""
        doc = MockDoc({"law_name": "中华人民共和国食品安全法"})
        assert is_law_relevant_to_case("餐饮企业食品卫生问题", doc) is True

    def test_relevant_docs_still_allowed_when_filtering_food(self):
        """客户名单泄露案 → 反不正当竞争法、民法典 doc 仍返回 True"""
        for law in ["中华人民共和国反不正当竞争法", "中华人民共和国民法典", "中华人民共和国劳动合同法"]:
            doc = MockDoc({"law_name": law})
            assert is_law_relevant_to_case(self.CLIENT_LIST_INPUT, doc) is True


class TestDECOMPOSEFoodSafetyRule:
    """验证 DECOMPOSE_PROMPT 包含 食品安全法 规则"""

    def test_contains_food_safety_rule(self):
        """DECOMPOSE_PROMPT 应包含食品安全法输出限制规则"""
        assert "食品安全法" in DECOMPOSE_PROMPT
        assert "食品生产" in DECOMPOSE_PROMPT


class TestREPORTFoodSafetyRule:
    """验证 REPORT_PROMPT 包含 食品安全法 规则"""

    def test_contains_food_safety_rule(self):
        """REPORT_PROMPT 应包含食品安全法引用限制规则"""
        assert "食品安全法" in REPORT_PROMPT
        assert "食品生产" in REPORT_PROMPT
        assert "餐饮" in REPORT_PROMPT
