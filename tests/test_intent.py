"""意图分类器单元测试。"""
import pytest
from app.classifier import classify_intent


class TestClassifyIntent:

    # --- qa ---
    def test_short_message_returns_qa(self):
        assert classify_intent("你好") == "qa"

    def test_general_question_returns_qa(self):
        assert classify_intent("试用期最长是多久？") == "qa"

    # --- statute ---
    def test_statute_keyword(self):
        assert classify_intent("诉讼时效是多久") == "statute"

    def test_statute_short(self):
        assert classify_intent("还来得及吗") == "statute"

    def test_statute_arbitration(self):
        assert classify_intent("仲裁时效期间怎么算") == "statute"

    # --- analysis ---
    def test_analysis_keyword_long(self):
        assert classify_intent("我被公司无故辞退了，能帮我分析一下案情吗") == "analysis"

    def test_analysis_sue(self):
        assert classify_intent("我想起诉对方违约，帮我分析一下案情怎么维权比较好") == "analysis"

    def test_generic_rights_question_stays_qa(self):
        assert classify_intent("公司高管挪用了员工的工资款，员工该怎么维权？能追究刑事责任吗？") == "qa"

    def test_personal_fact_pattern_can_enter_analysis(self):
        assert classify_intent("我在公司工作三年，上个月被辞退了，没有补偿，我该怎么维权？") == "analysis"

    def test_labor_case_narrative_can_enter_analysis_without_explicit_request(self):
        assert classify_intent("我在公司工作两年，没有签劳动合同，上个月被口头辞退，月工资8000，公司没给补偿。") == "analysis"

    def test_analysis_and_document_workflow_starts_with_analysis(self):
        question = (
            "请帮我分析案情并生成后续文书。我叫张三，电话13800000000，"
            "2022年3月1日入职北京星河科技有限公司，2024年4月30日被公司书面解除，月工资8000元。"
        )

        assert classify_intent(question) == "analysis"

    def test_named_labor_case_without_first_person_enters_analysis(self):
        question = (
            "王五在杭州云舟信息技术有限公司做客服，月工资6500元，"
            "上个月被口头辞退，没有签劳动合同，有工资流水和聊天记录。"
        )

        assert classify_intent(question) == "analysis"

    def test_labor_noncompete_cross_case_enters_analysis(self):
        question = (
            "我从一家软件公司离职后去了竞争对手，公司说我违反竞业限制和保密协议，"
            "还拖欠我最后两个月工资。我想知道劳动仲裁能不能一起处理。"
        )

        assert classify_intent(question) == "analysis"

    @pytest.mark.parametrize("question", [
        "我上个月被公司口头辞退，没签劳动合同，工资每月7000元，想先分析再准备劳动仲裁申请书。",
        "朋友涉嫌盗窃被刑拘，家属想知道会不会判刑、能不能取保候审，请帮忙分析。",
        "我在装修队干活，按天结算每天300元，干了三个多月受伤后老板不让我去了，也没签合同，只有微信转账和工友证言。",
        "我是劳务派遣工，和派遣公司签了合同，被用工单位退回后派遣公司一直不给安排工作，也不给发工资。请分析能不能仲裁。",
        "我在一家奶茶店上班半年，老板没有给我签劳动合同，突然让我不用来了，工资还有两个月没结。店没有告诉我营业执照名称。",
    ])
    def test_closed_loop_scenario_intents_enter_analysis(self, question):
        assert classify_intent(question) == "analysis"

    # --- document ---
    def test_document_arbitration(self):
        assert classify_intent("帮我写一份劳动仲裁申请书") == "document"

    def test_document_complaint(self):
        assert classify_intent("写一个民事起诉状") == "document"

    def test_document_lawyer_letter(self):
        assert classify_intent("起草一份律师函") == "document"

    def test_document_contract_review(self):
        assert classify_intent("合同审查一下") == "document"

    def test_apply_arbitration_is_not_document_generation(self):
        assert classify_intent("老板拖欠工资还威胁恐吓员工，员工该怎么维权？能否同时报警和申请仲裁？") == "qa"

    # --- priority ---
    def test_document_over_statute(self):
        # 同时包含文书和时效关键词，文书优先
        assert classify_intent("写一份仲裁申请书，诉讼时效是多久") == "document"
