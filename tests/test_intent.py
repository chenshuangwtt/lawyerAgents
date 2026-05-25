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

    # --- document ---
    def test_document_arbitration(self):
        assert classify_intent("帮我写一份劳动仲裁申请书") == "document"

    def test_document_complaint(self):
        assert classify_intent("写一个民事起诉状") == "document"

    def test_document_lawyer_letter(self):
        assert classify_intent("起草一份律师函") == "document"

    def test_document_contract_review(self):
        assert classify_intent("合同审查一下") == "document"

    # --- priority ---
    def test_document_over_statute(self):
        # 同时包含文书和时效关键词，文书优先
        assert classify_intent("写一份仲裁申请书，诉讼时效是多久") == "document"
