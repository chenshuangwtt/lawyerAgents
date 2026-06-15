from unittest.mock import patch

from app.graph import classify


def test_graph_classify_reuses_precomputed_result():
    precomputed = {
        "domains": [
            {"domain": "劳动", "law_names": ["中华人民共和国劳动合同法"]},
            {"domain": "未成年人", "law_names": ["中华人民共和国未成年人保护法"]},
            {"domain": "治安", "law_names": ["中华人民共和国治安管理处罚法"]},
        ],
        "primary_domain": "劳动",
        "is_multi_domain": True,
    }

    with patch("app.graph.classify_question_multi") as mock_classify:
        result = classify({
            "question": "16 岁少年冒用他人身份证入职后遭遇工伤，公司发现后拒绝赔偿并报警，家长该怎么办？",
            "session_id": "test",
            "_classify_result": precomputed,
        })

    mock_classify.assert_not_called()
    assert result["domain"] == "劳动"
    assert result["domains"] == precomputed["domains"]
    assert result["is_multi_domain"] is True


def test_graph_state_schema_keeps_precomputed_classification():
    from app.graph import AgentState

    assert "_classify_result" in AgentState.__annotations__
