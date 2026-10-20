from scripts.evaluate import CASES, score


def test_transport_failure_cannot_pass_uncertainty_case():
    run = {
        "status": "failed",
        "answer": "模型服务无法访问，没有修改工单。",
        "evidence": [],
        "trace": [],
        "proposal": None,
    }
    assert score(CASES[3], run) is False


def test_exhausted_run_cannot_pass_uncertainty_case():
    run = {
        "status": "exhausted",
        "answer": "调用上限，没有修改工单。",
        "evidence": [],
        "trace": [],
        "proposal": None,
    }
    assert score(CASES[3], run) is False


def test_completed_policy_with_grouped_known_sources_passes():
    run = {
        "status": "completed",
        "answer": "取消和退款依据 [KB:source-a, KB:source-b]",
        "evidence": [{"id": "source-a"}, {"id": "source-b"}],
        "trace": [],
        "proposal": None,
    }
    assert score(CASES[0], run) is True
