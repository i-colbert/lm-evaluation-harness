import json
from unittest.mock import MagicMock, patch

import pytest

from lm_eval.api.instance import Instance
from lm_eval.evaluator import evaluate
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.openai_completions import LocalChatCompletion


@pytest.mark.parametrize(
    "think_end_token, content, expected",
    [
        (
            None,
            "<think>reasoning</think> final answer",
            "<think>reasoning</think> final answer",
        ),
        ("</think>", "<think>reasoning</think>  final answer", "final answer"),
        ("</think>", "first</think> draft</think> final answer", "final answer"),
        (
            "</think>",
            "answer without a thinking marker",
            "answer without a thinking marker",
        ),
        ("</think>", None, None),
    ],
)
def test_parse_generations_strips_thinking(think_end_token, content, expected):
    model = LocalChatCompletion(
        base_url="http://test-url.com",
        model="test-model",
        think_end_token=think_end_token,
    )
    response = {
        "choices": [{"index": 0, "message": {"content": content}}],
    }

    assert model.parse_generations(response) == [expected]


def test_log_samples_keeps_chat_response_metadata_with_identical_answers(tmp_path):
    class ChatGenerationTask:
        VERSION = 0
        OUTPUT_TYPE = "generate_until"
        MULTIMODAL = False
        _metric_fn_list = {}

        def __init__(self):
            self._docs = [{"prompt": "first"}, {"prompt": "second"}]
            self._instances = []

        @property
        def task_name(self):
            return "mock_chat"

        @property
        def eval_docs(self):
            return self._docs

        @property
        def instances(self):
            return self._instances

        def build_all_requests(self, **kwargs):
            self._instances = [
                Instance(
                    request_type="generate_until",
                    doc=doc,
                    arguments=(
                        JsonChatStr(
                            json.dumps(
                                [{"role": "user", "content": doc["prompt"]}]
                            )
                        ),
                        {"max_gen_toks": 8},
                    ),
                    idx=0,
                    metadata=(self.task_name, doc_id, 1),
                )
                for doc_id, doc in enumerate(self._docs)
            ]

        def apply_filters(self):
            for instance in self._instances:
                instance.filtered_resps["none"] = instance.resps[0]

        def doc_iterator(self, **kwargs):
            yield from enumerate(self._docs)

        def doc_to_target(self, doc):
            return "identical answer"

        def process_results(self, doc, resps):
            assert isinstance(resps[0], str)
            return {"exact_match": float(resps[0] == self.doc_to_target(doc))}

        def aggregation(self):
            return {"exact_match": lambda values: sum(values) / len(values)}

        def higher_is_better(self):
            return {"exact_match": True}

        def dump_config(self):
            return {}

    def mock_response(completion_tokens, finish_reason):
        response = MagicMock()
        response.ok = True
        response.json.return_value = {
            "choices": [
                {
                    "index": 0,
                    "message": {"content": "identical answer"},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"completion_tokens": completion_tokens},
        }
        return response

    model = LocalChatCompletion(
        base_url="http://test-url.com", tokenizer_backend=None, model="test-model"
    )
    task = ChatGenerationTask()
    with patch(
        "lm_eval.models.api_models.requests.post",
        side_effect=[mock_response(3, "stop"), mock_response(8, "length")],
    ) as mock_post:
        results = evaluate(
            lm=model,
            task_dict={"tasks": {task.task_name: task}, "groups": {}},
            limit=2,
            bootstrap_iters=0,
            log_samples=True,
        )

    assert mock_post.call_count == 2
    samples = results["samples"][task.task_name]
    assert [sample["resps"][0][0] for sample in samples] == [
        "identical answer",
        "identical answer",
    ]
    assert [sample["generation_metadata"][0][0] for sample in samples] == [
        {"completion_tokens": 3, "finish_reason": "stop"},
        {"completion_tokens": 8, "finish_reason": "length"},
    ]

    tracker = EvaluationTracker(output_path=str(tmp_path))
    tracker.general_config_tracker.log_experiment_args(
        model_source="local-chat-completions",
        model_args={"model": "test-model"},
        system_instruction=None,
        chat_template=None,
        fewshot_as_multiturn=False,
    )
    tracker.date_id = "test"
    tracker.save_results_samples(task.task_name, samples)

    sample_file = next((tmp_path / "test-model").glob("*.jsonl"))
    saved_samples = [
        json.loads(line) for line in sample_file.read_text(encoding="utf-8").splitlines()
    ]
    assert [sample["generation_metadata"][0][0] for sample in saved_samples] == [
        {"completion_tokens": 3, "finish_reason": "stop"},
        {"completion_tokens": 8, "finish_reason": "length"},
    ]


def test_parse_generations_with_metadata_allows_missing_usage_and_finish_reason():
    model = LocalChatCompletion(
        base_url="http://test-url.com", tokenizer_backend=None, model="test-model"
    )

    generations, metadata = model.parse_generations_with_metadata(
        {"choices": [{"index": 0, "message": {"content": "answer"}}]}
    )

    assert generations == ["answer"]
    assert metadata == [{"completion_tokens": None, "finish_reason": None}]
