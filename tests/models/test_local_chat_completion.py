import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.model import CachingLM, hash_args
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


def _chat_arguments(prompt, do_sample=False):
    gen_kwargs = {"max_gen_toks": 8}
    if do_sample:
        gen_kwargs["do_sample"] = True
    return (
        JsonChatStr(json.dumps([{"role": "user", "content": prompt}])),
        gen_kwargs,
    )


def _mock_response(completion_tokens, finish_reason):
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


def _make_chat_task(request_prompts, repeats=1, do_sample=False):
    class ChatGenerationTask:
        VERSION = 0
        OUTPUT_TYPE = "generate_until"
        MULTIMODAL = False
        _metric_fn_list = {}

        def __init__(self):
            self._docs = [{"prompts": request_prompts}]
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
            doc = self._docs[0]
            self._instances = [
                Instance(
                    request_type="generate_until",
                    doc=doc,
                    arguments=_chat_arguments(prompt, do_sample=do_sample),
                    idx=idx,
                    metadata=(self.task_name, 0, repeats),
                )
                for idx, prompt in enumerate(request_prompts)
            ]

        def apply_filters(self):
            for instance in self._instances:
                instance.filtered_resps["none"] = instance.resps[0]

        def doc_iterator(self, **kwargs):
            yield from enumerate(self._docs)

        def doc_to_target(self, doc):
            return "identical answer"

        def process_results(self, doc, resps):
            assert all(type(resp) is str for resp in resps)
            return {
                "exact_match": float(
                    all(resp == self.doc_to_target(doc) for resp in resps)
                )
            }

        def aggregation(self):
            return {"exact_match": lambda values: sum(values) / len(values)}

        def higher_is_better(self):
            return {"exact_match": True}

        def dump_config(self):
            return {}

    return ChatGenerationTask()


def test_log_samples_aligns_mixed_cached_and_fresh_responses(tmp_path):
    task = _make_chat_task(["cached", "fresh"])
    model = LocalChatCompletion(
        base_url="http://test-url.com", tokenizer_backend=None, model="test-model"
    )
    cached_model = CachingLM(model, str(tmp_path / "responses.db"))
    cached_model.dbdict[hash_args("generate_until", _chat_arguments("cached"))] = (
        "identical answer"
    )

    with patch(
        "lm_eval.models.api_models.requests.post",
        return_value=_mock_response(8, "length"),
    ) as mock_post:
        results = evaluate(
            lm=cached_model,
            task_dict={"tasks": {task.task_name: task}, "groups": {}},
            limit=1,
            bootstrap_iters=0,
            log_samples=True,
        )

    assert mock_post.call_count == 1
    sample = results["samples"][task.task_name][0]
    assert sample["resps"] == [["identical answer"], ["identical answer"]]
    assert sample["generation_metadata"] == [
        [None],
        [{"completion_tokens": 8, "finish_reason": "length"}],
    ]

    cached_task = _make_chat_task(["cached", "fresh"])
    with patch("lm_eval.models.api_models.requests.post") as mock_post:
        cached_results = evaluate(
            lm=cached_model,
            task_dict={"tasks": {cached_task.task_name: cached_task}, "groups": {}},
            limit=1,
            bootstrap_iters=0,
            log_samples=True,
        )

    assert mock_post.call_count == 0
    cached_sample = cached_results["samples"][cached_task.task_name][0]
    assert cached_sample["generation_metadata"] == sample["generation_metadata"]


class _AsyncResponse:
    def __init__(self, completion_tokens, finish_reason, delay):
        self.completion_tokens = completion_tokens
        self.finish_reason = finish_reason
        self.delay = delay
        self.ok = True
        self.status = 200

    async def json(self):
        await asyncio.sleep(self.delay)
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"content": "identical answer"},
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": {"completion_tokens": self.completion_tokens},
        }

    def raise_for_status(self):
        pass


class _AsyncResponseContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        pass


class _AsyncSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.post_count = 0

    def post(self, *args, **kwargs):
        self.post_count += 1
        return _AsyncResponseContext(next(self.responses))


class _AsyncSessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        pass


def test_log_samples_aligns_concurrent_repeats():
    task = _make_chat_task(["repeated"], repeats=2, do_sample=True)
    model = LocalChatCompletion(
        base_url="http://test-url.com",
        tokenizer_backend=None,
        model="test-model",
        num_concurrent=2,
    )
    session = _AsyncSession(
        [
            _AsyncResponse(3, "stop", delay=0.01),
            _AsyncResponse(8, "length", delay=0),
        ]
    )

    with (
        patch("lm_eval.models.api_models.ClientSession", return_value=_AsyncSessionContext(session)),
        patch("lm_eval.models.api_models.TCPConnector"),
    ):
        results = evaluate(
            lm=model,
            task_dict={"tasks": {task.task_name: task}, "groups": {}},
            limit=1,
            bootstrap_iters=0,
            log_samples=True,
        )

    assert session.post_count == 2
    sample = results["samples"][task.task_name][0]
    assert sample["resps"] == [["identical answer", "identical answer"]]
    assert sample["generation_metadata"] == [
        [
            {"completion_tokens": 3, "finish_reason": "stop"},
            {"completion_tokens": 8, "finish_reason": "length"},
        ]
    ]
