"""
Test 28: LLM tool-loop safety fallback without a real LLM server.
"""

from __future__ import annotations

from _working_test_utils import patched_attr, require, run_tests

import vision.llm_interface as llm_module


class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, call_id, function_name, arguments="{}"):
        self.id = call_id
        self.function = FakeFunction(function_name, arguments)


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_none=True):
        """
        Match the subset of the OpenAI SDK message API used by llm_interface.py.

        The real SDK returns Pydantic models. The runtime code converts those
        models to dicts before appending assistant tool-call requests to history,
        so this fake keeps the test close to the production object shape without
        importing openai.
        """
        del exclude_none
        data = {"role": "assistant"}
        if self.content is not None:
            data["content"] = self.content
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in self.tool_calls
            ]
        return data


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeCompletion:
    def __init__(self, message):
        self.choices = [FakeChoice(message)]


class FakeCompletions:
    def __init__(self, messages):
        self._messages = list(messages)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._messages:
            raise AssertionError("FakeCompletions received more calls than expected")
        return FakeCompletion(self._messages.pop(0))


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, messages):
        self.completions = FakeCompletions(messages)
        self.chat = FakeChat(self.completions)


def make_fake_llm(messages):
    fake_llm = llm_module.LLMinterface.__new__(llm_module.LLMinterface)
    fake_llm.openai_client = FakeClient(messages)
    fake_llm.tools = [{"type": "function", "function": {"name": "fake_tool"}}]
    fake_llm.model = "fake-model"
    fake_llm.messages = [{"role": "system", "content": "test system prompt"}]
    fake_llm.text = "find the red cup"
    fake_llm.reply = None
    fake_llm.completion = None
    return fake_llm


def test_tool_round_limit_forces_plain_text_final_answer():
    stubborn_tool_request = FakeMessage(
        tool_calls=[FakeToolCall("call-1", "get_birds_eye_view")]
    )
    final_answer = FakeMessage(content="I found enough evidence to stop and summarize.")
    fake_llm = make_fake_llm([stubborn_tool_request, final_answer])

    # Setting the limit to zero makes the first tool request exceed the limit.
    # That isolates the fallback behavior: no real camera/tool dispatch should
    # happen, and the second completion request should be a no-tools summary.
    with patched_attr(llm_module.cfg, "LLM_MAX_TOOL_ROUNDS", 0):
        llm_module.LLMinterface.send_message_with_tools(fake_llm, None, None)

    calls = fake_llm.openai_client.completions.calls
    require(len(calls) == 2, "expected initial tool request plus forced final request")
    require("tools" in calls[0], "first request should offer tools")
    require(calls[0].get("tool_choice") == "auto", "first request should allow auto tool choice")
    require("tools" not in calls[1], "forced final request must omit tools")
    require("tool_choice" not in calls[1], "forced final request should not include tool_choice")
    require(fake_llm.reply == final_answer.content, "forced final answer was not used")
    require(
        any(
            message.get("content") == llm_module.FINAL_ANSWER_PROMPT
            for message in fake_llm.messages
            if message.get("role") == "user"
        ),
        "final no-tools prompt should be recorded in history",
    )
    require(
        any(
            message.get("content") == "ERROR: maximum tool-call rounds reached."
            for message in fake_llm.messages
            if message.get("role") == "tool"
        ),
        "tool-limit error should be recorded for the pending tool call",
    )


if __name__ == "__main__":
    raise SystemExit(
        run_tests(
            [
                (
                    "tool round limit forces plain text final answer",
                    test_tool_round_limit_forces_plain_text_final_answer,
                ),
            ]
        )
    )
