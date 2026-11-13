import json
import unittest

from tensorflow_core.chat_protocol import extract_json_object, normalize_tool_call


FENCE = chr(96) * 3
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sandbox_command",
            "parameters": {"type": "object"},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_workspace_file",
            "parameters": {"type": "object"},
        },
    },
]


class ExtractJsonObjectTests(unittest.TestCase):
    def test_accepts_standalone_object(self):
        value = extract_json_object('{"tool":"run_sandbox_command","command":"pwd"}')
        self.assertEqual(value, {"tool": "run_sandbox_command", "command": "pwd"})

    def test_accepts_json_code_fence(self):
        value = extract_json_object(f' {FENCE}json\n{{"kind":"cube"}}\n{FENCE} ')
        self.assertEqual(value, {"kind": "cube"})

    def test_rejects_prose_wrapped_json(self):
        self.assertIsNone(
            extract_json_object('Here is an example: {"tool":"run_sandbox_command","command":"pwd"}')
        )

    def test_rejects_trailing_prose(self):
        self.assertIsNone(extract_json_object('{"kind":"cube"} and that is the plan'))

    def test_rejects_non_json_code_fence(self):
        self.assertIsNone(
            extract_json_object(f'{FENCE}python\n{{"tool":"run_sandbox_command"}}\n{FENCE}')
        )

    def test_rejects_arrays_and_scalars(self):
        self.assertIsNone(extract_json_object('[{"kind":"cube"}]'))
        self.assertIsNone(extract_json_object('"not an object"'))

    def test_rejects_malformed_json(self):
        self.assertIsNone(extract_json_object('{"tool":'))


class NormalizeToolCallTests(unittest.TestCase):
    def test_normalizes_allowed_command(self):
        content = json.dumps({
            "tool": "run_sandbox_command",
            "command": "pwd",
            "args": ["-P"],
        })
        message, calls = normalize_tool_call(content, TOOLS)
        self.assertEqual(message, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["type"], "function")
        self.assertEqual(calls[0]["function"]["name"], "run_sandbox_command")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"command": "pwd", "args": ["-P"]},
        )

    def test_rejects_tool_json_embedded_in_prose(self):
        content = 'I can show an example: {"tool":"run_sandbox_command","command":"pwd"}'
        message, calls = normalize_tool_call(content, TOOLS)
        self.assertEqual(message, content)
        self.assertEqual(calls, [])

    def test_rejects_unlisted_tool(self):
        content = '{"tool":"delete_everything","target":"workspace"}'
        message, calls = normalize_tool_call(content, TOOLS)
        self.assertEqual(message, content)
        self.assertEqual(calls, [])

    def test_rejects_invalid_command_arguments(self):
        content = '{"tool":"run_sandbox_command","command":"pwd","args":[1]}'
        message, calls = normalize_tool_call(content, TOOLS)
        self.assertEqual(message, content)
        self.assertEqual(calls, [])

    def test_rejects_invalid_file_arguments(self):
        content = '{"tool":"write_workspace_file","path":"notes.txt","content":null}'
        message, calls = normalize_tool_call(content, TOOLS)
        self.assertEqual(message, content)
        self.assertEqual(calls, [])

    def test_no_tools_means_no_tool_call(self):
        content = '{"tool":"run_sandbox_command","command":"pwd"}'
        message, calls = normalize_tool_call(content, None)
        self.assertEqual(message, content)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
