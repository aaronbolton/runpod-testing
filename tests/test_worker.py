"""Checks that need no GPU and no model: routing and argv validation."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import proxy  # noqa: E402
import server  # noqa: E402


class Routing(unittest.TestCase):
    def test_messages_go_to_chat(self):
        route, body = proxy._route({"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(route, proxy.CHAT)
        self.assertIn("messages", body)

    def test_prompt_goes_to_completions(self):
        route, _ = proxy._route({"prompt": "hi"})
        self.assertEqual(route, proxy.COMPLETIONS)

    def test_explicit_route_wins(self):
        route, body = proxy._route(
            {"openai_route": "/v1/embeddings", "openai_input": {"input": "x"}}
        )
        self.assertEqual(route, "/v1/embeddings")
        self.assertEqual(body, {"input": "x"})

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            proxy._route({})


class Argv(unittest.TestCase):
    def setUp(self):
        for key in (
            "MODEL_PATH", "MMPROJ_PATH", "MODEL_REPO",
            "MODEL_FILE", "MMPROJ_FILE", "LLAMA_ARGS", "HF_MODEL",
        ):
            os.environ.pop(key, None)

    def test_model_path_builds_argv(self):
        os.environ["MODEL_PATH"] = "/models/x.gguf"
        os.environ["LLAMA_ARGS"] = "--ctx-size 4096 -ngl all"
        argv = server.build_argv()
        self.assertEqual(argv[:3], [server.BIN, "-m", "/models/x.gguf"])
        self.assertIn("--ctx-size", argv)
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")

    def test_mmproj_is_passed(self):
        os.environ["MODEL_PATH"] = "/models/x.gguf"
        os.environ["MMPROJ_PATH"] = "/models/p.gguf"
        self.assertIn("--mmproj", server.build_argv())

    def test_reserved_flags_are_rejected(self):
        os.environ["MODEL_PATH"] = "/models/x.gguf"
        for bad in ("--port 1", "-m /other.gguf", "--mmproj /p.gguf", "-hf a/b"):
            os.environ["LLAMA_ARGS"] = bad
            with self.assertRaises(ValueError, msg=bad):
                server.build_argv()

    def test_missing_config_is_rejected(self):
        with self.assertRaises(ValueError):
            server.build_argv()
        os.environ["MODEL_REPO"] = "org/name"
        with self.assertRaises(ValueError):
            server.build_argv()

    def test_hf_model_downloads(self):
        os.environ["HF_MODEL"] = "unsloth/gemma-3-270m-it-GGUF:Q6_K"
        argv = server.build_argv()
        self.assertEqual(argv[1:3], ["-hf", "unsloth/gemma-3-270m-it-GGUF:Q6_K"])

    def test_two_sources_are_rejected(self):
        os.environ["MODEL_PATH"] = "/models/x.gguf"
        os.environ["HF_MODEL"] = "org/name:Q4_K_M"
        with self.assertRaises(ValueError):
            server.build_argv()

    def test_snapshot_prefers_refs_main(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            root = cache / "models--org--name"
            (root / "refs").mkdir(parents=True)
            (root / "refs" / "main").write_text("abc123\n")
            (root / "snapshots" / "abc123").mkdir(parents=True)
            (root / "snapshots" / "zzz999").mkdir(parents=True)
            self.assertEqual(
                server._snapshot("org/name", cache),
                root / "snapshots" / "abc123",
            )

    def test_snapshot_missing_is_explained(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                server._snapshot("org/name", Path(tmp))
            self.assertIn("Model field", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
