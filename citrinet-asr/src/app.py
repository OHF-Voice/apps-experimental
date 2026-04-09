#!/usr/bin/env python3

import argparse
import asyncio
import logging
import os
import shlex
import tempfile
import threading
import time
import wave
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union, cast

from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.middleware.proxy_fix import ProxyFix
from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

if TYPE_CHECKING:
    from nemo.collections.asr.models import ASRModel

_LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
EPS = "<eps>"
BLANK = "<blank>"

# -----------------------------------------------------------------------------


async def main() -> None:
    """Run app."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument("--model", required=True, help="NeMo model name")
    parser.add_argument(
        "--sentences", required=True, help="Path to sentences text file"
    )
    parser.add_argument(
        "--train-dir", required=True, help="Directory to write training files"
    )
    parser.add_argument("--cache-dir", help="Path to HuggingFace cache")
    parser.add_argument("--http-host", default="127.0.0.1", help="Host for web UI")
    parser.add_argument("--http-port", default=5000, type=int, help="Port for web UI")
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    logging.getLogger("numba").setLevel(logging.ERROR)

    # Ensure directories exist
    train_dir = Path(args.train_dir) / args.model
    train_dir.mkdir(parents=True, exist_ok=True)

    sentences_path = Path(args.sentences)
    sentences_path.parent.mkdir(parents=True, exist_ok=True)
    if not sentences_path.exists():
        sentences_path.write_text("what time is it\n", encoding="utf-8")

    if args.cache_dir:
        cache_dir = str(Path(args.cache_dir).resolve())
        os.environ["HF_HUB_CACHE"] = cache_dir
        os.environ["NEMO_CACHE_DIR"] = cache_dir
        _LOGGER.debug("Set cache directory: %s", cache_dir)

    # Have to import after setting cache dir
    from nemo.collections.asr.models import ASRModel

    # stt_<lang>_<name>
    language = args.model.split("_")[1]

    _LOGGER.info("Loading %s (language=%s)", args.model, language)
    model = cast(ASRModel, ASRModel.from_pretrained(model_name=args.model))
    model.eval()
    _LOGGER.debug("Loaded %s", args.model)

    # Run web UI
    flask_app = get_app(model, train_dir, sentences_path)

    def run_flask():
        flask_app.run(host=args.http_host, port=args.http_port, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Train and start Wyoming server
    _LOGGER.info("Training started")
    await train(model, train_dir, sentences_path)
    _LOGGER.info("Training finished")

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")

    try:
        await server.run(
            partial(WyomingEventHandler, model, language, args.model, train_dir)
        )
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------


class WyomingEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        model: "ASRModel",
        language: str,
        model_name: str,
        train_dir: Path,
        *args,
        **kwargs,
    ) -> None:
        """Initialize event handler."""
        super().__init__(*args, **kwargs)

        self.client_id = str(time.monotonic_ns())
        self.model = model
        self.language = language
        self.model_name = model_name
        self.train_dir = train_dir

        self._wav_file: Optional[wave.Wave_write] = None
        self._wav_path = train_dir / f"{self.client_id}.wav"

        self._info_event: Optional[Event] = None

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming event."""
        try:
            return await self._handle_event(event)
        except Exception:
            _LOGGER.exception("Error handling event")

        return True

    async def _handle_event(self, event: Event) -> bool:
        """Handle Wyoming event."""
        if Describe.is_type(event.type):
            await self._write_info()
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            if self._wav_file is None:
                _LOGGER.debug("Receiving audio")
                self._wav_file = wave.open(str(self._wav_path), "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)

            self._wav_file.writeframes(chunk.audio)
        elif AudioStop.is_type(event.type):
            assert self._wav_file is not None
            self._wav_file.close()

            _LOGGER.debug("Transcribing %s", self._wav_path)
            start_time = time.monotonic()
            text = await transcribe(self.model, self.train_dir, self._wav_path)
            end_time = time.monotonic()
            _LOGGER.debug("Transcribed in %s second(s)", end_time - start_time)

            _LOGGER.debug("Transcript (%s): %s", self.language, text)

            await self.write_event(
                Transcript(text=text, language=self.language).event()
            )

            # Reset
            self._wav_path.unlink(missing_ok=True)
            self._wav_file = None

        return True

    async def _write_info(self) -> None:
        if self._info_event is not None:
            await self.write_event(self._info_event)
            return

        info = Info(
            asr=[
                AsrProgram(
                    name="citrinet-asr",
                    attribution=Attribution(
                        name="The Home Assistant Authors",
                        url="http://github.com/OHF-voice",
                    ),
                    description="Citrinet ASR",
                    installed=True,
                    version="1.0.0",
                    models=[
                        AsrModel(
                            name=self.model_name,
                            attribution=Attribution(
                                name="NVIDIA",
                                url="https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html",
                            ),
                            installed=True,
                            description=self.model_name,
                            version=None,
                            languages=[self.language],
                        )
                    ],
                )
            ]
        )

        self._info_event = info.event()
        await self.write_event(self._info_event)


# -----------------------------------------------------------------------------


async def train(model: "ASRModel", train_dir: Path, sentences_path: Path) -> None:
    tokenizer = model.tokenizer

    idx2char: Dict[int, str] = {}
    char2idx: Dict[str, int] = {}

    # Load alphabet
    a_idx = 1  # <eps> = 0
    for t_id in range(len(tokenizer.vocab)):
        t_str = f"t_{t_id}"
        idx2char[a_idx] = t_str
        char2idx[t_str] = a_idx
        a_idx += 1

    blank_id = len(tokenizer.vocab) + 1  # offset by 1 for <eps>
    idx2char[blank_id] = BLANK
    char2idx[BLANK] = blank_id

    # CTC tokens
    tokens_with_blank = train_dir / "tokens_with_blank.txt"
    tokens_without_blank = train_dir / "tokens_without_blank.txt"
    with open(tokens_with_blank, "w", encoding="utf-8") as tokens_with_blank_file, open(
        tokens_without_blank, "w", encoding="utf-8"
    ) as tokens_without_blank_file:
        # NOTE: <eps> *MUST* be id 0
        for tokens_file in (tokens_with_blank_file, tokens_without_blank_file):
            print(EPS, 0, file=tokens_file)
            for i, c in idx2char.items():
                if c == BLANK:
                    continue

                print(c, i, file=tokens_file)

        print(BLANK, blank_id, file=tokens_with_blank_file)

    # token -> char
    token2char_fst = train_dir / "token2char.fst"
    if not (await _verify_fst(token2char_fst)):
        token2char_txt = train_dir / "token2char.fst.txt"
        with open(token2char_txt, "w", encoding="utf-8") as token2char_file:
            start = 0

            char2state = {c: i for i, c in enumerate(char2idx, start=1)}

            # blank loop at start
            print(start, start, BLANK, EPS, file=token2char_file)

            # make start final
            print(start, file=token2char_file)

            for c, c_state in char2state.items():
                if c == BLANK:
                    continue

                # first occurrence emits the symbol
                print(start, c_state, c, c, file=token2char_file)

                # repeated same symbol without blank emits nothing
                print(c_state, c_state, c, EPS, file=token2char_file)

                # blank resets to start
                print(c_state, start, BLANK, EPS, file=token2char_file)

                # different symbol emits immediately
                for c_other, c_other_state in char2state.items():
                    if c_other in (c, BLANK):
                        continue
                    print(
                        c_state, c_other_state, c_other, c_other, file=token2char_file
                    )

                # make token state final
                print(c_state, file=token2char_file)

        await async_run_pipeline(
            [
                "fstcompile",
                shlex.quote(f"--isymbols={tokens_with_blank}"),
                shlex.quote(f"--osymbols={tokens_without_blank}"),
                shlex.quote(str(token2char_txt)),
            ],
            ["fstdeterminize"],
            ["fstminimize"],
            ["fstpush", "--push_weights"],
            ["fstarcsort", "--sort_type=ilabel", "-", shlex.quote(str(token2char_fst))],
        )

    # char -> sentence
    char2sen_txt = train_dir / "char2sen.fst.txt"
    all_words: Set[str] = set()
    with open(sentences_path, "r", encoding="utf-8") as sentences_file, open(
        char2sen_txt, "w", encoding="utf-8"
    ) as char2sen_file:
        start_state = 0
        current_state = 1

        for line in sentences_file:
            line = line.strip()
            if not line:
                continue

            words = [f"t_{t_id}" for t_id in tokenizer.text_to_ids(line)]
            if not words:
                continue

            all_words.update(words)

            print(start_state, current_state, EPS, file=char2sen_file)
            for word in words:
                print(current_state, current_state + 1, word, file=char2sen_file)
                current_state += 1

            print(current_state, file=char2sen_file)

    char2sen_fst = train_dir / "char2sen.fst"
    await _try_minimize(
        [
            "fstcompile",
            "--acceptor",
            shlex.quote(f"--isymbols={tokens_without_blank}"),
            shlex.quote(str(char2sen_txt)),
        ],
        char2sen_fst,
    )

    # token -> char -> sentence
    token2sen_fst = train_dir / "token2sen.fst"
    await async_run_pipeline(
        [
            "fstcompose",
            shlex.quote(str(token2char_fst)),
            shlex.quote(str(char2sen_fst)),
        ],
        ["fstrmepsilon"],
        ["fstpush", "--push_weights"],
        ["fstarcsort", "--sort_type=ilabel", "-", shlex.quote(str(token2sen_fst))],
    )


async def transcribe(model: "ASRModel", train_dir: Path, wav_path: Path) -> str:
    # Importing here because HF_HUB_CACHE is set in main()
    import torch  # pylint:disable=import-outside-toplevel

    _LOGGER.debug("Loading audio: %s", wav_path)
    with wave.open(str(wav_path), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnchannels() == 1

        audio = (
            torch.frombuffer(
                wav_file.readframes(wav_file.getnframes()), dtype=torch.int16
            ).to(torch.float32)
            / 32768.0
        )

    signal = audio.unsqueeze(0)  # [1, num_samples]
    signal_len = torch.tensor([signal.shape[1]], dtype=torch.long)

    tokenizer = model.tokenizer

    _LOGGER.debug("Decoding into logits")
    with torch.no_grad():
        log_probs, encoded_lengths, *_ = model(
            input_signal=signal,
            input_signal_length=signal_len,
        )

    T = int(encoded_lengths[0].item())
    log_probs = log_probs[0, :T, :].detach().cpu()  # [T, V]

    # Decode
    _LOGGER.debug("Converting logits to tokens")
    tokens_txt = train_dir / "tokens_with_blank.txt"
    char2idx: Dict[str, int] = {}
    with open(tokens_txt, "r", encoding="utf-8") as words_file:
        for line in words_file:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue

            label = parts[0]
            if label == EPS:
                continue

            char2idx[label] = int(parts[1])

    blank_id = char2idx[BLANK]
    idx2char = {i: c for c, i in char2idx.items()}

    allowed_token_ids: Set[int] = set()
    with open(tokens_txt, "r", encoding="utf-8") as words_file:
        for line in words_file:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue

            label = parts[0]
            if label.startswith("t_"):
                t_id = int(label.split("_", maxsplit=1)[1])
                allowed_token_ids.add(t_id)

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        logits_txt = temp_dir / "logits.fst.txt"
        with open(logits_txt, "w", encoding="utf-8") as logits_file:
            current = 0
            for current_probs in log_probs:
                for i, log_prob in enumerate(current_probs, start=1):
                    token_id = i - 1
                    if (i == blank_id) or (i >= len(idx2char)):
                        c = BLANK
                    elif token_id not in allowed_token_ids:
                        continue
                    else:
                        c = idx2char[i]

                    cost = -float(log_prob.item())
                    print(current, current + 1, c, cost, file=logits_file)

                current += 1

            print(current, file=logits_file)

        # tokens -> chars -> words -> sentences
        tokens_txt = train_dir / "tokens_with_blank.txt"
        token2sen_fst = train_dir / "token2sen.fst"
        stdout = await async_run_pipeline(
            [
                "fstcompile",
                shlex.quote(f"--isymbols={tokens_txt}"),
                "--acceptor",
                shlex.quote(str(logits_txt)),
            ],
            ["fstdeterminize"],
            ["fstminimize"],
            ["fstpush", "--push_weights"],
            ["fstarcsort", "--sort_type=ilabel"],
            ["fstcompose", "-", shlex.quote(str(token2sen_fst))],
            ["fstshortestpath"],
            ["fstproject", "--project_type=output"],
            ["fstrmepsilon"],
            ["fsttopsort"],
            [
                "fstprint",
                shlex.quote(f"--isymbols={tokens_txt}"),
                shlex.quote(f"--osymbols={tokens_txt}"),
            ],
        )

    fst_output_text = stdout.decode(encoding="utf-8")
    _LOGGER.debug("Raw output: %s", fst_output_text)

    token_ids: List[int] = []
    sentence_prob = 0.0
    for line in fst_output_text.splitlines():
        line_parts = line.strip().split()
        if len(line_parts) < 4:
            continue

        word = line_parts[3]
        if word.startswith("t_"):
            token_id = int(word.split("_", maxsplit=1)[1])
            token_ids.append(token_id)

        if len(line_parts) > 4:
            word_prob = float(line_parts[4])
            sentence_prob += word_prob

    text = tokenizer.ids_to_text(token_ids)
    return text


async def _try_minimize(
    compile_command: List[str],
    fst_path: Union[str, Path],
    arc_sort_type: str = "ilabel",
) -> None:
    try:
        # With minimize
        await async_run_pipeline(
            compile_command,
            ["fstdeterminize"],
            ["fstminimize"],
            ["fstpush", "--push_weights"],
            [
                "fstarcsort",
                f"--sort_type={arc_sort_type}",
                "-",
                shlex.quote(str(fst_path)),
            ],
        )
    except Exception:
        # Without minimize
        await async_run_pipeline(
            compile_command,
            ["fstpush", "--push_weights"],
            [
                "fstarcsort",
                f"--sort_type={arc_sort_type}",
                "-",
                shlex.quote(str(fst_path)),
            ],
        )


async def _verify_fst(path: Path) -> bool:
    result = await asyncio.create_subprocess_exec(
        "fstinfo",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await result.communicate()

    if result.returncode != 0:
        return False

    output = stdout.decode()

    if "# of arcs" not in output:
        return False

    return True


async def async_run_pipeline(  # pylint: disable=redefined-builtin
    *commands: List[str], input: Optional[bytes] = None, **kwargs
) -> bytes:
    if "stderr" not in kwargs:
        kwargs["stderr"] = asyncio.subprocess.PIPE

    if input is not None:
        kwargs["stdin"] = asyncio.subprocess.PIPE

    command_str = " | ".join((shlex.join(c) for c in commands))
    _LOGGER.debug(command_str)

    proc = await asyncio.create_subprocess_shell(
        command_str,
        stdout=asyncio.subprocess.PIPE,
        **kwargs,
    )
    stdout, stderr = await proc.communicate(input=input)
    if proc.returncode != 0:
        error_text = f"Unexpected error running command {command_str}"
        if stderr:
            error_text += f": {stderr.decode()}"
        elif stdout:
            error_text += f": {stdout.decode()}"

        raise RuntimeError(error_text)

    return stdout


# -----------------------------------------------------------------------------


def get_app(model: "ASRModel", train_dir: Path, sentences_path: Path) -> Flask:
    flask_app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
    flask_app.secret_key = "90a238ad-7e69-4438-85dc-eee0a68c7435"

    flask_app.wsgi_app = ProxyFix(flask_app.wsgi_app, x_proto=1, x_host=1)  # type: ignore[method-assign]
    flask_app.wsgi_app = IngressPrefixMiddleware(flask_app.wsgi_app)  # type: ignore[method-assign]

    @flask_app.context_processor
    def inject_url_for():
        return dict(url_for=url_for)

    @flask_app.route("/", methods=["GET"])
    def index():
        content = sentences_path.read_text(encoding="utf-8")
        return render_template("index.html", content=content)

    @flask_app.route("/save", methods=["POST"])
    async def save():
        content = request.form.get("content", "")
        sentences_path.write_text(content, encoding="utf-8")
        _LOGGER.info("Retraining...")
        await train(model, train_dir, sentences_path)
        _LOGGER.debug("Training finished")

        return jsonify({"ok": True, "message": "Saved successfully."})

    return flask_app


class IngressPrefixMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        ingress_path = environ.get("HTTP_X_INGRESS_PATH", "")
        if ingress_path:
            environ["SCRIPT_NAME"] = ingress_path
            path_info = environ.get("PATH_INFO", "")
            if path_info.startswith(ingress_path):
                environ["PATH_INFO"] = path_info[len(ingress_path) :] or "/"
        return self.app(environ, start_response)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
