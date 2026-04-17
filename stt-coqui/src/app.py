#!/usr/bin/env python3

import argparse
import asyncio
import itertools
import logging
import math
import platform
import shlex
import shutil
import struct
import tarfile
import tempfile
import threading
import time
import unicodedata
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

import aiohttp
from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.middleware.proxy_fix import ProxyFix
from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

from hassil_fst import EPS, SPACE, TEMPLATE_CHARS, templates_to_fst

if TYPE_CHECKING:
    from nemo.collections.asr.models import ASRModel

_LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
BLANK = "<blank>"

IS_ARM64 = platform.machine().lower() in ("arm64", "aarch64")

URL_FORMAT = "https://huggingface.co/datasets/rhasspy/rhasspy-speech/resolve/main/models/{model_id}.tar.gz?download=true"
SUPPORTED_LANGUAGES = [
    "am_ET",
    "br_FR",
    "ca_ES",
    "cnh_MM",
    "cs_CZ",
    "ctp_MX",
    "cv_RU",
    "cy_GB",
    "de_DE",
    "dv_MV",
    "el_GR",
    "en_US",
    "es_ES",
    "et_EE",
    "eu_ES",
    "fa_IR",
    "fi_FI",
    "fy_NL",
    "ga_IE",
    "hi_IN",
    "hsb_DE",
    "hu_HU",
    "id_ID",
    "it_IT",
    "ka_GE",
    "kv_RU",
    "ky_KG",
    "lb_LU",
    "lg_UG",
    "lt_LT",
    "lv_LV",
    "mn_MN",
    "mt_MT",
    "nl_NL",
    "or_IN",
    "pl_PL",
    "pt_PT",
    "rm_CH_sursilv",
    "rm_CH_vallader",
    "ro_RO",
    "ru_RU",
    "rw_RW",
    "sah_RU",
    "sl_SI",
    "sw_CD",
    "ta_IN",
    "th_TH",
    "tos_MX",
    "tr_TR",
    "tt_RU",
    "uk_UA",
    "wo_SN",
    "xty_MX",
    "yo_NG",
]

# -----------------------------------------------------------------------------


async def main() -> None:
    """Run app."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--model",
        required=True,
        help=f"Coqui model language or directory ({', '.join(SUPPORTED_LANGUAGES)})",
    )
    parser.add_argument(
        "--language", help="Provide model language if using a directory"
    )
    parser.add_argument(
        "--sentences", required=True, help="Path to sentences text file"
    )
    parser.add_argument(
        "--train-dir", required=True, help="Directory to write training files"
    )
    parser.add_argument("--cache-dir", required=True, help="Path to cache model files")
    parser.add_argument("--http-host", default="127.0.0.1", help="Host for web UI")
    parser.add_argument("--http-port", default=5000, type=int, help="Port for web UI")
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    exe_ext = "arm64" if IS_ARM64 else "x86_64"
    exe_path = BASE_DIR / "bin" / f"stt_onlyprobs.{exe_ext}"

    # Ensure directories exist
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download model if necessary
    model_dir = Path(args.model)
    if model_dir.exists():
        model_id = model_dir.name
        language = args.language or ""
        _LOGGER.debug("Will load model from %s", model_dir)
    else:
        # From HuggingFace
        language = args.model
        model_id = f"{language}-coqui"
        model_dir = await download_model(model_id, cache_dir)

    train_dir = Path(args.train_dir) / model_id
    train_dir.mkdir(parents=True, exist_ok=True)

    sentences_path = Path(args.sentences)
    sentences_path.parent.mkdir(parents=True, exist_ok=True)
    if not sentences_path.exists():
        sentences_path.write_text("what time is it\n", encoding="utf-8")

    # Run web UI
    flask_app = get_app(model_dir, train_dir, sentences_path, language)

    def run_flask():
        flask_app.run(host=args.http_host, port=args.http_port, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Train and start Wyoming server
    _LOGGER.info("Training started")
    await train(model_dir, train_dir, sentences_path, language)
    _LOGGER.info("Training finished")

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")

    try:
        await server.run(
            partial(WyomingEventHandler, model_dir, language, train_dir, exe_path)
        )
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------


class WyomingEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        model_dir: Path,
        language: str,
        train_dir: Path,
        exe_path: Path,
        *args,
        **kwargs,
    ) -> None:
        """Initialize event handler."""
        super().__init__(*args, **kwargs)

        self.client_id = str(time.monotonic_ns())
        self.model_dir = model_dir
        self.language = language
        self.train_dir = train_dir
        self.exe_path = exe_path

        self._proc: "Optional[asyncio.subprocess.Process]" = None
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
            if self._proc is None:
                _LOGGER.debug("Receiving audio")
                self._proc = await asyncio.create_subprocess_exec(
                    str(self.exe_path),
                    str(self.model_dir / "model.tflite"),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )

            chunk = AudioChunk.from_event(event)

            # Write chunk size (4 bytes), then chunk
            assert self._proc is not None
            assert self._proc.stdin is not None

            self._proc.stdin.write(struct.pack("I", len(chunk.audio)))
            self._proc.stdin.write(chunk.audio)
            await self._proc.stdin.drain()

        elif AudioStop.is_type(event.type):
            assert self._proc is not None
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None

            _LOGGER.debug("Transcribing...")
            start_time = time.monotonic()

            # Zero-length chunk signals end
            self._proc.stdin.write(struct.pack("I", 0))
            await self._proc.stdin.drain()

            line = (await self._proc.stdout.readline()).decode().strip()
            probs: List[List[float]] = []
            while line:
                probs.append([float(p) for p in line.split()])
                line = (await self._proc.stdout.readline()).decode().strip()

            self._proc.terminate()
            await self._proc.wait()

            text = await transcribe(self.train_dir, probs)

            end_time = time.monotonic()
            _LOGGER.debug("Transcribed in %s second(s)", end_time - start_time)

            _LOGGER.debug("Transcript (%s): %s", self.language, text)

            await self.write_event(
                Transcript(text=text, language=self.language).event()
            )

            self._proc = None

        return True

    async def _write_info(self) -> None:
        if self._info_event is not None:
            await self.write_event(self._info_event)
            return

        info = Info(
            asr=[
                AsrProgram(
                    name="stt-coqui",
                    attribution=Attribution(
                        name="The Home Assistant Authors",
                        url="http://github.com/OHF-voice",
                    ),
                    description="Coqui STT",
                    installed=True,
                    version="1.0.0",
                    models=[
                        AsrModel(
                            name="stt-coqui",
                            attribution=Attribution(
                                name="STT Models",
                                url="https://github.com/coqui-ai/STT-models",
                            ),
                            installed=True,
                            description="Coqui STT",
                            version=None,
                            languages=SUPPORTED_LANGUAGES,
                        )
                    ],
                )
            ]
        )

        self._info_event = info.event()
        await self.write_event(self._info_event)


# -----------------------------------------------------------------------------


async def download_model(model_id: str, cache_dir: Path) -> Path:
    model_dir = cache_dir / model_id
    if model_dir.exists():
        # Already downloaded
        return model_dir

    url = URL_FORMAT.format(model_id=model_id)
    _LOGGER.debug("Downloading model %s at %s to %s", model_id, url, model_dir)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()

                with tempfile.NamedTemporaryFile(
                    mode="wb+", suffix=".tar.gz"
                ) as temp_file:
                    async for chunk in response.content.iter_chunked(2048):
                        temp_file.write(chunk)

                    temp_file.seek(0)
                    with tarfile.open(temp_file.name, mode="r:gz") as tar:
                        tar.extractall(path=cache_dir)

        _LOGGER.debug("Downloaded model %s", model_id)
    except Exception:
        _LOGGER.exception("Unexpected error while downloading model %s", model_id)

        # Delete directory is it can be re-downloaded
        if model_dir.exists():
            shutil.rmtree(model_dir)

        raise

    return model_dir


async def train(
    model_dir: Path, train_dir: Path, sentences_path: Path, language: str
) -> None:
    idx2char: Dict[int, str] = {}
    char2idx: Dict[str, int] = {}

    alphabet_path = model_dir / "alphabet.txt"

    # Load alphabet
    a_idx = 1  # <eps> = 0
    with open(alphabet_path, "r", encoding="utf-8") as a_file:
        for line in a_file:
            line = line.strip()
            if line.startswith("#"):
                continue

            if not line:
                line = " "
            elif line == "\\#":
                line = "#"

            c = line[0]
            if c == " ":
                c = SPACE

            idx2char[a_idx] = c
            char2idx[c] = a_idx
            a_idx += 1

    blank_id = a_idx
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
    with open(sentences_path, "r", encoding="utf-8") as sentences_file, open(
        char2sen_txt, "w", encoding="utf-8"
    ) as char2sen_file:
        templates = []
        warned_chars: Set[str] = set()
        for line in sentences_file:
            # Normalize
            original_line = line
            line = line.strip()
            line = unicodedata.normalize("NFC", line)
            line = line.lower()
            chars = []
            in_list_name = False
            for char in line:
                if (char == " ") and (not in_list_name):
                    char = SPACE

                if (char == "{") and (not in_list_name):
                    in_list_name = True
                elif (char == "}") and in_list_name:
                    in_list_name = False
                elif in_list_name:
                    chars.append(char)
                elif (char in char2idx) or (char in TEMPLATE_CHARS):
                    chars.append(char)
                elif char not in warned_chars:
                    _LOGGER.warning("Skipping '%s' in '%s'", char, line)
                    warned_chars.add(char)

            line = line.strip()
            if not line:
                _LOGGER.warning("Skipped line with no valid chars: %s", original_line)
                continue

            templates.append(line)
            _LOGGER.debug(line)

        fst = templates_to_fst(templates, language)
        fst.write(char2sen_file)

    char2sen_fst = train_dir / "char2sen.fst"
    await _try_minimize(
        [
            "fstcompile",
            shlex.quote(f"--isymbols={tokens_without_blank}"),
            shlex.quote(f"--osymbols={tokens_without_blank}"),
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


async def transcribe(train_dir: Path, probs: List[List[float]]) -> str:
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

            # Add space to the end and make it the most probable
            space_prob = 0.99
            nonspace_prob = ((1 - space_prob) / (len(probs[0]) - 1)) + 1e-9
            space_probs = [
                space_prob if c == SPACE else nonspace_prob for c in char2idx
            ]

            for current_probs in itertools.chain(probs, [space_probs]):
                for i, prob in enumerate(current_probs, start=1):
                    log_prob = -math.log(prob + 1e-9)
                    if (i == blank_id) or (i >= len(idx2char)):
                        c = BLANK
                    else:
                        c = idx2char[i]

                    print(current, current + 1, c, log_prob, file=logits_file)

                current += 1

            print(current, file=logits_file)

        # tokens -> chars -> words -> sentences
        tokens_txt = train_dir / "tokens_with_blank.txt"
        token2sen_fst = train_dir / "token2sen.fst"
        stdout = await async_run_pipeline(
            [
                "fstcompile",
                shlex.quote(f"--isymbols={tokens_txt}"),
                shlex.quote(f"--osymbols={tokens_txt}"),
                "--acceptor",
                shlex.quote(str(logits_txt)),
            ],
            ["fstdeterminize"],
            ["fstminimize"],
            ["fstpush", "--push_weights"],
            ["fstarcsort", "--sort_type=olabel"],
            # ["fstprune", f"--weight={prune_threshold}"],  # prune logits
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

    text = stdout.decode(encoding="utf-8")
    _LOGGER.debug("Raw text: %s", text)

    chars: List[str] = []
    sentence_prob = 0.0
    for line in text.splitlines():
        line_parts = line.strip().split()
        if len(line_parts) < 4:
            continue

        char = line_parts[3]
        if char == SPACE:
            char = " "

        chars.append(char)

        if len(line_parts) > 4:
            word_prob = float(line_parts[4])
            sentence_prob += word_prob

    text = "".join(chars)

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

    stdout, _stderr = await result.communicate()

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


def get_app(
    model_dir: Path, train_dir: Path, sentences_path: Path, language: str
) -> Flask:
    flask_app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
    flask_app.secret_key = "90a238ad-7e69-4438-85dc-eee0a68c7435"

    flask_app.wsgi_app = ProxyFix(flask_app.wsgi_app, x_proto=1, x_host=1)  # type: ignore[method-assign]
    flask_app.wsgi_app = IngressPrefixMiddleware(flask_app.wsgi_app)  # type: ignore[method-assign]

    @flask_app.context_processor
    def inject_url_for():
        return dict(url_for=url_for)  # pylint: disable=use-dict-literal

    @flask_app.route("/", methods=["GET"])
    def index():
        content = sentences_path.read_text(encoding="utf-8")
        return render_template("index.html", content=content)

    @flask_app.route("/save", methods=["POST"])
    async def save():
        content = request.form.get("content", "")
        sentences_path.write_text(content, encoding="utf-8")
        _LOGGER.info("Retraining...")
        await train(model_dir, train_dir, sentences_path, language)
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
