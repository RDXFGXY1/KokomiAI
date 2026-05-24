"""
KOKOMI — Rich terminal UI
NullStudio Inc. | Tactical HUD aesthetic
"""

import time
import threading
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich.columns import Columns
from rich.rule import Rule
from rich.prompt import Prompt
from rich import box

from memory.memory_manager import MemoryManager
from personality.kokomi import KokomiPersonality
from personality.response_shaper import ResponseShaper
from planner.task_manager import TaskManager
from planner.project_tracker import ProjectTracker
from tools.file_scanner import FileScanner
from brain.inference import KokomiInference
from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Palette ───────────────────────────────────────────────────────────────────
RED      = "#c0392b"
RED_DIM  = "#7b241c"
RED_GLOW = "#e74c3c"
DARK     = "#0a0a0a"
PANEL_BG = "#111111"
DIM      = "#3d3d3d"
MID      = "#666666"
LIGHT    = "#cccccc"
WHITE    = "#f0f0f0"
GREEN    = "#1abc9c"
AMBER    = "#f39c12"
BLUE     = "#2980b9"

console = Console(highlight=False)

HELP_TEXT = [
    ("/help",              "show this panel"),
    ("/status",            "model load status"),
    ("/projects",          "list active projects"),
    ("/project <name>",    "create a project"),
    ("/tasks <project>",   "list project tasks"),
    ("/task <p> <title>",  "add a task"),
    ("/done <id>",         "complete a task"),
    ("/remember <text>",   "store a memory"),
    ("/recall <query>",    "search memories"),
    ("/scan",              "directory tree"),
    ("/todos",             "find TODO comments"),
    ("/clear",             "clear the screen"),
    ("/quit",              "exit"),
]

BOOT_FRAMES = [
    "INITIALIZING NEURAL CORE",
    "LOADING WEIGHT MATRICES",
    "CALIBRATING ATTENTION HEADS",
    "BUILDING MEMORY INDEX",
    "ESTABLISHING PERSONALITY LAYER",
    "KOKOMI ONLINE",
]


# ── Render helpers ────────────────────────────────────────────────────────────

def _header() -> Panel:
    t = Text(justify="center")
    t.append("▄ ▄ ▄  ", style=f"dim {RED_DIM}")
    t.append("K O K O M I", style=f"bold {RED_GLOW}")
    t.append("  ▄ ▄ ▄\n", style=f"dim {RED_DIM}")
    t.append("LOCAL AI ASSISTANT  //  NULLSTUDIO INC.", style=f"dim {MID}")
    return Panel(t, border_style=RED_DIM, padding=(0, 2))


def _clock_badge() -> Text:
    now = datetime.now()
    t = Text()
    t.append("◈ ", style=f"dim {RED}")
    t.append(now.strftime("%H:%M:%S"), style=f"bold {LIGHT}")
    t.append(f"  {now.strftime('%Y-%m-%d')}", style=f"dim {MID}")
    return t


def _status_bar(model_loaded: bool, session_msgs: int) -> Panel:
    t = Text()
    # model status
    if model_loaded:
        t.append("● MODEL ", style=f"bold {GREEN}")
        t.append("ONLINE", style=f"bold {GREEN}")
    else:
        t.append("● MODEL ", style=f"bold {RED}")
        t.append("OFFLINE", style=f"bold {RED}")
    t.append("   │   ", style=f"dim {DIM}")
    # session
    t.append("MSGS ", style=f"dim {MID}")
    t.append(str(session_msgs), style=f"bold {AMBER}")
    t.append("   │   ", style=f"dim {DIM}")
    # clock
    t.append(_clock_badge())
    return Panel(t, border_style=DIM, padding=(0, 1))


def _msg_kokomi(text: str) -> Text:
    t = Text()
    t.append("  KOKOMI", style=f"bold {RED_GLOW}")
    t.append(" ▸ ", style=f"dim {RED_DIM}")
    t.append(text, style=WHITE)
    return t


def _msg_user(text: str) -> Text:
    t = Text()
    t.append("  YOU   ", style=f"bold {MID}")
    t.append(" ▸ ", style=f"dim {DIM}")
    t.append(text, style=LIGHT)
    return t


def _msg_system(text: str) -> Text:
    t = Text()
    t.append("  SYS   ", style=f"dim {DIM}")
    t.append(" ▸ ", style=f"dim {DIM}")
    t.append(text, style=f"dim {MID}")
    return t


def _msg_error(text: str) -> Text:
    t = Text()
    t.append("  ERR   ", style=f"bold {RED}")
    t.append(" ▸ ", style=f"dim {RED_DIM}")
    t.append(text, style=f"{RED_GLOW}")
    return t


def _msg_success(text: str) -> Text:
    t = Text()
    t.append("  OK    ", style=f"bold {GREEN}")
    t.append(" ▸ ", style=f"dim {DIM}")
    t.append(text, style=f"{GREEN}")
    return t


def _divider() -> Rule:
    return Rule(style=f"dim {DIM}", characters="─")


def _thinking_line() -> Text:
    t = Text()
    t.append("  KOKOMI", style=f"bold {RED_DIM}")
    t.append(" ▸ ", style=f"dim {RED_DIM}")
    t.append("processing", style=f"dim {MID}")
    t.append(" ···", style=f"bold {RED}")
    return t


def _help_panel() -> Panel:
    table = Table(box=None, padding=(0, 2), show_header=False)
    table.add_column(style=f"bold {RED}", no_wrap=True)
    table.add_column(style=f"dim {MID}")
    for cmd, desc in HELP_TEXT:
        table.add_row(cmd, desc)
    return Panel(
        table,
        title=f"[bold {RED}]COMMANDS[/]",
        border_style=RED_DIM,
        padding=(1, 2),
    )


def _tasks_panel(tasks: list[dict], title: str) -> Panel:
    if not tasks:
        return Panel(
            Text("  no open tasks", style=f"dim {MID}"),
            title=f"[bold {RED}]{title.upper()}[/]",
            border_style=RED_DIM,
        )
    table = Table(box=box.SIMPLE, show_header=True, header_style=f"dim {MID}")
    table.add_column("ID",   style=f"dim {MID}", width=4)
    table.add_column("PRI",  style=AMBER, width=5)
    table.add_column("TASK", style=WHITE)
    PRIO = {1: "LOW", 2: "NRM", 3: "HI", 4: "CRIT"}
    for t in tasks:
        table.add_row(str(t["id"]), PRIO.get(t["priority"], "?"), t["title"])
    return Panel(
        table,
        title=f"[bold {RED}]{title.upper()} — TASKS[/]",
        border_style=RED_DIM,
        padding=(0, 1),
    )


def _projects_panel(projects: list[dict]) -> Panel:
    if not projects:
        return Panel(
            Text("  no active projects", style=f"dim {MID}"),
            title=f"[bold {RED}]PROJECTS[/]",
            border_style=RED_DIM,
        )
    table = Table(box=box.SIMPLE, show_header=True, header_style=f"dim {MID}")
    table.add_column("ID",   style=f"dim {MID}", width=4)
    table.add_column("NAME", style=f"bold {LIGHT}")
    table.add_column("DESC", style=f"dim {MID}")
    for p in projects:
        table.add_row(str(p["id"]), p["name"], p.get("description", ""))
    return Panel(
        table,
        title=f"[bold {RED}]PROJECTS[/]",
        border_style=RED_DIM,
        padding=(0, 1),
    )


# ── Boot sequence ─────────────────────────────────────────────────────────────

def _boot_sequence():
    console.print()
    console.print(_header())
    console.print()
    with Live(console=console, refresh_per_second=10) as live:
        for i, frame in enumerate(BOOT_FRAMES):
            bar_done  = "█" * (i + 1)
            bar_empty = "░" * (len(BOOT_FRAMES) - i - 1)
            pct       = int(((i + 1) / len(BOOT_FRAMES)) * 100)
            t = Text()
            t.append(f"  [{bar_done}", style=f"bold {RED}")
            t.append(f"{bar_empty}]", style=f"dim {RED_DIM}")
            t.append(f"  {pct:>3}%  ", style=f"bold {LIGHT}")
            t.append(frame, style=f"dim {MID}")
            live.update(t)
            time.sleep(0.18)
    console.print()


# ── Main Terminal class ───────────────────────────────────────────────────────

class Terminal:
    def __init__(self, settings: Settings = None, checkpoint_path: str = None):
        self.settings      = settings or Settings.load()
        self.memory        = MemoryManager(self.settings)
        self.persona       = KokomiPersonality.from_config(self.settings.personality)
        self.shaper        = ResponseShaper(self.persona)
        self.tasks         = TaskManager(self.memory)
        self.projects      = ProjectTracker(self.memory)
        self.scanner       = FileScanner(".")
        self.inference     = KokomiInference(settings=self.settings)
        self.session_msgs  = 0

        ckpt = checkpoint_path or self.inference.find_latest_checkpoint()
        if ckpt and Path(ckpt).exists():
            success = self.inference.load(ckpt)
            if not success:
                logger.error(f"Failed to load checkpoint: {ckpt}")
        else:
            logger.warning("No checkpoint found. Running without model.")

    def run(self):
        _boot_sequence()
        console.print(_status_bar(self.inference.is_loaded(), self.session_msgs))
        console.print()
        console.print(_msg_kokomi(self.persona.greeting()))
        console.print(_msg_system("type /help for commands"))
        console.print()

        while True:
            try:
                # Input prompt
                console.print(Rule(style=f"dim {DIM}", characters="─"))
                raw = console.input(
                    f"[bold {RED}]  ▸[/][{LIGHT}] "
                ).strip()
                console.print(f"[/]", end="")
            except (EOFError, KeyboardInterrupt):
                console.print()
                console.print(_msg_kokomi("Shutting down. Later."))
                console.print()
                break

            if not raw:
                continue

            self.memory.log("user", raw)
            self.session_msgs += 1
            console.print(_msg_user(raw))

            response = self._route(raw)
            if response:
                self.memory.log("assistant", response)
                console.print(_msg_kokomi(response))

    def _route(self, text: str) -> str | None:
        if text.startswith("/"):
            return self._handle_command(text)

        if self.inference.is_loaded():
            # Show thinking indicator
            console.print(_thinking_line())
            try:
                response = self.inference.generate(text, max_tokens=150, temperature=0.8)
                # Clear the thinking line by overwriting
                console.print("\033[1A\033[2K", end="")
                return response
            except Exception as e:
                console.print("\033[1A\033[2K", end="")
                logger.error(f"Generation failed: {e}")
                return f"generation error: {e}"
        else:
            return "model not loaded — run python main.py --train first"

    def _handle_command(self, text: str) -> str | None:
        parts = text.split(maxsplit=2)
        cmd   = parts[0].lower()

        match cmd:
            case "/help":
                console.print(_help_panel())

            case "/status":
                if self.inference.is_loaded():
                    console.print(_msg_success("model loaded and ready"))
                else:
                    console.print(_msg_error("model not loaded"))
                    latest = self.inference.find_latest_checkpoint()
                    if latest:
                        console.print(_msg_system(f"latest checkpoint: {Path(latest).name}"))

            case "/clear":
                console.clear()
                console.print(_header())
                console.print(_status_bar(self.inference.is_loaded(), self.session_msgs))
                console.print()

            case "/projects":
                console.print(_projects_panel(self.projects.list_projects()))

            case "/project":
                if len(parts) < 2:
                    return "usage: /project <name>"
                self.projects.new_project(parts[1])
                console.print(_msg_success(f"project '{parts[1]}' created"))

            case "/tasks":
                if len(parts) < 2:
                    return "usage: /tasks <project>"
                console.print(_tasks_panel(self.tasks.list_tasks(parts[1]), parts[1]))

            case "/task":
                if len(parts) < 3:
                    return "usage: /task <project> <title>"
                self.tasks.add(parts[1], parts[2])
                console.print(_msg_success(f"task added to {parts[1]}"))

            case "/done":
                if len(parts) < 2 or not parts[1].isdigit():
                    return "usage: /done <task_id>"
                self.tasks.complete(int(parts[1]))
                console.print(_msg_success(f"task {parts[1]} completed"))

            case "/remember":
                if len(parts) < 2:
                    return "usage: /remember <text>"
                self.memory.remember(" ".join(parts[1:]))
                console.print(_msg_success("memory stored"))

            case "/recall":
                if len(parts) < 2:
                    return "usage: /recall <query>"
                results = self.memory.recall(" ".join(parts[1:]))
                if not results:
                    console.print(_msg_system("nothing found"))
                else:
                    for m in results:
                        console.print(_msg_system(m["content"]))

            case "/scan":
                tree = self.scanner.tree()
                console.print(Panel(
                    Text(tree, style=f"dim {MID}"),
                    title=f"[bold {RED}]DIRECTORY TREE[/]",
                    border_style=RED_DIM,
                ))

            case "/todos":
                todos = self.scanner.find_todos_summary()
                console.print(Panel(
                    Text(todos, style=f"dim {AMBER}"),
                    title=f"[bold {RED}]TODO SCAN[/]",
                    border_style=RED_DIM,
                ))

            case "/quit" | "/exit":
                console.print(_msg_kokomi("Later."))
                console.print()
                raise SystemExit(0)

            case _:
                console.print(_msg_error(f"unknown command: {cmd} — try /help"))

        return None