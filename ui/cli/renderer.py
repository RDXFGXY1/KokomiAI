"""
Rich-based terminal renderer.
All visual output goes through here — never print() directly in other modules.
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box

console = Console()

# ── NullStudio palette ────────────────────────────────────────
ACCENT  = "#c0392b"   # dark red
DIM     = "#555555"
LIGHT   = "#e8e8e8"
SUCCESS = "#27ae60"
WARN    = "#f39c12"


def print_header():
    """Splash header on startup."""
    art = Text()
    art.append("  K O K O M I\n", style=f"bold {ACCENT}")
    art.append("  local ai assistant\n", style=DIM)
    art.append("  NullStudio Inc.\n", style=DIM)
    console.print(Panel(art, border_style=ACCENT, expand=False))


def print_kokomi(text: str):
    """Kokomi's response."""
    console.print(f"[{ACCENT}]KOKOMI[/] {text}")


def print_user(text: str):
    """Echo user input (for log mode)."""
    console.print(f"[{DIM}]YOU[/] {text}")


def print_info(text: str):
    console.print(f"[{DIM}]ℹ  {text}[/]")


def print_success(text: str):
    console.print(f"[{SUCCESS}]✓  {text}[/]")


def print_error(text: str):
    console.print(f"[{ACCENT}]✗  {text}[/]")


def print_tasks(tasks: list[dict], title: str = "Tasks"):
    if not tasks:
        print_info("No tasks.")
        return
    table = Table(title=title, box=box.SIMPLE, style=DIM, title_style=f"bold {ACCENT}")
    table.add_column("ID",       style=DIM,   width=5)
    table.add_column("Priority", style=WARN,  width=8)
    table.add_column("Task",     style=LIGHT)
    PRIO = {1: "LOW", 2: "NORMAL", 3: "HIGH", 4: "CRIT"}
    for t in tasks:
        table.add_row(str(t["id"]), PRIO.get(t["priority"], "?"), t["title"])
    console.print(table)


def print_projects(projects: list[dict]):
    if not projects:
        print_info("No active projects.")
        return
    table = Table(title="Projects", box=box.SIMPLE, style=DIM, title_style=f"bold {ACCENT}")
    table.add_column("ID",   style=DIM, width=5)
    table.add_column("Name", style=LIGHT)
    table.add_column("Desc", style=DIM)
    for p in projects:
        table.add_row(str(p["id"]), p["name"], p.get("description", ""))
    console.print(table)
