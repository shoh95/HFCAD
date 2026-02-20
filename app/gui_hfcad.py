#!/usr/bin/env python3
"""PyQt GUI for HFCAD sweep generation and execution."""

from __future__ import annotations

import configparser
import csv
import difflib
import itertools
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from PyQt5.QtCore import QThread, Qt, pyqtSignal
    from PyQt5.QtGui import QFont, QFontDatabase, QPixmap
    from PyQt5.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QScrollArea,
        QSpinBox,
        QStackedWidget,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    PYQT_BACKEND = "PyQt5"
    HEADER_STRETCH = QHeaderView.Stretch
    SELECT_ROWS = QAbstractItemView.SelectRows
    EXTENDED_SELECTION = QAbstractItemView.ExtendedSelection
    SCROLLBAR_ALWAYS_ON = Qt.ScrollBarAlwaysOn
    SCROLLBAR_AS_NEEDED = Qt.ScrollBarAsNeeded
    ALIGN_CENTER = Qt.AlignCenter
    KEEP_ASPECT_RATIO = Qt.KeepAspectRatio
    SMOOTH_TRANSFORMATION = Qt.SmoothTransformation
    ITEM_IS_EDITABLE = Qt.ItemIsEditable
    MESSAGEBOX_YES = QMessageBox.Yes
    MESSAGEBOX_NO = QMessageBox.No

    def app_exec(app: QApplication) -> int:
        return app.exec_()

except ImportError:
    from PyQt6.QtCore import QThread, Qt, pyqtSignal
    from PyQt6.QtGui import QFont, QFontDatabase, QPixmap
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QScrollArea,
        QSpinBox,
        QStackedWidget,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    PYQT_BACKEND = "PyQt6"
    HEADER_STRETCH = QHeaderView.ResizeMode.Stretch
    SELECT_ROWS = QAbstractItemView.SelectionBehavior.SelectRows
    EXTENDED_SELECTION = QAbstractItemView.SelectionMode.ExtendedSelection
    SCROLLBAR_ALWAYS_ON = Qt.ScrollBarPolicy.ScrollBarAlwaysOn
    SCROLLBAR_AS_NEEDED = Qt.ScrollBarPolicy.ScrollBarAsNeeded
    ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
    KEEP_ASPECT_RATIO = Qt.AspectRatioMode.KeepAspectRatio
    SMOOTH_TRANSFORMATION = Qt.TransformationMode.SmoothTransformation
    ITEM_IS_EDITABLE = Qt.ItemFlag.ItemIsEditable
    MESSAGEBOX_YES = QMessageBox.StandardButton.Yes
    MESSAGEBOX_NO = QMessageBox.StandardButton.No

    def app_exec(app: QApplication) -> int:
        return app.exec()


CASE_NAME_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
LOG_SUMMARY_KEY = "__summary__"
OUTPUT_PREVIEW_TEXT = 0
OUTPUT_PREVIEW_IMAGE = 1
OUTPUT_HOLD_SLOT_COUNT = 3
CRUISE_ALT_SWEEP_PARAM = "constraint_brief.cruisealt_m"
CLIMB_ALT_SWEEP_PARAM = "constraint_brief.climbalt_m"
TURN_ALT_SWEEP_PARAM = "constraint_brief.turnalt_m"
SYNCED_ALT_SWEEP_PARAMS = (CLIMB_ALT_SWEEP_PARAM, TURN_ALT_SWEEP_PARAM)
CRUISE_ALT_SYNC_LINK_GROUP = "auto_cruisealt"
BOOL_CONFIG_PARAMS = {
    "constraint_sizing.enable",
    "constraint_sizing.scan_quiet",
    "constraint_sizing.use_combined_for_phases",
    "constraint_sizing.ws_auto_widen_enable",
    "constraint_sizing.write_trade_csv",
    "fuselage.legacy_kc_overwrite_bug",
    "solver.initial_bracket_probe_enable",
    "solver.newton_use_bracketing",
    "tail.enable",
    "tail.main_wing_location",
}
ENUM_CONFIG_OPTIONS: Dict[str, Tuple[str, ...]] = {
    "constraint_sizing.selection": ("min_combined_pw", "min_mtom"),
    "constraint_sizing.takeoff_constraint": ("take-off", "climb", "cruise", "turn", "servceil", "combined"),
    "constraint_sizing.climb_constraint": ("take-off", "climb", "cruise", "turn", "servceil", "combined"),
    "constraint_sizing.cruise_constraint": ("take-off", "climb", "cruise", "turn", "servceil", "combined"),
    "constraint_sizing.propulsion_type": ("electric", "piston", "turboprop", "jet", "turbofan"),
    # beta accepts booleans and numeric text in main.py; include common fixed value.
    "fuel_cell_op.beta": ("True", "False", "1.05"),
    # main.py accepts these aliases in _normalize_comp_mass_mode.
    "solver.compressor_mass_mode": ("surrogate", "fast", "approx", "regression", "cadquery", "cad", "full"),
    # 'legacy'/'picard' are fixed-point aliases in main.py.
    "solver.mtom_solver": ("newton", "fixed_point", "legacy", "picard"),
    "tail.wingpos": ("rand", "fwd", "aft"),
    # main.py accepts 'to_climb_only' and legacy alias 'climb_only'.
    "weights.battery_sizing_mode": ("fixed_time", "TO_Climb_only", "to_climb_only", "climb_only"),
}

try:
    from openpyxl import Workbook

    OPENPYXL_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dependency
    Workbook = None  # type: ignore[assignment]
    OPENPYXL_AVAILABLE = False

try:
    from matplotlib.figure import Figure

    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    except Exception:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dependency
    Figure = None  # type: ignore[assignment]
    FigureCanvas = None  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False


class NoWheelComboBox(QComboBox):
    """Combo box that never consumes mouse-wheel scrolling."""

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    """Spin box that never consumes mouse-wheel scrolling."""

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()


@dataclass
class SweepParameter:
    name: str
    min_value: float
    max_value: float
    delta: float
    include_in_name: bool
    abbreviation: str
    link_group: str = ""


@dataclass
class ConstantParameter:
    name: str
    value: str


def new_config_parser() -> configparser.ConfigParser:
    cp = configparser.ConfigParser(
        interpolation=None,
        inline_comment_prefixes=("#", ";"),
    )
    cp.optionxform = str
    return cp


def split_parameter(param_name: str) -> Tuple[str, str]:
    if "." not in param_name:
        raise ValueError(f"Invalid parameter name (expected section.key): {param_name}")
    section, key = param_name.split(".", 1)
    return section, key


def fmt_float_for_ini(value: float) -> str:
    if math.isfinite(value):
        if abs(value - round(value)) <= 1e-12:
            return str(int(round(value)))
        return f"{value:.12g}"
    return str(value)


def fmt_float_for_case(value: float) -> str:
    if abs(value - round(value)) <= 1e-9:
        return str(int(round(value)))
    s = f"{value:.8f}".rstrip("0").rstrip(".")
    return s if s else "0"


def parse_bool_text(value: str, *, allow_numeric: bool = True) -> Optional[bool]:
    normalized = value.strip().lower()
    if normalized in {"yes", "y", "true", "t", "on"}:
        return True
    if normalized in {"no", "n", "false", "f", "off"}:
        return False
    if allow_numeric and normalized in {"1", "0"}:
        return normalized == "1"
    return None


def build_series(min_value: float, max_value: float, delta: float) -> List[float]:
    if delta <= 0:
        raise ValueError("delta must be > 0")

    if min_value == max_value:
        return [min_value]

    direction = 1.0 if max_value > min_value else -1.0
    step = direction * delta
    eps = abs(delta) * 1e-9 + 1e-12

    values: List[float] = []
    current = min_value
    while (current <= max_value + eps) if direction > 0 else (current >= max_value - eps):
        values.append(round(current, 12))
        current += step

    if abs(values[-1] - max_value) > eps:
        values.append(max_value)

    return values


class ExecutionWorker(QThread):
    log_signal = pyqtSignal(str)
    case_done_signal = pyqtSignal(str, bool, int, str)
    summary_signal = pyqtSignal(int, int, int, float, str)

    def __init__(
        self,
        case_paths: Sequence[Path],
        python_bin: str,
        main_script: str,
        out_dir: Path,
        run_dir: Path,
        mode: str,
        jobs: int,
        timeout_sec: int,
        success_artifact: str,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.case_paths = list(case_paths)
        self.python_bin = python_bin
        self.main_script = main_script
        self.out_dir = out_dir
        self.run_dir = run_dir
        self.mode = mode
        self.jobs = jobs
        self.timeout_sec = timeout_sec
        self.success_artifact = success_artifact.strip()
        self._stop_requested = Event()
        self._active_lock = Lock()
        self._active_procs: Dict[str, subprocess.Popen] = {}

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested.is_set()

    def request_stop(self) -> None:
        self._stop_requested.set()
        self.log_signal.emit("[run] stop requested by user.")
        terminated = self._terminate_active_processes(wait=False)
        if terminated > 0:
            self.log_signal.emit(f"[run] stop signal sent to {terminated} active case process(es).")

    def _register_active_proc(self, case_name: str, proc: subprocess.Popen) -> None:
        with self._active_lock:
            self._active_procs[case_name] = proc

    def _unregister_active_proc(self, case_name: str, proc: Optional[subprocess.Popen] = None) -> None:
        with self._active_lock:
            current = self._active_procs.get(case_name)
            if current is None:
                return
            if proc is not None and current is not proc:
                return
            self._active_procs.pop(case_name, None)

    def _terminate_process_tree(self, proc: subprocess.Popen, timeout_s: float = 5.0) -> None:
        if proc.poll() is not None:
            return

        try:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

        try:
            proc.wait(timeout=timeout_s)
            return
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            return

        try:
            if os.name == "nt":
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        try:
            proc.wait(timeout=timeout_s)
        except Exception:
            pass

    def _signal_stop_process(self, proc: subprocess.Popen) -> bool:
        if proc.poll() is not None:
            return False

        try:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                return False
        return True

    def _terminate_active_processes(self, *, wait: bool) -> int:
        with self._active_lock:
            active = list(self._active_procs.items())

        terminated = 0
        for _, proc in active:
            if proc.poll() is not None:
                continue
            if wait:
                self._terminate_process_tree(proc)
                terminated += 1
            elif self._signal_stop_process(proc):
                terminated += 1
        return terminated

    def run(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        ok_list_path = self.run_dir / "ok.list"
        fail_list_path = self.run_dir / "fail.list"
        summary_path = self.run_dir / "summary.txt"
        ok_list_path.write_text("", encoding="utf-8")
        fail_list_path.write_text("", encoding="utf-8")

        total = len(self.case_paths)
        ok = 0
        fail = 0
        start_time = time.time()

        self.log_signal.emit(f"[run] mode={self.mode} jobs={self.jobs} total_cases={total}")
        self.log_signal.emit(f"[run] run_dir={self.run_dir}")

        if self.mode == "parallel":
            with ThreadPoolExecutor(max_workers=max(1, self.jobs)) as pool:
                futures = {pool.submit(self._run_one_case, case_path): case_path for case_path in self.case_paths}
                for future in as_completed(futures):
                    if self._stop_requested.is_set():
                        for pending in futures:
                            if not pending.done():
                                pending.cancel()

                    case_path = futures[future]
                    case_name = case_path.stem
                    log_path = Path("")
                    return_code = 130
                    is_ok = False
                    try:
                        case_name, is_ok, return_code, log_path = future.result()
                    except CancelledError:
                        pass

                    if is_ok:
                        ok += 1
                        with ok_list_path.open("a", encoding="utf-8") as f:
                            f.write(f"{case_name}.ini\n")
                    else:
                        fail += 1
                        with fail_list_path.open("a", encoding="utf-8") as f:
                            f.write(f"{case_name}.ini\n")

                    self.case_done_signal.emit(case_name, is_ok, return_code, str(log_path))
        else:
            for case_path in self.case_paths:
                if self._stop_requested.is_set():
                    break
                case_name, is_ok, return_code, log_path = self._run_one_case(case_path)
                if is_ok:
                    ok += 1
                    with ok_list_path.open("a", encoding="utf-8") as f:
                        f.write(f"{case_name}.ini\n")
                else:
                    fail += 1
                    with fail_list_path.open("a", encoding="utf-8") as f:
                        f.write(f"{case_name}.ini\n")

                self.case_done_signal.emit(case_name, is_ok, return_code, str(log_path))

        elapsed = time.time() - start_time
        lines = [
            f"Run directory : {self.run_dir}",
            f"Output dir    : {self.out_dir}",
            f"Mode          : {self.mode}",
            f"Jobs          : {self.jobs}",
            f"Timeout(s)    : {self.timeout_sec}",
            f"Total cases   : {total}",
            f"OK            : {ok}",
            f"FAIL          : {fail}",
            f"Elapsed(s)    : {elapsed:.1f}",
            f"Stopped       : {'yes' if self._stop_requested.is_set() else 'no'}",
            "",
            f"OK list   : {ok_list_path}",
            f"FAIL list : {fail_list_path}",
        ]
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.summary_signal.emit(total, ok, fail, elapsed, str(summary_path))

    def _run_one_case(self, case_path: Path) -> Tuple[str, bool, int, Path]:
        case_name = case_path.stem
        out_case_name = case_name[6:] if case_name.startswith("input_") else case_name
        log_path = self.run_dir / f"{case_name}.log"

        cmd = [
            self.python_bin,
            self.main_script,
            "-i",
            str(case_path),
            "--outdir",
            str(self.out_dir),
        ]

        return_code = 1
        proc: Optional[subprocess.Popen] = None
        deadline = time.time() + self.timeout_sec if self.timeout_sec > 0 else None
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"[command] {' '.join(cmd)}\n")
            log_file.write(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            log_file.flush()

            try:
                if self._stop_requested.is_set():
                    return_code = 130
                    log_file.write("\n[stopped] Case skipped due to stop request before launch.\n")
                    log_file.flush()
                    artifact_ok = False
                    is_ok = False
                    self.log_signal.emit(f"[case] {case_name} -> FAIL (rc={return_code}, artifact_ok={artifact_ok})")
                    return case_name, is_ok, return_code, log_path

                popen_kwargs: Dict[str, object] = {
                    "stdout": log_file,
                    "stderr": subprocess.STDOUT,
                }
                if os.name == "nt":
                    popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    popen_kwargs["start_new_session"] = True

                proc = subprocess.Popen(cmd, **popen_kwargs)
                self._register_active_proc(case_name, proc)
                while True:
                    if self._stop_requested.is_set():
                        return_code = 130
                        self._terminate_process_tree(proc)
                        log_file.write("\n[stopped] Case stopped by user request.\n")
                        break

                    if deadline is not None and time.time() > deadline:
                        return_code = 124
                        self._terminate_process_tree(proc)
                        log_file.write(f"\n[timeout] Case timed out after {self.timeout_sec} seconds.\n")
                        break

                    poll_rc = proc.poll()
                    if poll_rc is not None:
                        return_code = int(poll_rc)
                        break

                    time.sleep(0.2)
            except Exception as exc:  # pragma: no cover - defensive runtime protection
                return_code = 1
                log_file.write(f"\n[error] Execution failed before process launch/finish: {exc}\n")
            finally:
                self._unregister_active_proc(case_name, proc)

        artifact_ok = False
        if self.success_artifact:
            artifact_path = self.out_dir / out_case_name / self.success_artifact
            artifact_ok = artifact_path.exists() and artifact_path.stat().st_size > 0

        is_ok = ((return_code == 0) or artifact_ok) and (return_code != 130)
        self.log_signal.emit(
            f"[case] {case_name} -> {'OK' if is_ok else 'FAIL'} "
            f"(rc={return_code}, artifact_ok={artifact_ok})"
        )
        return case_name, is_ok, return_code, log_path


class HFCADGui(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"HFCAD PyQt GUI ({PYQT_BACKEND})")
        self.screen_width_px, self.screen_height_px = self._detect_screen_size()
        self.resize(*self._recommended_window_size())
        self.setMinimumSize(1160, 720)

        self.parameter_options: List[str] = []
        self.loaded_parameter_items: List[Tuple[str, str]] = []
        self.visible_loaded_parameter_items: List[Tuple[str, str]] = []
        self.sweep_parameters: List[SweepParameter] = []
        self.constant_parameters: List[ConstantParameter] = []
        self.generated_cases: List[Tuple[Path, Dict[str, str]]] = []
        self.execution_worker: Optional[ExecutionWorker] = None
        self._close_after_stop_requested = False
        self.completed_count = 0
        self.total_count = 0
        self._param_source_kind = ""
        self._param_source_path: Optional[Path] = None
        self._summary_log_lines: List[str] = []
        self._case_log_paths: Dict[str, Path] = {}
        self._mainpage_font_family: Optional[str] = None
        self.mainpage_title_label: Optional[QLabel] = None
        self.mainpage_developer_label: Optional[QLabel] = None
        self.mainpage_description_label: Optional[QLabel] = None
        self.mainpage_contact_label: Optional[QLabel] = None
        self._output_full_pixmap: Optional[QPixmap] = None
        self._collected_case_rows: List[Dict[str, float | int | str]] = []
        self._collected_numeric_columns: List[str] = []
        self._last_exported_excel: Optional[Path] = None
        self._updating_sweep_table = False
        self._updating_const_table = False

        self._build_ui()
        self._build_menu()
        self._set_defaults()

    def _detect_screen_size(self) -> Tuple[int, int]:
        screen = QApplication.primaryScreen()
        if screen is None:
            return 1920, 1080
        geometry = screen.availableGeometry()
        return max(geometry.width(), 1024), max(geometry.height(), 720)

    def _recommended_window_size(self) -> Tuple[int, int]:
        width = min(max(int(self.screen_width_px * 0.86), 1280), 2200)
        height = min(max(int(self.screen_height_px * 0.88), 760), 1360)
        return width, height

    @staticmethod
    def _korean_capable_font_family() -> Optional[str]:
        candidates = [
            "Noto Sans CJK KR",
            "Noto Sans KR",
            "Malgun Gothic",
            "Apple SD Gothic Neo",
            "NanumGothic",
            "Nanum Gothic",
            "UnDotum",
            "Gulim",
            "Batang",
            "Arial Unicode MS",
        ]
        try:
            families = set(QFontDatabase().families())
        except Exception:
            return None
        for family in candidates:
            if family in families:
                return family
        return None

    @staticmethod
    def _table_min_height_for_rows(table: QTableWidget, min_rows: int = 5) -> int:
        row_h = max(table.verticalHeader().defaultSectionSize(), 24)
        header_h = max(table.horizontalHeader().sizeHint().height(), 24)
        frame_h = table.frameWidth() * 2
        return header_h + frame_h + row_h * max(min_rows, 1) + 4

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_mainpage_fonts()
        self._update_output_image_display()

    def _apply_mainpage_fonts(self) -> None:
        if (
            self.mainpage_title_label is None
            or self.mainpage_developer_label is None
            or self.mainpage_description_label is None
            or self.mainpage_contact_label is None
        ):
            return

        scale_ref = min(max(self.width(), 640), max(self.height(), 480))
        title_pt = max(14, min(32, int(scale_ref * 0.024)))
        medium_pt = max(11, min(22, int(scale_ref * 0.016)))
        body_pt = max(9, min(16, int(scale_ref * 0.011)))

        title_font = QFont(self.font())
        title_font.setPointSize(title_pt)
        title_font.setBold(True)
        if self._mainpage_font_family:
            title_font.setFamily(self._mainpage_font_family)
        self.mainpage_title_label.setFont(title_font)

        medium_font = QFont(self.font())
        medium_font.setPointSize(medium_pt)
        if self._mainpage_font_family:
            medium_font.setFamily(self._mainpage_font_family)
        self.mainpage_developer_label.setFont(medium_font)

        body_font = QFont(self.font())
        body_font.setPointSize(body_pt)
        if self._mainpage_font_family:
            body_font.setFamily(self._mainpage_font_family)
        self.mainpage_description_label.setFont(body_font)
        self.mainpage_contact_label.setFont(body_font)

    @staticmethod
    def _normalize_path(path: Path) -> Path:
        try:
            return path.expanduser().resolve()
        except Exception:
            return path.expanduser()

    def _set_parameter_source(self, source_kind: str, source_path: Path) -> bool:
        normalized_path = self._normalize_path(source_path)
        changed = (source_kind != self._param_source_kind) or (normalized_path != self._param_source_path)
        self._param_source_kind = source_kind
        self._param_source_path = normalized_path
        return changed

    def _load_parameter_items(
        self,
        loaded_items: List[Tuple[str, str]],
        *,
        reset_user_parameters: bool,
    ) -> None:
        self.loaded_parameter_items = sorted(loaded_items, key=lambda x: x[0])
        self.parameter_options = [name for name, _ in self.loaded_parameter_items]
        self._refresh_parameter_selectors()
        self._refresh_loaded_params_table()

        if reset_user_parameters:
            self.sweep_parameters = []
            self.constant_parameters = []
        else:
            valid_params = set(self.parameter_options)
            self.sweep_parameters = [sp for sp in self.sweep_parameters if sp.name in valid_params]
            sweep_names = {sp.name for sp in self.sweep_parameters}
            self.constant_parameters = [
                cp for cp in self.constant_parameters if cp.name in valid_params and cp.name not in sweep_names
            ]

        self._commit_parameter_change(
            enforce_sync=True,
            refresh_sweep_table=True,
            refresh_constant_table=True,
            refresh_sweep_editor=True,
            refresh_const_editor=True,
        )

    def _filter_query(self) -> str:
        return self.parameter_search_edit.text().strip().lower()

    @staticmethod
    def _is_subsequence(query: str, text: str) -> bool:
        if not query:
            return True
        idx = 0
        for ch in text:
            if ch == query[idx]:
                idx += 1
                if idx >= len(query):
                    return True
        return False

    @staticmethod
    def _fuzzy_ratio(query: str, text: str) -> float:
        if not query or not text:
            return 0.0
        return difflib.SequenceMatcher(None, query, text).ratio()

    def _fuzzy_match_text(self, query: str, text: str) -> bool:
        if not query:
            return True
        hay = text.strip().lower()
        if not hay:
            return False
        if query in hay:
            return True
        if self._is_subsequence(query, hay):
            return True

        # Compare against whole text and tokenized chunks for typo-tolerant matches.
        if len(query) >= 3 and self._fuzzy_ratio(query, hay) >= 0.72:
            return True
        for token in re.split(r"[.\-_\s]+", hay):
            if not token:
                continue
            if query in token or self._is_subsequence(query, token):
                return True
            if len(query) >= 3 and self._fuzzy_ratio(query, token) >= 0.72:
                return True
        return False

    def _filtered_parameter_options(self) -> List[str]:
        query = self._filter_query()
        if not query:
            return list(self.parameter_options)
        return [name for name in self.parameter_options if self._fuzzy_match_text(query, name)]

    def _refresh_parameter_selectors(self) -> None:
        filtered_options = self._filtered_parameter_options()
        previous_sweep = self.sweep_param_combo.currentText().strip()
        previous_const = self.const_param_combo.currentText().strip()

        self.sweep_param_combo.blockSignals(True)
        self.sweep_param_combo.clear()
        self.sweep_param_combo.addItems(filtered_options)
        if previous_sweep:
            idx = self.sweep_param_combo.findText(previous_sweep)
            if idx >= 0:
                self.sweep_param_combo.setCurrentIndex(idx)
        self.sweep_param_combo.blockSignals(False)

        self.const_param_combo.blockSignals(True)
        self.const_param_combo.clear()
        self.const_param_combo.addItems(filtered_options)
        if previous_const:
            idx = self.const_param_combo.findText(previous_const)
            if idx >= 0:
                self.const_param_combo.setCurrentIndex(idx)
        self.const_param_combo.blockSignals(False)

    def on_parameter_search_changed(self, _: str = "") -> None:
        self._refresh_loaded_params_table()
        self._refresh_parameter_selectors()
        self.on_sweep_parameter_changed()
        self.on_const_parameter_changed()

    def _build_ui(self) -> None:
        central = QWidget()
        root_layout = QVBoxLayout(central)

        tabs = QTabWidget()
        root_layout.addWidget(tabs)
        self.setCentralWidget(central)

        main_page = QWidget()
        generator_page = QWidget()
        output_page = QWidget()
        logs_page = QWidget()
        tabs.addTab(main_page, "MainPage")
        tabs.addTab(generator_page, "Auto Input Generator")
        tabs.addTab(output_page, "Output")
        tabs.addTab(logs_page, "Execution Logs")

        main_page_layout = QVBoxLayout(main_page)
        self._mainpage_font_family = self._korean_capable_font_family()
        main_page_layout.addStretch(1)
        self.mainpage_title_label = QLabel(
            "GNU Hydrogen Fuelcell - Battery Hybrid Aircraft Conceptual Design Code V2"
        )
        self.mainpage_title_label.setAlignment(ALIGN_CENTER)
        self.mainpage_developer_label = QLabel("Code builder: Seunghwan Oh")
        self.mainpage_developer_label.setAlignment(ALIGN_CENTER)
        self.mainpage_description_label = QLabel(
            "Initial code was built by Seung-uk Oh, Won-ho Kim in 2024. "
            "The code was then expanded to support advanced solver function and GUI wapper by Seung-hwan Oh."
        )
        self.mainpage_description_label.setAlignment(ALIGN_CENTER)
        self.mainpage_description_label.setWordWrap(True)
        self.mainpage_contact_label = QLabel("Contact: shoh95@gnu.ac.kr")
        self.mainpage_contact_label.setAlignment(ALIGN_CENTER)
        self._apply_mainpage_fonts()
        main_page_layout.addWidget(self.mainpage_title_label)
        main_page_layout.addSpacing(12)
        main_page_layout.addWidget(self.mainpage_developer_label)
        main_page_layout.addSpacing(8)
        main_page_layout.addWidget(self.mainpage_description_label)
        main_page_layout.addSpacing(8)
        main_page_layout.addWidget(self.mainpage_contact_label)
        main_page_layout.addStretch(2)

        generator_page_layout = QVBoxLayout(generator_page)
        generator_scroll = QScrollArea()
        generator_scroll.setWidgetResizable(True)
        generator_scroll.setVerticalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
        generator_scroll.setHorizontalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
        generator_page_layout.addWidget(generator_scroll)

        generator_container = QWidget()
        generator_scroll.setWidget(generator_container)
        generator_layout = QVBoxLayout(generator_container)

        file_group = QGroupBox("Template and Paths")
        file_layout = QGridLayout(file_group)

        self.template_edit = QLineEdit()
        self.template_browse_btn = QPushButton("Browse...")
        self.template_load_btn = QPushButton("Load Parameters")
        file_layout.addWidget(QLabel("Template INI"), 0, 0)
        file_layout.addWidget(self.template_edit, 0, 1)
        file_layout.addWidget(self.template_browse_btn, 0, 2)
        file_layout.addWidget(self.template_load_btn, 0, 3)

        self.input_out_edit = QLineEdit()
        self.input_out_browse_btn = QPushButton("Browse...")
        file_layout.addWidget(QLabel("Generated Input Dir"), 1, 0)
        file_layout.addWidget(self.input_out_edit, 1, 1)
        file_layout.addWidget(self.input_out_browse_btn, 1, 2)

        self.results_out_edit = QLineEdit()
        self.results_out_browse_btn = QPushButton("Browse...")
        file_layout.addWidget(QLabel("Results Output Dir"), 2, 0)
        file_layout.addWidget(self.results_out_edit, 2, 1)
        file_layout.addWidget(self.results_out_browse_btn, 2, 2)

        self.log_root_edit = QLineEdit()
        self.log_root_browse_btn = QPushButton("Browse...")
        file_layout.addWidget(QLabel("Run Log Root Dir"), 3, 0)
        file_layout.addWidget(self.log_root_edit, 3, 1)
        file_layout.addWidget(self.log_root_browse_btn, 3, 2)

        self.python_edit = QLineEdit()
        file_layout.addWidget(QLabel("Python Executable"), 4, 0)
        file_layout.addWidget(self.python_edit, 4, 1, 1, 3)

        self.main_script_edit = QLineEdit()
        self.main_script_browse_btn = QPushButton("Browse...")
        file_layout.addWidget(QLabel("Main Script"), 5, 0)
        file_layout.addWidget(self.main_script_edit, 5, 1)
        file_layout.addWidget(self.main_script_browse_btn, 5, 2)

        self.case_prefix_edit = QLineEdit()
        file_layout.addWidget(QLabel("Case Name Prefix"), 6, 0)
        file_layout.addWidget(self.case_prefix_edit, 6, 1, 1, 3)

        generator_layout.addWidget(file_group)

        loaded_group = QGroupBox("Loaded INI Parameters")
        loaded_layout = QVBoxLayout(loaded_group)
        loaded_actions = QHBoxLayout()
        self.import_case_btn = QPushButton("Import Whole Case File")
        self.import_case_folder_btn = QPushButton("Import Case Folder")
        self.add_loaded_to_sweep_btn = QPushButton("Add Selected To Sweep")
        self.add_loaded_to_constant_btn = QPushButton("Add Selected To Constants")
        loaded_actions.addWidget(self.import_case_btn)
        loaded_actions.addWidget(self.import_case_folder_btn)
        loaded_actions.addWidget(self.add_loaded_to_sweep_btn)
        loaded_actions.addWidget(self.add_loaded_to_constant_btn)
        loaded_actions.addStretch(1)
        loaded_layout.addLayout(loaded_actions)

        loaded_search_row = QHBoxLayout()
        self.parameter_search_edit = QLineEdit()
        self.parameter_search_edit.setPlaceholderText("Search (supports fuzzy find, e.g., typos/partial)...")
        self.parameter_search_edit.setClearButtonEnabled(True)
        loaded_search_row.addWidget(QLabel("Search"))
        loaded_search_row.addWidget(self.parameter_search_edit, 1)
        loaded_layout.addLayout(loaded_search_row)

        self.loaded_params_table = QTableWidget(0, 2)
        self.loaded_params_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.loaded_params_table.horizontalHeader().setSectionResizeMode(HEADER_STRETCH)
        self.loaded_params_table.setSelectionBehavior(SELECT_ROWS)
        self.loaded_params_table.setSelectionMode(EXTENDED_SELECTION)
        self.loaded_params_table.setVerticalScrollBarPolicy(SCROLLBAR_ALWAYS_ON)
        min_rows_height = self._table_min_height_for_rows(self.loaded_params_table, min_rows=5)
        loaded_min_h = int(max(180, min(self.screen_height_px * 0.20, 320)))
        loaded_max_h = int(max(240, min(self.screen_height_px * 0.28, 420)))
        self.loaded_params_table.setMinimumHeight(max(loaded_min_h, min_rows_height))
        self.loaded_params_table.setMaximumHeight(loaded_max_h)
        loaded_layout.addWidget(self.loaded_params_table)
        generator_layout.addWidget(loaded_group)

        sweep_group = QGroupBox("Sweep Parameters (min/max/delta + case-name options)")
        sweep_layout = QVBoxLayout(sweep_group)
        sweep_form = QGridLayout()

        self.sweep_param_combo = NoWheelComboBox()
        self.sweep_min_edit = QLineEdit()
        self.sweep_max_edit = QLineEdit()
        self.sweep_delta_edit = QLineEdit()
        self.sweep_include_checkbox = QCheckBox("Include in case name")
        self.sweep_include_checkbox.setChecked(True)
        self.sweep_abbrev_edit = QLineEdit()
        self.sweep_link_edit = QLineEdit()
        self.sweep_add_btn = QPushButton("Add Sweep Parameter")
        self.sweep_remove_btn = QPushButton("Remove Selected")
        self.sweep_auto_link_btn = QPushButton("Auto-Link Selected")
        self.sweep_set_link_btn = QPushButton("Set Link (Selected)")
        self.sweep_clear_link_btn = QPushButton("Clear Link (Selected)")

        sweep_form.addWidget(QLabel("Parameter"), 0, 0)
        sweep_form.addWidget(self.sweep_param_combo, 0, 1)
        sweep_form.addWidget(QLabel("Min"), 0, 2)
        sweep_form.addWidget(self.sweep_min_edit, 0, 3)
        sweep_form.addWidget(QLabel("Max"), 0, 4)
        sweep_form.addWidget(self.sweep_max_edit, 0, 5)
        sweep_form.addWidget(QLabel("Delta"), 0, 6)
        sweep_form.addWidget(self.sweep_delta_edit, 0, 7)
        sweep_form.addWidget(self.sweep_include_checkbox, 1, 0, 1, 2)
        sweep_form.addWidget(QLabel("Abbreviation"), 1, 2)
        sweep_form.addWidget(self.sweep_abbrev_edit, 1, 3)
        sweep_form.addWidget(QLabel("Link group"), 1, 4)
        sweep_form.addWidget(self.sweep_link_edit, 1, 5)
        sweep_form.addWidget(self.sweep_remove_btn, 1, 7)
        sweep_form.addWidget(self.sweep_auto_link_btn, 2, 4)
        sweep_form.addWidget(self.sweep_add_btn, 2, 5)
        sweep_form.addWidget(self.sweep_set_link_btn, 2, 6)
        sweep_form.addWidget(self.sweep_clear_link_btn, 2, 7)
        sweep_layout.addLayout(sweep_form)

        self.sweep_table = QTableWidget(0, 7)
        self.sweep_table.setHorizontalHeaderLabels(
            ["Parameter", "Min", "Max", "Delta", "In Case Name", "Abbrev", "Link"]
        )
        self.sweep_table.horizontalHeader().setSectionResizeMode(HEADER_STRETCH)
        self.sweep_table.setSelectionBehavior(SELECT_ROWS)
        self.sweep_table.setSelectionMode(EXTENDED_SELECTION)
        self.sweep_table.setVerticalScrollBarPolicy(SCROLLBAR_ALWAYS_ON)
        self.sweep_table.setMinimumHeight(self._table_min_height_for_rows(self.sweep_table, min_rows=5))
        sweep_layout.addWidget(self.sweep_table)
        generator_layout.addWidget(sweep_group)

        const_group = QGroupBox("Constant Parameters (applied to all cases)")
        const_layout = QVBoxLayout(const_group)
        const_form = QGridLayout()

        self.const_param_combo = NoWheelComboBox()
        self.const_value_edit = QLineEdit()
        self.const_value_combo = NoWheelComboBox()
        self.const_value_stack = QStackedWidget()
        self.const_value_stack.addWidget(self.const_value_edit)
        self.const_value_stack.addWidget(self.const_value_combo)
        self.const_add_btn = QPushButton("Add Constant Parameter")
        self.const_remove_btn = QPushButton("Remove Selected")
        self.const_add_all_btn = QPushButton("Add All Non-Sweep")
        const_form.addWidget(QLabel("Parameter"), 0, 0)
        const_form.addWidget(self.const_param_combo, 0, 1)
        const_form.addWidget(QLabel("Value"), 0, 2)
        const_form.addWidget(self.const_value_stack, 0, 3)
        const_form.addWidget(self.const_add_btn, 0, 4)
        const_form.addWidget(self.const_remove_btn, 0, 5)
        const_form.addWidget(self.const_add_all_btn, 0, 6)
        const_layout.addLayout(const_form)

        self.const_table = QTableWidget(0, 2)
        self.const_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.const_table.horizontalHeader().setSectionResizeMode(HEADER_STRETCH)
        self.const_table.setVerticalScrollBarPolicy(SCROLLBAR_ALWAYS_ON)
        self.const_table.setMinimumHeight(self._table_min_height_for_rows(self.const_table, min_rows=5))
        const_layout.addWidget(self.const_table)
        generator_layout.addWidget(const_group)

        execute_group = QGroupBox("Generate and Execute")
        execute_layout = QGridLayout(execute_group)

        self.mode_combo = NoWheelComboBox()
        self.mode_combo.addItems(["sequential", "parallel"])
        self.jobs_spin = NoWheelSpinBox()
        self.jobs_spin.setMinimum(1)
        self.jobs_spin.setMaximum(128)
        self.jobs_spin.setValue(4)

        self.timeout_spin = NoWheelSpinBox()
        self.timeout_spin.setMinimum(0)
        self.timeout_spin.setMaximum(10_000_000)
        self.timeout_spin.setValue(0)
        self.timeout_spin.setSuffix(" s")

        self.success_artifact_edit = QLineEdit("ConvergedData.txt")
        self.strict_ini_check = QCheckBox("Strict INI standards")
        self.strict_ini_check.setChecked(False)
        self.strict_ini_check.setToolTip(
            "If enabled, non-standard enum/bool values fail generated INI validation."
        )
        self.update_params_btn = QPushButton("Update Params")
        self.refresh_all_btn = QPushButton("Refresh All")
        self.make_input_btn = QPushButton("Make Input Files")
        self.execute_btn = QPushButton("Execute Cases")
        self.stop_btn = QPushButton("Stop Run")
        self._set_stop_button_running_state(False)

        execute_layout.addWidget(QLabel("Run mode"), 0, 0)
        execute_layout.addWidget(self.mode_combo, 0, 1)
        execute_layout.addWidget(QLabel("Parallel jobs"), 0, 2)
        execute_layout.addWidget(self.jobs_spin, 0, 3)
        execute_layout.addWidget(QLabel("Timeout per case"), 0, 4)
        execute_layout.addWidget(self.timeout_spin, 0, 5)
        execute_layout.addWidget(self.strict_ini_check, 0, 6, 1, 2)
        execute_layout.addWidget(QLabel("Success artifact"), 1, 0)
        execute_layout.addWidget(self.success_artifact_edit, 1, 1, 1, 3)
        execute_layout.addWidget(self.update_params_btn, 1, 4)
        execute_layout.addWidget(self.refresh_all_btn, 1, 5)
        execute_layout.addWidget(self.make_input_btn, 1, 6)
        execute_layout.addWidget(self.execute_btn, 1, 7)
        execute_layout.addWidget(self.stop_btn, 1, 8)
        generator_layout.addWidget(execute_group)

        diff_group = QGroupBox("Major Differences Between Cases")
        diff_group.setMaximumHeight(int(max(120, min(self.screen_height_px * 0.16, 220))))
        diff_layout = QVBoxLayout(diff_group)
        self.diff_text = QPlainTextEdit()
        self.diff_text.setReadOnly(True)
        diff_layout.addWidget(self.diff_text)
        generator_layout.addWidget(diff_group)

        generator_layout.setStretch(0, 0)  # Template and Paths
        generator_layout.setStretch(1, 1)  # Loaded INI Parameters
        generator_layout.setStretch(2, 2)  # Sweep Parameters
        generator_layout.setStretch(3, 1)  # Constant Parameters
        generator_layout.setStretch(4, 0)  # Generate and Execute
        generator_layout.setStretch(5, 0)  # Major Differences

        output_layout = QVBoxLayout(output_page)
        output_group = QGroupBox("Case Outputs")
        output_group_layout = QGridLayout(output_group)

        self.output_results_dir_label = QLabel("")
        self.output_results_dir_label.setWordWrap(True)
        self.output_case_combo = NoWheelComboBox()
        self.output_file_combo = NoWheelComboBox()
        self.output_search_edit = QLineEdit()
        self.output_search_edit.setPlaceholderText("Search cases/files/parameters...")
        self.output_search_edit.setClearButtonEnabled(True)
        self.output_refresh_btn = QPushButton("Refresh")
        self.output_update_notebook_btn = QPushButton("Update Selected Notebook")
        self.output_export_excel_btn = QPushButton("Export All Case Data -> Excel")
        self.output_file_path_label = QLabel("No output file selected.")
        self.output_file_path_label.setWordWrap(True)
        self.output_last_excel_label = QLabel("No Excel export yet.")
        self.output_last_excel_label.setWordWrap(True)
        self.output_plot_x_combo = NoWheelComboBox()
        self.output_plot_y_combo = NoWheelComboBox()
        self.output_hold_pairs: List[Tuple[NoWheelComboBox, NoWheelComboBox]] = []
        for _ in range(OUTPUT_HOLD_SLOT_COUNT):
            self.output_hold_pairs.append((NoWheelComboBox(), NoWheelComboBox()))
        # Backward-compatible aliases for the first hold slot.
        self.output_hold_param_combo = self.output_hold_pairs[0][0]
        self.output_hold_value_combo = self.output_hold_pairs[0][1]
        self.output_plot_type_combo = NoWheelComboBox()
        self.output_plot_type_combo.addItems(
            ["function(mean)", "function(mean+minmax)", "scatter(raw)", "line(raw)"]
        )
        self.output_plot_btn = QPushButton("Draw Graph")

        output_group_layout.addWidget(QLabel("Results Dir"), 0, 0)
        output_group_layout.addWidget(self.output_results_dir_label, 0, 1, 1, 4)
        output_group_layout.addWidget(QLabel("Case"), 1, 0)
        output_group_layout.addWidget(self.output_case_combo, 1, 1)
        output_group_layout.addWidget(self.output_refresh_btn, 1, 2)
        output_group_layout.addWidget(self.output_update_notebook_btn, 1, 3)
        output_group_layout.addWidget(self.output_export_excel_btn, 1, 4)
        output_group_layout.addWidget(QLabel("Search"), 2, 0)
        output_group_layout.addWidget(self.output_search_edit, 2, 1, 1, 4)
        output_group_layout.addWidget(QLabel("File"), 3, 0)
        output_group_layout.addWidget(self.output_file_combo, 3, 1, 1, 4)
        output_group_layout.addWidget(QLabel("Selected Path"), 4, 0)
        output_group_layout.addWidget(self.output_file_path_label, 4, 1, 1, 4)
        output_group_layout.addWidget(QLabel("Exported Excel"), 5, 0)
        output_group_layout.addWidget(self.output_last_excel_label, 5, 1, 1, 4)
        output_group_layout.addWidget(QLabel("Graph X"), 6, 0)
        output_group_layout.addWidget(self.output_plot_x_combo, 6, 1)
        output_group_layout.addWidget(QLabel("Graph Y"), 6, 2)
        output_group_layout.addWidget(self.output_plot_y_combo, 6, 3)
        output_group_layout.addWidget(self.output_plot_type_combo, 6, 4)
        output_group_layout.addWidget(self.output_plot_btn, 6, 5)
        for hold_idx, (hold_param_combo, hold_value_combo) in enumerate(self.output_hold_pairs):
            row = 7 + hold_idx
            output_group_layout.addWidget(QLabel(f"Hold Param {hold_idx + 1}"), row, 0)
            output_group_layout.addWidget(hold_param_combo, row, 1, 1, 2)
            output_group_layout.addWidget(QLabel(f"Hold Value {hold_idx + 1}"), row, 3)
            output_group_layout.addWidget(hold_value_combo, row, 4, 1, 2)
        output_layout.addWidget(output_group)

        self.output_text = QPlainTextEdit()
        self.output_text.setReadOnly(True)

        self.output_image_label = QLabel("No image selected.")
        self.output_image_label.setAlignment(ALIGN_CENTER)
        self.output_image_scroll = QScrollArea()
        self.output_image_scroll.setWidgetResizable(True)
        self.output_image_scroll.setVerticalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
        self.output_image_scroll.setHorizontalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
        self.output_image_scroll.setWidget(self.output_image_label)

        self.output_preview_stack = QStackedWidget()
        self.output_preview_stack.addWidget(self.output_text)
        self.output_preview_stack.addWidget(self.output_image_scroll)
        self.output_preview_stack.setCurrentIndex(OUTPUT_PREVIEW_TEXT)
        output_layout.addWidget(self.output_preview_stack, 2)

        if MATPLOTLIB_AVAILABLE and Figure is not None and FigureCanvas is not None:
            self.output_plot_figure = Figure(figsize=(6, 3), dpi=100)
            self.output_plot_canvas = FigureCanvas(self.output_plot_figure)
            output_layout.addWidget(self.output_plot_canvas, 2)
        else:
            self.output_plot_figure = None
            self.output_plot_canvas = None
            self.output_plot_canvas_label = QLabel(
                "Graph view unavailable: matplotlib backend could not be loaded."
            )
            self.output_plot_canvas_label.setWordWrap(True)
            output_layout.addWidget(self.output_plot_canvas_label)

        logs_layout = QVBoxLayout(logs_page)
        status_row = QHBoxLayout()
        self.progress_label = QLabel("No run in progress.")
        status_row.addWidget(self.progress_label)
        status_row.addStretch(1)
        logs_layout.addLayout(status_row)

        log_view_row = QHBoxLayout()
        self.log_view_combo = NoWheelComboBox()
        self.log_view_combo.addItem("Summary", LOG_SUMMARY_KEY)
        log_view_row.addWidget(QLabel("Display"))
        log_view_row.addWidget(self.log_view_combo)
        log_view_row.addStretch(1)
        logs_layout.addLayout(log_view_row)

        self.logs_text = QPlainTextEdit()
        self.logs_text.setReadOnly(True)
        logs_layout.addWidget(self.logs_text)

        self.template_browse_btn.clicked.connect(self.on_browse_template)
        self.template_load_btn.clicked.connect(self.on_load_parameters)
        self.template_edit.editingFinished.connect(self.on_load_parameters)
        self.input_out_browse_btn.clicked.connect(self.on_browse_input_out_dir)
        self.results_out_browse_btn.clicked.connect(self.on_browse_results_dir)
        self.results_out_edit.editingFinished.connect(self.on_results_dir_changed)
        self.log_root_browse_btn.clicked.connect(self.on_browse_log_root_dir)
        self.main_script_browse_btn.clicked.connect(self.on_browse_main_script)
        self.sweep_add_btn.clicked.connect(self.on_add_sweep_parameter)
        self.sweep_remove_btn.clicked.connect(self.on_remove_sweep_parameter)
        self.sweep_auto_link_btn.clicked.connect(self.on_auto_link_selected_sweeps)
        self.sweep_set_link_btn.clicked.connect(self.on_set_sweep_link_group)
        self.sweep_clear_link_btn.clicked.connect(self.on_clear_sweep_link_group)
        self.const_add_btn.clicked.connect(self.on_add_constant_parameter)
        self.const_remove_btn.clicked.connect(self.on_remove_constant_parameter)
        self.const_add_all_btn.clicked.connect(self.on_add_all_constants_except_sweep)
        self.update_params_btn.clicked.connect(self.on_update_params_from_tables)
        self.refresh_all_btn.clicked.connect(self.on_refresh_all)
        self.make_input_btn.clicked.connect(self.on_make_input_files)
        self.execute_btn.clicked.connect(self.on_execute_cases)
        self.stop_btn.clicked.connect(self.on_stop_run)
        self.log_view_combo.currentIndexChanged.connect(self.on_log_view_changed)
        self.output_case_combo.currentIndexChanged.connect(self.on_output_case_changed)
        self.output_file_combo.currentIndexChanged.connect(self.on_output_file_changed)
        self.output_search_edit.textChanged.connect(self.on_output_search_changed)
        self.output_refresh_btn.clicked.connect(self.on_refresh_outputs)
        self.output_update_notebook_btn.clicked.connect(self.on_update_selected_notebook)
        self.output_export_excel_btn.clicked.connect(self.on_export_all_case_data_to_excel)
        self.output_plot_btn.clicked.connect(self.on_draw_collected_plot)
        for hold_param_combo, _ in self.output_hold_pairs:
            hold_param_combo.currentIndexChanged.connect(self.on_output_hold_parameter_changed)
        self.sweep_param_combo.currentTextChanged.connect(self.on_sweep_parameter_changed)
        self.const_param_combo.currentTextChanged.connect(self.on_const_parameter_changed)
        self.const_value_combo.currentTextChanged.connect(self.const_value_edit.setText)
        self.sweep_table.itemChanged.connect(self.on_sweep_table_item_changed)
        self.const_table.itemChanged.connect(self.on_const_table_item_changed)
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        self.add_loaded_to_sweep_btn.clicked.connect(self.on_add_loaded_selection_to_sweep)
        self.add_loaded_to_constant_btn.clicked.connect(self.on_add_loaded_selection_to_constants)
        self.import_case_btn.clicked.connect(self.on_import_case_file)
        self.import_case_folder_btn.clicked.connect(self.on_import_case_folder)
        self.parameter_search_edit.textChanged.connect(self.on_parameter_search_changed)

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        load_ini_action = file_menu.addAction("Load INI...")
        load_ini_action.setShortcut("Ctrl+O")
        load_ini_action.triggered.connect(self.on_menu_load_ini)

        import_case_action = file_menu.addAction("Import Whole Case File...")
        import_case_action.setShortcut("Ctrl+I")
        import_case_action.triggered.connect(self.on_import_case_file)

        import_folder_action = file_menu.addAction("Import Case Folder...")
        import_folder_action.setShortcut("Ctrl+Shift+I")
        import_folder_action.triggered.connect(self.on_import_case_folder)

        refresh_all_action = file_menu.addAction("Refresh All")
        refresh_all_action.setShortcut("F5")
        refresh_all_action.triggered.connect(self.on_refresh_all)

        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)

    def _set_defaults(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        default_template = self._find_default_template(repo_root)
        default_main = self._find_default_main_script(repo_root)

        self.template_edit.setText(str(default_template))
        self.input_out_edit.setText(str(repo_root / "app" / "input_sweep_gui"))
        self.results_out_edit.setText(str(repo_root / "results" / "input_sweep_gui"))
        self.log_root_edit.setText(str(repo_root / "results" / "_logs"))
        self.python_edit.setText(sys.executable)
        self.main_script_edit.setText(str(default_main))
        self.case_prefix_edit.setText("")
        self.strict_ini_check.setChecked(False)

        self.on_load_parameters()
        self.on_mode_changed()
        self.on_refresh_outputs()

    @staticmethod
    def _find_default_main_script(repo_root: Path) -> Path:
        preferred = repo_root / "app" / "main.py"
        if preferred.exists():
            return preferred
        fallback = repo_root / "app" / "HFCBattACDesign_SH_OOP_inpu_260117.py"
        return fallback

    @staticmethod
    def _find_default_template(repo_root: Path) -> Path:
        preferred = repo_root / "app" / "example_input" / "h3000_M0.3_R1000km_PL1000kg.ini"
        if preferred.exists():
            return preferred

        candidates = sorted((repo_root / "app").glob("*.ini"))
        if candidates:
            return candidates[0]
        return repo_root / "app" / "input_HFCAD.ini"

    def on_browse_template(self) -> None:
        start_dir = str(Path(self.template_edit.text()).expanduser().parent)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Template INI",
            start_dir,
            "INI Files (*.ini *.txt);;All Files (*)",
        )
        if path:
            self.template_edit.setText(path)
            self.on_load_parameters()

    def on_menu_load_ini(self) -> None:
        self.on_browse_template()

    def on_import_case_file(self) -> None:
        start_dir = str(Path(self.template_edit.text()).expanduser().parent)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Whole Case INI",
            start_dir,
            "INI Files (*.ini *.txt);;All Files (*)",
        )
        if not path:
            return

        case_path = Path(path).expanduser()
        cp = new_config_parser()
        read_ok = cp.read(str(case_path))
        if not read_ok:
            self._show_warning("Could not read selected case INI file.")
            return

        loaded_items: List[Tuple[str, str]] = []
        for section in cp.sections():
            for key in cp[section].keys():
                loaded_items.append((f"{section}.{key}", cp[section][key]))

        self._set_parameter_source("case_file", case_path)
        self.template_edit.setText(str(case_path))
        self._load_parameter_items(loaded_items, reset_user_parameters=True)

        # Import the full case as constants.
        self.constant_parameters = [
            ConstantParameter(name=name, value=value)
            for name, value in self.loaded_parameter_items
        ]
        self._commit_parameter_change(
            refresh_constant_table=True,
            refresh_const_editor=True,
        )

        self._log(
            f"[import-case] Imported {len(self.loaded_parameter_items)} parameters from {case_path} "
            f"and loaded them into constants."
        )

    def on_import_case_folder(self) -> None:
        start_dir = str(Path(self.input_out_edit.text()).expanduser())
        folder = QFileDialog.getExistingDirectory(self, "Import Case Folder", start_dir)
        if not folder:
            return

        folder_path = Path(folder).expanduser()
        ini_files = sorted(folder_path.glob("*.ini"))
        if not ini_files:
            self._show_warning("No .ini files found in selected folder.")
            return

        case_dicts: List[Dict[str, str]] = []
        for ini_path in ini_files:
            data = self._read_ini_dict(ini_path)
            if data:
                case_dicts.append(data)

        if not case_dicts:
            self._show_warning("No readable INI files found in selected folder.")
            return

        self._set_parameter_source("case_folder", folder_path)
        first_case = case_dicts[0]
        self.template_edit.setText(str(ini_files[0]))

        all_keys = sorted({k for d in case_dicts for k in d.keys()})
        loaded_items = [(k, first_case.get(k, "")) for k in all_keys]
        self._load_parameter_items(loaded_items, reset_user_parameters=True)

        non_numeric_varied = 0
        for key in all_keys:
            values = []
            for d in case_dicts:
                if key in d:
                    values.append(d[key])
            unique_values = sorted(set(values))
            if len(unique_values) <= 1:
                if unique_values:
                    self.constant_parameters.append(ConstantParameter(name=key, value=unique_values[0]))
                continue

            numeric_values: List[float] = []
            numeric_ok = True
            for v in unique_values:
                try:
                    numeric_values.append(float(v))
                except ValueError:
                    numeric_ok = False
                    break

            if numeric_ok:
                numeric_values = sorted(set(numeric_values))
                if len(numeric_values) >= 2:
                    deltas = [b - a for a, b in zip(numeric_values[:-1], numeric_values[1:])]
                    positive_deltas = [d for d in deltas if d > 1e-12]
                    delta = min(positive_deltas) if positive_deltas else 1.0
                else:
                    delta = 1.0
                self.sweep_parameters.append(
                    SweepParameter(
                        name=key,
                        min_value=min(numeric_values),
                        max_value=max(numeric_values),
                        delta=delta,
                        include_in_name=True,
                        abbreviation=self._default_abbreviation(key),
                    )
                )
            else:
                non_numeric_varied += 1
                self.constant_parameters.append(ConstantParameter(name=key, value=unique_values[0]))

        self._commit_parameter_change(
            enforce_sync=True,
            refresh_sweep_table=True,
            refresh_constant_table=True,
            refresh_sweep_editor=True,
            refresh_const_editor=True,
        )
        self._log(
            f"[import-folder] folder={folder_path}, files={len(case_dicts)}, "
            f"sweep_params={len(self.sweep_parameters)}, constants={len(self.constant_parameters)}, "
            f"non_numeric_varied={non_numeric_varied}"
        )

    def _read_ini_dict(self, ini_path: Path) -> Dict[str, str]:
        cp = new_config_parser()
        read_ok = cp.read(str(ini_path))
        if not read_ok:
            return {}
        data: Dict[str, str] = {}
        for section in cp.sections():
            for key in cp[section].keys():
                data[f"{section}.{key}"] = cp[section][key]
        return data

    def on_browse_input_out_dir(self) -> None:
        start_dir = self.input_out_edit.text() or str(Path.cwd())
        path = QFileDialog.getExistingDirectory(self, "Select Generated Input Directory", start_dir)
        if path:
            self.input_out_edit.setText(path)

    def on_browse_results_dir(self) -> None:
        start_dir = self.results_out_edit.text() or str(Path.cwd())
        path = QFileDialog.getExistingDirectory(self, "Select Results Output Directory", start_dir)
        if path:
            self.results_out_edit.setText(path)
            self.on_results_dir_changed()

    def on_results_dir_changed(self) -> None:
        self._collected_case_rows = []
        self._collected_numeric_columns = []
        self._last_exported_excel = None
        self.output_last_excel_label.setText("No Excel export yet.")
        self.on_refresh_outputs()

    def on_refresh_all(self) -> None:
        self._refresh_parameter_views(
            refresh_loaded_params=True,
            refresh_parameter_selectors=True,
            refresh_sweep_table=True,
            refresh_constant_table=True,
            refresh_sweep_editor=True,
            refresh_const_editor=True,
            clear_generated_cases=False,
        )
        self.on_refresh_outputs()
        self._refresh_log_views(refresh_current=True)
        self._log("[refresh] Refreshed all GUI views.")

    def on_refresh_outputs(self) -> None:
        self._refresh_output_views(
            refresh_cases=True,
            refresh_plot_metrics=True,
        )

    def on_output_case_changed(self, _: int = 0) -> None:
        self._refresh_output_views(refresh_files=True)

    def on_output_file_changed(self, _: int = 0) -> None:
        self._refresh_output_views(refresh_preview=True)

    def on_output_search_changed(self, _: str = "") -> None:
        self._refresh_output_views(
            refresh_cases=True,
            refresh_plot_metrics=True,
        )

    def on_output_hold_parameter_changed(self, _: int = 0) -> None:
        self._refresh_output_views(refresh_hold_values=True)

    def on_update_selected_notebook(self) -> None:
        notebook_path = self._selected_output_file_path()
        if notebook_path is None or notebook_path.suffix.lower() != ".ipynb":
            self._show_warning("Select a .ipynb file in the Output tab first.")
            return

        try:
            nb = json.loads(notebook_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._show_warning(f"Failed to read notebook:\n{exc}")
            return

        cells = nb.get("cells")
        if not isinstance(cells, list):
            self._show_warning("Invalid notebook format: missing cell list.")
            return

        excel_path = str(self._last_exported_excel) if self._last_exported_excel else "<exported_excel_path>.xlsx"
        x_metric = self._selected_metric_key(self.output_plot_x_combo) or "<x_column>"
        y_metric = self._selected_metric_key(self.output_plot_y_combo) or "<y_column>"

        marker = "HFCAD GUI Auto Analysis"
        markdown_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## HFCAD GUI Auto Analysis\n",
                f"- Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                f"- Excel data: `{excel_path}`\n",
                f"- Suggested axes: `{x_metric}` vs `{y_metric}`\n",
            ],
        }
        code_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# {marker}\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                f'excel_path = r"{excel_path}"\n',
                f'x_col = "{x_metric}"\n',
                f'y_col = "{y_metric}"\n',
                "df = pd.read_excel(excel_path)\n",
                "display(df.head())\n",
                "if x_col in df.columns and y_col in df.columns:\n",
                "    df.plot.scatter(x=x_col, y=y_col, figsize=(8, 5), grid=True)\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "else:\n",
                "    print('Select valid x/y columns in GUI and update notebook again.')\n",
            ],
        }

        replace_idx = None
        for i, cell in enumerate(cells):
            source = "".join(cell.get("source", [])) if isinstance(cell, dict) else ""
            if marker in source:
                replace_idx = i
                break

        if replace_idx is None:
            cells.extend([markdown_cell, code_cell])
        else:
            cells[replace_idx] = code_cell
            md_idx = replace_idx - 1
            if (
                md_idx >= 0
                and isinstance(cells[md_idx], dict)
                and "HFCAD GUI Auto Analysis" in "".join(cells[md_idx].get("source", []))
            ):
                cells[md_idx] = markdown_cell
            else:
                cells.insert(replace_idx, markdown_cell)

        try:
            notebook_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as exc:
            self._show_warning(f"Failed to write notebook:\n{exc}")
            return

        self._log(f"[output-notebook] updated notebook: {notebook_path}")
        self._refresh_output_views(refresh_preview=True)

    def on_export_all_case_data_to_excel(self) -> None:
        if not OPENPYXL_AVAILABLE or Workbook is None:
            self._show_warning("openpyxl is not available. Install it to export Excel files.")
            return

        out_dir = Path(self.results_out_edit.text().strip()).expanduser()
        if not out_dir.exists() or not out_dir.is_dir():
            self._show_warning(f"Results output directory does not exist:\n{out_dir}")
            return

        rows, numeric_columns = self._collect_all_case_data_rows(out_dir)
        if not rows:
            self._show_warning("No case output directories found to export.")
            return
        if len(numeric_columns) <= 1:
            self._show_warning("No numeric fields were extracted from case outputs.")
            return

        excel_path = out_dir / f"all_case_data_{time.strftime('%Y%m%d-%H%M%S')}.xlsx"
        try:
            self._write_rows_to_excel(rows, excel_path, sheet_name="all_case_data")
        except Exception as exc:
            self._show_warning(f"Failed to export Excel file:\n{exc}")
            return

        self._collected_case_rows = rows
        self._collected_numeric_columns = numeric_columns
        self._last_exported_excel = excel_path
        self.output_last_excel_label.setText(str(excel_path))
        self._refresh_output_views(refresh_plot_metrics=True)
        self._log(
            f"[output-export] exported {len(rows)} case rows with "
            f"{len(numeric_columns)} numeric columns: {excel_path}"
        )

    def on_export_txt_metrics_to_excel(self) -> None:
        # Backward-compatible alias for previous button/action wiring.
        self.on_export_all_case_data_to_excel()

    def on_draw_collected_plot(self) -> None:
        if not MATPLOTLIB_AVAILABLE or self.output_plot_canvas is None or self.output_plot_figure is None:
            self._show_warning("Matplotlib plotting backend is not available.")
            return
        if not self._collected_case_rows:
            self._show_warning("No collected case data. Export all case data first.")
            return

        x_col = self._selected_metric_key(self.output_plot_x_combo)
        y_col = self._selected_metric_key(self.output_plot_y_combo)
        if not x_col or not y_col:
            self._show_warning("Select X and Y columns for plotting.")
            return

        hold_filters: List[Tuple[str, float]] = []
        for hold_idx, (hold_param_combo, hold_value_combo) in enumerate(self.output_hold_pairs):
            hold_col = self._selected_metric_key(hold_param_combo)
            if not hold_col:
                continue
            hold_value_data = hold_value_combo.currentData()
            if not isinstance(hold_value_data, (int, float)):
                self._show_warning(f"Select a numeric hold value for Hold {hold_idx + 1}.")
                return
            hold_filters.append((hold_col, float(hold_value_data)))

        points = []
        for row in self._collected_case_rows:
            hold_matched = True
            for hold_col, hold_value in hold_filters:
                hold_row_value = row.get(hold_col)
                if not isinstance(hold_row_value, (int, float)):
                    hold_matched = False
                    break
                tol = max(1e-9, 1e-6 * max(1.0, abs(hold_value)))
                if abs(float(hold_row_value) - float(hold_value)) > tol:
                    hold_matched = False
                    break
            if not hold_matched:
                continue

            x_val = row.get(x_col)
            y_val = row.get(y_col)
            if isinstance(x_val, (int, float)) and isinstance(y_val, (int, float)):
                points.append((float(x_val), float(y_val), str(row.get("case_name", ""))))

        if not points:
            self._show_warning("No numeric data points found for the selected x/y columns.")
            return

        plot_mode = self.output_plot_type_combo.currentText().strip().lower()
        points.sort(key=lambda p: p[0])
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]
        labels = [p[2] for p in points]

        self.output_plot_figure.clear()
        ax = self.output_plot_figure.add_subplot(111)
        if plot_mode.startswith("function"):
            grouped: Dict[float, List[float]] = {}
            for x_val, y_val, _ in points:
                grouped.setdefault(x_val, []).append(y_val)

            if len(grouped) < 2:
                self._show_warning(
                    f"Selected X parameter '{x_col}' has fewer than two unique values.\n"
                    "Choose another X parameter for function analysis."
                )
                return

            unique_x = sorted(grouped.keys())
            y_mean = [sum(grouped[x]) / len(grouped[x]) for x in unique_x]
            ax.plot(unique_x, y_mean, marker="o", linestyle="-", label="mean")

            if "minmax" in plot_mode:
                y_min = [min(grouped[x]) for x in unique_x]
                y_max = [max(grouped[x]) for x in unique_x]
                ax.fill_between(unique_x, y_min, y_max, alpha=0.20, label="min-max")

            ax.set_title(
                f"{y_col} = f({x_col}) [{len(unique_x)} unique X, {len(points)} cases]"
            )
            if len(unique_x) <= 20:
                for x_val, y_val in zip(unique_x, y_mean):
                    ax.annotate(f"{y_val:.3g}", (x_val, y_val), fontsize=7, alpha=0.85)
            if "minmax" in plot_mode:
                ax.legend(loc="best")
        else:
            if plot_mode == "line(raw)":
                ax.plot(x_values, y_values, marker="o", linestyle="-")
            else:
                ax.scatter(x_values, y_values)
            ax.set_title(f"{y_col} vs {x_col} [{len(points)} cases, raw]")
            if len(points) <= 20:
                for x_val, y_val, label in zip(x_values, y_values, labels):
                    ax.annotate(label, (x_val, y_val), fontsize=7, alpha=0.85)

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.25)
        self.output_plot_figure.tight_layout()
        self.output_plot_canvas.draw_idle()
        hold_text = ""
        if hold_filters:
            hold_parts = [f"{col}={fmt_float_for_ini(value)}" for col, value in hold_filters]
            hold_text = f", holds={'; '.join(hold_parts)}"
        self._log(f"[output-graph] plotted {len(points)} points: y={y_col}, x={x_col}, mode={plot_mode}{hold_text}")

    def _selected_output_file_path(self) -> Optional[Path]:
        file_data = self.output_file_combo.currentData()
        if not file_data:
            return None
        return Path(str(file_data)).expanduser()

    def _refresh_output_views(
        self,
        *,
        refresh_cases: bool = False,
        refresh_files: bool = False,
        refresh_preview: bool = False,
        refresh_plot_metrics: bool = False,
        refresh_hold_values: bool = False,
        preferred_file_path: str = "",
    ) -> None:
        if refresh_cases:
            refresh_files = True
        if refresh_files:
            refresh_preview = True
        if refresh_plot_metrics:
            refresh_hold_values = True

        if refresh_cases:
            self._refresh_output_cases(
                preferred_file_path=preferred_file_path,
                refresh_files=refresh_files,
                refresh_preview=refresh_preview,
            )
        elif refresh_files:
            self._refresh_output_files(
                preferred_file_path=preferred_file_path,
                refresh_preview=refresh_preview,
            )
        elif refresh_preview:
            self._refresh_output_preview()

        if refresh_plot_metrics:
            self._refresh_plot_metric_combos(refresh_hold_values=refresh_hold_values)
        elif refresh_hold_values:
            self._refresh_hold_value_combo()

    def _refresh_output_cases(
        self,
        preferred_file_path: str = "",
        *,
        refresh_files: bool = True,
        refresh_preview: bool = True,
    ) -> None:
        out_dir = Path(self.results_out_edit.text().strip()).expanduser()
        self.output_results_dir_label.setText(str(out_dir))
        current_case_dir = str(self.output_case_combo.currentData() or "")
        current_file_path = preferred_file_path or str(self.output_file_combo.currentData() or "")
        query = self._output_search_query()

        self.output_case_combo.blockSignals(True)
        self.output_file_combo.blockSignals(True)
        self.output_case_combo.clear()
        self.output_file_combo.clear()
        self.output_case_combo.blockSignals(False)
        self.output_file_combo.blockSignals(False)

        if not out_dir.exists() or not out_dir.is_dir():
            self.output_file_path_label.setText("No output file selected.")
            self._show_output_text_preview(f"Results output directory does not exist:\n{out_dir}")
            return

        case_dirs = sorted((p for p in out_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
        filtered_case_dirs = [p for p in case_dirs if not query or self._fuzzy_match_text(query, p.name)]
        if query and not filtered_case_dirs and self._query_matches_output_metric(query):
            # Keep case browsing available when the query is targeting graph parameters.
            filtered_case_dirs = case_dirs
        if not filtered_case_dirs:
            self.output_file_path_label.setText("No output file selected.")
            if query:
                self._show_output_text_preview(
                    f"No case output directories match search '{query}' under:\n{out_dir}"
                )
            else:
                self._show_output_text_preview(f"No case output directories found under:\n{out_dir}")
            return

        self.output_case_combo.blockSignals(True)
        for case_dir in filtered_case_dirs:
            self.output_case_combo.addItem(case_dir.name, str(case_dir))

        target_idx = 0
        if current_case_dir:
            idx = self.output_case_combo.findData(current_case_dir)
            if idx >= 0:
                target_idx = idx
        self.output_case_combo.setCurrentIndex(target_idx)
        self.output_case_combo.blockSignals(False)
        if refresh_files:
            self._refresh_output_files(
                preferred_file_path=current_file_path,
                refresh_preview=refresh_preview,
            )

    def _refresh_output_files(self, preferred_file_path: str = "", *, refresh_preview: bool = True) -> None:
        case_dir_data = self.output_case_combo.currentData()
        case_dir = Path(str(case_dir_data)).expanduser() if case_dir_data else None
        query = self._output_search_query()
        self.output_file_combo.blockSignals(True)
        self.output_file_combo.clear()
        self.output_file_combo.blockSignals(False)

        if case_dir is None or not case_dir.exists() or not case_dir.is_dir():
            self.output_file_path_label.setText("No output file selected.")
            self._show_output_text_preview("No output case selected.")
            return

        files = sorted((p for p in case_dir.rglob("*") if p.is_file()), key=lambda p: str(p.relative_to(case_dir)))
        filtered_files = files
        if query:
            filtered_files = [
                p
                for p in files
                if self._fuzzy_match_text(query, p.name)
                or self._fuzzy_match_text(query, str(p.relative_to(case_dir)))
            ]
            if not filtered_files and self._query_matches_output_metric(query):
                # Keep file browsing available when the query is targeting graph parameters.
                filtered_files = files
        if not filtered_files:
            self.output_file_path_label.setText("No output file selected.")
            if query:
                self._show_output_text_preview(
                    f"No files match search '{query}' in output case directory:\n{case_dir}"
                )
            else:
                self._show_output_text_preview(f"No files found in output case directory:\n{case_dir}")
            return

        success_artifact = self.success_artifact_edit.text().strip()
        default_idx = 0

        self.output_file_combo.blockSignals(True)
        for i, file_path in enumerate(filtered_files):
            rel_path = str(file_path.relative_to(case_dir))
            self.output_file_combo.addItem(rel_path, str(file_path))
            if preferred_file_path and str(file_path) == preferred_file_path:
                default_idx = i
            if success_artifact and file_path.name == success_artifact:
                default_idx = i
        self.output_file_combo.setCurrentIndex(default_idx)
        self.output_file_combo.blockSignals(False)
        if refresh_preview:
            self._refresh_output_preview()

    def _refresh_output_preview(self) -> None:
        file_data = self.output_file_combo.currentData()
        if not file_data:
            self.output_file_path_label.setText("No output file selected.")
            self.output_text.clear()
            return

        file_path = Path(str(file_data)).expanduser()
        self.output_file_path_label.setText(str(file_path))
        if not file_path.exists() or not file_path.is_file():
            self._show_output_text_preview(f"Selected output file does not exist:\n{file_path}")
            return

        suffix = file_path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}:
            self._show_output_image_preview(file_path)
            return

        self._show_output_text_preview(self._read_output_preview_text(file_path))

    def _show_output_text_preview(self, text: str) -> None:
        self._output_full_pixmap = None
        self.output_image_label.clear()
        self.output_preview_stack.setCurrentIndex(OUTPUT_PREVIEW_TEXT)
        self.output_text.setPlainText(text)

    def _show_output_image_preview(self, image_path: Path) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self._show_output_text_preview(f"Could not load image:\n{image_path}")
            return
        self._output_full_pixmap = pixmap
        self.output_preview_stack.setCurrentIndex(OUTPUT_PREVIEW_IMAGE)
        self._update_output_image_display()

    def _update_output_image_display(self) -> None:
        if self._output_full_pixmap is None:
            return
        if not hasattr(self, "output_image_scroll"):
            return
        viewport_size = self.output_image_scroll.viewport().size()
        if viewport_size.width() <= 2 or viewport_size.height() <= 2:
            return
        scaled = self._output_full_pixmap.scaled(
            viewport_size,
            KEEP_ASPECT_RATIO,
            SMOOTH_TRANSFORMATION,
        )
        self.output_image_label.setPixmap(scaled)
        self.output_image_label.resize(scaled.size())

    def _refresh_plot_metric_combos(self, *, refresh_hold_values: bool = True) -> None:
        ordered_metrics = self._filtered_output_metrics()
        current_x = self._selected_metric_key(self.output_plot_x_combo)
        current_y = self._selected_metric_key(self.output_plot_y_combo)
        current_holds = [self._selected_metric_key(param_combo) for param_combo, _ in self.output_hold_pairs]

        self.output_plot_x_combo.blockSignals(True)
        self.output_plot_y_combo.blockSignals(True)
        for hold_param_combo, _ in self.output_hold_pairs:
            hold_param_combo.blockSignals(True)
        self.output_plot_x_combo.clear()
        self.output_plot_y_combo.clear()
        for hold_param_combo, _ in self.output_hold_pairs:
            hold_param_combo.clear()
            hold_param_combo.addItem("(No Hold)", "")

        if not ordered_metrics:
            self.output_plot_x_combo.blockSignals(False)
            self.output_plot_y_combo.blockSignals(False)
            for hold_param_combo, _ in self.output_hold_pairs:
                hold_param_combo.blockSignals(False)
            if refresh_hold_values:
                self._refresh_hold_value_combo()
            return

        for metric in ordered_metrics:
            self.output_plot_x_combo.addItem(metric, metric)
            self.output_plot_y_combo.addItem(metric, metric)
            for hold_param_combo, _ in self.output_hold_pairs:
                hold_param_combo.addItem(metric, metric)

        x_idx = self._find_combo_data_index(self.output_plot_x_combo, current_x)
        y_idx = self._find_combo_data_index(self.output_plot_y_combo, current_y)
        self.output_plot_x_combo.setCurrentIndex(x_idx if x_idx >= 0 else 0)
        self.output_plot_y_combo.setCurrentIndex(y_idx if y_idx >= 0 else (1 if len(ordered_metrics) > 1 else 0))
        for hold_idx, (hold_param_combo, _) in enumerate(self.output_hold_pairs):
            current_hold = current_holds[hold_idx] if hold_idx < len(current_holds) else ""
            h_idx = self._find_combo_data_index(hold_param_combo, current_hold)
            hold_param_combo.setCurrentIndex(h_idx if h_idx >= 0 else 0)

        self.output_plot_x_combo.blockSignals(False)
        self.output_plot_y_combo.blockSignals(False)
        for hold_param_combo, _ in self.output_hold_pairs:
            hold_param_combo.blockSignals(False)
        if refresh_hold_values:
            self._refresh_hold_value_combo()

    def _ordered_plot_metrics(self) -> List[str]:
        metric_set = set(self._collected_numeric_columns)
        if "case_index" not in metric_set:
            metric_set.add("case_index")
        if not metric_set:
            return []

        sweep_first: List[str] = []
        seen = set()
        for sp in self.sweep_parameters:
            metric = self._resolve_sweep_metric_column(sp, metric_set)
            if metric and metric not in seen:
                sweep_first.append(metric)
                seen.add(metric)

        remaining = sorted(m for m in metric_set if m not in seen)
        return sweep_first + remaining

    def _output_search_query(self) -> str:
        return self.output_search_edit.text().strip().lower()

    def _filtered_output_metrics(self) -> List[str]:
        ordered_metrics = self._ordered_plot_metrics()
        query = self._output_search_query()
        if not query:
            return ordered_metrics

        matched_metrics = [metric for metric in ordered_metrics if self._fuzzy_match_text(query, metric)]
        # Keep current behavior for non-metric searches (case/file focused).
        return matched_metrics if matched_metrics else ordered_metrics

    def _query_matches_output_metric(self, query: str) -> bool:
        if not query:
            return False
        return any(self._fuzzy_match_text(query, metric) for metric in self._ordered_plot_metrics())

    def _resolve_sweep_metric_column(self, sweep: SweepParameter, metric_set: set[str]) -> Optional[str]:
        try:
            section, key = split_parameter(sweep.name)
        except ValueError:
            section, key = "", sweep.name

        candidates: List[str] = []
        if section:
            candidates.append(
                f"ini.case.{self._safe_metric_name(section)}.{self._safe_metric_name(key)}"
            )
        if sweep.abbreviation.strip():
            candidates.append(f"case_token.{self._safe_metric_name(sweep.abbreviation)}")
        candidates.append(f"case_token.{self._safe_metric_name(key)}")

        for cand in candidates:
            if cand in metric_set:
                return cand
        return None

    @staticmethod
    def _find_combo_data_index(combo: QComboBox, data_value: str) -> int:
        if not data_value:
            return -1
        for i in range(combo.count()):
            if str(combo.itemData(i) or "") == data_value:
                return i
        return -1

    @staticmethod
    def _selected_metric_key(combo: QComboBox) -> str:
        data = combo.currentData()
        if data is None:
            return combo.currentText().strip()
        return str(data).strip()

    def _refresh_hold_value_combo(self) -> None:
        for hold_param_combo, hold_value_combo in self.output_hold_pairs:
            hold_col = self._selected_metric_key(hold_param_combo)
            prev_value = hold_value_combo.currentData()

            hold_value_combo.blockSignals(True)
            hold_value_combo.clear()

            if not hold_col:
                hold_value_combo.setEnabled(False)
                hold_value_combo.addItem("(All)", None)
                hold_value_combo.blockSignals(False)
                continue

            unique_values = sorted(
                {
                    float(row[hold_col])
                    for row in self._collected_case_rows
                    if hold_col in row and isinstance(row.get(hold_col), (int, float))
                }
            )
            if not unique_values:
                hold_value_combo.setEnabled(False)
                hold_value_combo.addItem("(No Values)", None)
                hold_value_combo.blockSignals(False)
                continue

            hold_value_combo.setEnabled(True)
            for value in unique_values:
                hold_value_combo.addItem(fmt_float_for_ini(value), value)

            target_idx = -1
            if isinstance(prev_value, (int, float)):
                for i, value in enumerate(unique_values):
                    if abs(float(value) - float(prev_value)) <= max(1e-9, 1e-6 * max(1.0, abs(float(value)))):
                        target_idx = i
                        break
            hold_value_combo.setCurrentIndex(target_idx if target_idx >= 0 else 0)
            hold_value_combo.blockSignals(False)

    def _collect_all_case_data_rows(self, out_dir: Path) -> Tuple[List[Dict[str, float | int | str]], List[str]]:
        rows: List[Dict[str, float | int | str]] = []
        numeric_columns: set[str] = {"case_index"}
        case_dirs = sorted((p for p in out_dir.iterdir() if p.is_dir()), key=lambda p: p.name)

        for case_index, case_dir in enumerate(case_dirs):
            row: Dict[str, float | int | str] = {
                "case_name": case_dir.name,
                "case_dir": str(case_dir),
                "case_index": case_index,
            }
            row["file_count"] = sum(1 for p in case_dir.rglob("*") if p.is_file())
            numeric_columns.add("file_count")

            for tok_name, tok_value in self._extract_case_name_numeric_tokens(case_dir.name).items():
                col_name = f"case_token.{tok_name}"
                row[col_name] = tok_value
                numeric_columns.add(col_name)

            ini_files = sorted(case_dir.rglob("*.ini"))
            for ini_path in ini_files:
                prefix = self._metric_file_prefix(case_dir, ini_path)
                parsed_ini = self._extract_ini_metrics(ini_path)
                for key, value in parsed_ini.items():
                    col_name = f"ini.{prefix}.{key}" if prefix else f"ini.{key}"
                    row[col_name] = value
                    if self._is_numeric_scalar(value):
                        numeric_columns.add(col_name)

            txt_files = sorted(case_dir.rglob("*.txt"))
            for txt_path in txt_files:
                prefix = self._metric_file_prefix(case_dir, txt_path)
                try:
                    text = txt_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                parsed_txt = self._extract_numeric_metrics_from_text(text)
                for key, value in parsed_txt.items():
                    col_name = f"txt.{prefix}.{key}" if prefix else f"txt.{key}"
                    row[col_name] = value
                    numeric_columns.add(col_name)

            csv_files = sorted(case_dir.rglob("*.csv"))
            for csv_path in csv_files:
                prefix = self._metric_file_prefix(case_dir, csv_path)
                parsed_csv = self._extract_csv_numeric_metrics(csv_path)
                for key, value in parsed_csv.items():
                    col_name = f"csv.{prefix}.{key}" if prefix else f"csv.{key}"
                    row[col_name] = value
                    if self._is_numeric_scalar(value):
                        numeric_columns.add(col_name)

            rows.append(row)

        return rows, sorted(numeric_columns)

    @staticmethod
    def _safe_metric_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()

    @classmethod
    def _metric_file_prefix(cls, case_dir: Path, file_path: Path) -> str:
        rel = file_path.relative_to(case_dir)
        rel_no_suffix = rel.with_suffix("")
        parts = list(rel_no_suffix.parts)
        if not parts:
            return ""

        normalized_parts: List[str] = []
        safe_case = cls._safe_metric_name(case_dir.name)
        for i, part in enumerate(parts):
            safe_part = cls._safe_metric_name(part)
            if i == len(parts) - 1:
                if safe_part in {safe_case, f"input_{safe_case}"}:
                    safe_part = "case"
                elif safe_case and safe_case in safe_part:
                    replaced = safe_part.replace(safe_case, "case").strip("_")
                    safe_part = replaced if replaced else "case"
            if safe_part:
                normalized_parts.append(safe_part)
        return ".".join(normalized_parts)

    @classmethod
    def _extract_numeric_metrics_from_text(cls, text: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        number_re = r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?"
        key_value_re = re.compile(
            rf"([A-Za-z][A-Za-z0-9_/\-\(\)\[\] .%]*?)\s*(?:=|:)\s*({number_re})"
        )
        phase_row_re = re.compile(
            rf"^([A-Za-z_]+)\s+({number_re})\s+({number_re})\s+({number_re})\s+({number_re})\s+"
            rf"({number_re})\s+({number_re})\s+({number_re})\s*$"
        )

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            normalized = re.sub(r"(?<=\d),(?=\d)", "", line)

            m_phase = phase_row_re.match(normalized)
            if m_phase:
                phase = cls._safe_metric_name(m_phase.group(1))
                keys = [
                    "time_min",
                    "total_kw",
                    "fc_stack_kw",
                    "battery_kw",
                    "compressor_kw",
                    "cooling_kw",
                    "mdot_h2_g_s",
                ]
                for i, key in enumerate(keys, start=2):
                    try:
                        metrics[f"phase.{phase}.{key}"] = float(m_phase.group(i))
                    except ValueError:
                        pass
                continue

            for m in key_value_re.finditer(normalized):
                key = cls._safe_metric_name(m.group(1))
                try:
                    metrics[key] = float(m.group(2))
                except ValueError:
                    continue

            if ":" in normalized:
                lhs, rhs = normalized.split(":", 1)
                prefix = cls._safe_metric_name(lhs)
                for chunk in rhs.split(","):
                    chunk = chunk.strip()
                    sub_match = re.match(rf"([A-Za-z][A-Za-z0-9_/\-\(\)\[\] .%]*)\s+({number_re})", chunk)
                    if not sub_match:
                        continue
                    sub_key = cls._safe_metric_name(sub_match.group(1))
                    try:
                        sub_val = float(sub_match.group(2))
                    except ValueError:
                        continue
                    metrics[f"{prefix}.{sub_key}"] = sub_val

        return metrics

    @staticmethod
    def _is_numeric_scalar(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _parse_value(value: str) -> float | str:
        raw = value.strip()
        if raw == "":
            return ""
        normalized = re.sub(r"(?<=\d),(?=\d)", "", raw)
        try:
            return float(normalized)
        except ValueError:
            return raw

    @classmethod
    def _extract_ini_metrics(cls, ini_path: Path) -> Dict[str, float | str]:
        cp = new_config_parser()
        read_ok = cp.read(str(ini_path))
        if not read_ok:
            return {}

        metrics: Dict[str, float | str] = {}
        for section in cp.sections():
            section_key = cls._safe_metric_name(section)
            for key, value in cp[section].items():
                safe_key = cls._safe_metric_name(key)
                metrics[f"{section_key}.{safe_key}"] = cls._parse_value(value)
        return metrics

    @classmethod
    def _extract_csv_numeric_metrics(cls, csv_path: Path) -> Dict[str, float | int]:
        metrics: Dict[str, float | int] = {}
        try:
            with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = list(reader.fieldnames or [])
        except Exception:
            return metrics

        metrics["row_count"] = len(rows)
        if not rows or not fieldnames:
            return metrics

        for field in fieldnames:
            safe_field = cls._safe_metric_name(field)
            if not safe_field:
                continue
            series: List[float] = []
            for row in rows:
                raw = (row.get(field) or "").strip()
                parsed = cls._parse_value(raw)
                if isinstance(parsed, (int, float)):
                    series.append(float(parsed))

            if not series:
                continue
            if len(series) == 1:
                metrics[safe_field] = series[0]
            else:
                metrics[f"{safe_field}.mean"] = sum(series) / len(series)
                metrics[f"{safe_field}.min"] = min(series)
                metrics[f"{safe_field}.max"] = max(series)
                metrics[f"{safe_field}.last"] = series[-1]

        return metrics

    @classmethod
    def _extract_case_name_numeric_tokens(cls, case_name: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for token in case_name.split("_"):
            match = re.fullmatch(r"([A-Za-z]+)([-+]?\d+(?:\.\d+)?)", token.strip())
            if not match:
                continue
            name = cls._safe_metric_name(match.group(1))
            if not name:
                continue
            try:
                metrics[name] = float(match.group(2))
            except ValueError:
                continue
        return metrics

    @staticmethod
    def _write_rows_to_excel(
        rows: Sequence[Dict[str, float | int | str]],
        excel_path: Path,
        *,
        sheet_name: str = "all_case_data",
    ) -> None:
        if Workbook is None:
            raise RuntimeError("openpyxl Workbook is not available")
        column_set = set()
        for row in rows:
            column_set.update(row.keys())

        preferred = ["case_name", "case_dir", "case_index"]
        tail_cols = sorted(c for c in column_set if c not in preferred)
        columns = [c for c in preferred if c in column_set] + tail_cols

        wb = Workbook()
        ws = wb.active
        ws.title = sheet_name[:31] if sheet_name else "all_case_data"
        ws.append(columns)
        for row in rows:
            ws.append([row.get(col, "") for col in columns])
        ws.freeze_panes = "A2"
        wb.save(excel_path)

    @staticmethod
    def _read_output_preview_text(file_path: Path, max_bytes: int = 400_000) -> str:
        try:
            file_size = file_path.stat().st_size
            with file_path.open("rb") as f:
                raw = f.read(max_bytes + 1)
        except Exception as exc:
            return f"[output-read-error] {exc}"

        if file_size == 0:
            return "[empty file]"

        if b"\x00" in raw[:4096]:
            return (
                "[binary-file] Preview is only available for text files.\n"
                f"Path: {file_path}\n"
                f"Size: {file_size} bytes"
            )

        text = raw[:max_bytes].decode("utf-8", errors="replace")
        if file_size > max_bytes or len(raw) > max_bytes:
            text += (
                f"\n\n[preview-truncated] showing first {max_bytes} bytes of "
                f"{file_size} bytes."
            )
        return text

    def on_browse_log_root_dir(self) -> None:
        start_dir = self.log_root_edit.text() or str(Path.cwd())
        path = QFileDialog.getExistingDirectory(self, "Select Run Log Root Directory", start_dir)
        if path:
            self.log_root_edit.setText(path)

    def on_browse_main_script(self) -> None:
        start_dir = str(Path(self.main_script_edit.text()).expanduser().parent)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Main Python Script",
            start_dir,
            "Python Files (*.py);;All Files (*)",
        )
        if path:
            self.main_script_edit.setText(path)

    def on_load_parameters(self) -> None:
        template_path = Path(self.template_edit.text()).expanduser()
        if not template_path.exists():
            self._show_warning("Template INI file not found.")
            return

        cp = new_config_parser()
        read_ok = cp.read(str(template_path))
        if not read_ok:
            self._show_warning("Could not read template INI file.")
            return

        options: List[str] = []
        loaded_items: List[Tuple[str, str]] = []
        for section in cp.sections():
            for key in cp[section].keys():
                name = f"{section}.{key}"
                options.append(name)
                loaded_items.append((name, cp[section][key]))

        options = sorted(options)
        source_changed = self._set_parameter_source("template", template_path)
        self._load_parameter_items(loaded_items, reset_user_parameters=source_changed)

        if source_changed:
            self._log(
                f"[template] Loaded {len(options)} parameters from {template_path} "
                "(new source selected; sweep/constants reset)"
            )
        else:
            self._log(f"[template] Reloaded {len(options)} parameters from {template_path}")

    def _refresh_loaded_params_table(self) -> None:
        query = self._filter_query()
        if query:
            self.visible_loaded_parameter_items = [
                (name, value)
                for name, value in self.loaded_parameter_items
                if self._fuzzy_match_text(query, name) or self._fuzzy_match_text(query, value)
            ]
        else:
            self.visible_loaded_parameter_items = list(self.loaded_parameter_items)

        self.loaded_params_table.setRowCount(len(self.visible_loaded_parameter_items))
        for i, (name, value) in enumerate(self.visible_loaded_parameter_items):
            self.loaded_params_table.setItem(i, 0, QTableWidgetItem(name))
            self.loaded_params_table.setItem(i, 1, QTableWidgetItem(value))

    def _selected_loaded_items(self) -> List[Tuple[str, str]]:
        rows = sorted({idx.row() for idx in self.loaded_params_table.selectionModel().selectedRows()})
        items: List[Tuple[str, str]] = []
        for row in rows:
            if 0 <= row < len(self.visible_loaded_parameter_items):
                items.append(self.visible_loaded_parameter_items[row])
        return items

    def on_add_loaded_selection_to_sweep(self) -> None:
        selected = self._selected_loaded_items()
        if not selected:
            self._show_warning("Select one or more loaded parameters first.")
            return

        added = 0
        skipped_existing = 0
        skipped_nonnumeric = 0
        for name, value in selected:
            if any(sp.name == name for sp in self.sweep_parameters):
                skipped_existing += 1
                continue
            try:
                numeric_value = float(value)
            except ValueError:
                skipped_nonnumeric += 1
                continue

            self.sweep_parameters.append(
                SweepParameter(
                    name=name,
                    min_value=numeric_value,
                    max_value=numeric_value,
                    delta=1.0,
                    include_in_name=True,
                    abbreviation=self._default_abbreviation(name),
                )
            )
            added += 1

        self._commit_parameter_change(
            enforce_sync=True,
            log_sync_notes=True,
            refresh_sweep_table=True,
            refresh_constant_table=True,
            refresh_const_editor=True,
        )
        self._log(
            f"[loaded->sweep] added={added}, skipped_existing={skipped_existing}, "
            f"skipped_nonnumeric={skipped_nonnumeric}"
        )

    def on_add_loaded_selection_to_constants(self) -> None:
        selected = self._selected_loaded_items()
        if not selected:
            self._show_warning("Select one or more loaded parameters first.")
            return

        added_or_updated = 0
        skipped_sweep = 0
        for name, value in selected:
            if any(sp.name == name for sp in self.sweep_parameters):
                skipped_sweep += 1
                continue

            idx = next((i for i, cp in enumerate(self.constant_parameters) if cp.name == name), None)
            if idx is None:
                self.constant_parameters.append(ConstantParameter(name=name, value=value))
            else:
                self.constant_parameters[idx] = ConstantParameter(name=name, value=value)
            added_or_updated += 1

        self._commit_parameter_change(
            refresh_constant_table=True,
            refresh_const_editor=True,
        )
        self._log(f"[loaded->constants] added_or_updated={added_or_updated}, skipped_sweep={skipped_sweep}")

    def on_add_sweep_parameter(self) -> None:
        param = self.sweep_param_combo.currentText().strip()
        if not param:
            self._show_warning("Select a parameter for sweep.")
            return

        if any(sp.name == param for sp in self.sweep_parameters):
            self._show_warning("This sweep parameter already exists.")
            return

        try:
            min_value = float(self.sweep_min_edit.text().strip())
            max_value = float(self.sweep_max_edit.text().strip())
            delta = float(self.sweep_delta_edit.text().strip())
        except ValueError:
            self._show_warning("min, max, and delta must be numeric values.")
            return

        if delta <= 0:
            self._show_warning("delta must be greater than zero.")
            return

        abbreviation = self.sweep_abbrev_edit.text().strip()
        if not abbreviation:
            abbreviation = self._default_abbreviation(param)

        include_flag = self.sweep_include_checkbox.isChecked()
        link_group = self.sweep_link_edit.text().strip()
        self.sweep_parameters.append(
            SweepParameter(
                name=param,
                min_value=min_value,
                max_value=max_value,
                delta=delta,
                include_in_name=include_flag,
                abbreviation=abbreviation,
                link_group=link_group,
            )
        )

        self.sweep_abbrev_edit.setText(self._default_abbreviation(param))
        self._commit_parameter_change(
            enforce_sync=True,
            log_sync_notes=True,
            refresh_sweep_table=True,
            refresh_constant_table=True,
            refresh_const_editor=True,
        )

    def on_sweep_parameter_changed(self) -> None:
        param = self.sweep_param_combo.currentText().strip()
        if param:
            self.sweep_abbrev_edit.setText(self._default_abbreviation(param))

    @staticmethod
    def _float_equal(a: float, b: float, tol: float = 1e-12) -> bool:
        return abs(a - b) <= tol

    def _enforce_cruise_altitude_sweep_sync(self) -> List[str]:
        """Mirror climb/turn altitude sweeps from cruise altitude when cruise sweep exists."""
        notes: List[str] = []
        cruise = next((sp for sp in self.sweep_parameters if sp.name == CRUISE_ALT_SWEEP_PARAM), None)
        if cruise is None:
            return notes

        sync_link_group = cruise.link_group.strip() or CRUISE_ALT_SYNC_LINK_GROUP
        if cruise.link_group != sync_link_group:
            cruise.link_group = sync_link_group
            notes.append(
                f"{CRUISE_ALT_SWEEP_PARAM}: set link_group='{sync_link_group}' for synchronized altitude sweep."
            )

        valid_names = set(self.parameter_options)
        for dep_name in SYNCED_ALT_SWEEP_PARAMS:
            if valid_names and dep_name not in valid_names:
                continue
            dep = next((sp for sp in self.sweep_parameters if sp.name == dep_name), None)
            if dep is None:
                self.sweep_parameters.append(
                    SweepParameter(
                        name=dep_name,
                        min_value=cruise.min_value,
                        max_value=cruise.max_value,
                        delta=cruise.delta,
                        include_in_name=False,
                        abbreviation=self._default_abbreviation(dep_name),
                        link_group=sync_link_group,
                    )
                )
                notes.append(f"{dep_name}: added and synchronized to {CRUISE_ALT_SWEEP_PARAM}.")
                continue

            dep_changed = False
            if not self._float_equal(dep.min_value, cruise.min_value):
                dep.min_value = cruise.min_value
                dep_changed = True
            if not self._float_equal(dep.max_value, cruise.max_value):
                dep.max_value = cruise.max_value
                dep_changed = True
            if not self._float_equal(dep.delta, cruise.delta):
                dep.delta = cruise.delta
                dep_changed = True
            if dep.link_group != sync_link_group:
                dep.link_group = sync_link_group
                dep_changed = True
            if dep.include_in_name:
                dep.include_in_name = False
                dep_changed = True

            if dep_changed:
                notes.append(f"{dep_name}: updated to match {CRUISE_ALT_SWEEP_PARAM}.")

        removed_constants = [cp.name for cp in self.constant_parameters if cp.name in SYNCED_ALT_SWEEP_PARAMS]
        if removed_constants:
            self.constant_parameters = [
                cp for cp in self.constant_parameters if cp.name not in SYNCED_ALT_SWEEP_PARAMS
            ]
            removed_preview = ", ".join(sorted(removed_constants))
            notes.append(
                f"removed conflicting constant override(s): {removed_preview} "
                f"(now synchronized with {CRUISE_ALT_SWEEP_PARAM})."
            )

        return notes

    def _selected_sweep_rows(self) -> List[int]:
        rows = sorted({idx.row() for idx in self.sweep_table.selectionModel().selectedRows()})
        if rows:
            return [r for r in rows if 0 <= r < len(self.sweep_parameters)]
        row = self.sweep_table.currentRow()
        if 0 <= row < len(self.sweep_parameters):
            return [row]
        return []

    def _reorder_sweep_parameters_by_link_group(self) -> None:
        """Place actively linked groups first and sort those groups by group name."""
        group_counts: Dict[str, int] = {}
        for sp in self.sweep_parameters:
            group_name = sp.link_group.strip()
            if group_name:
                group_counts[group_name] = group_counts.get(group_name, 0) + 1

        indexed = list(enumerate(self.sweep_parameters))

        def order_key(item: Tuple[int, SweepParameter]) -> Tuple[int, str, str, int]:
            original_idx, sp = item
            group_name = sp.link_group.strip()
            if group_name and group_counts.get(group_name, 0) >= 2:
                return (0, group_name.lower(), group_name, original_idx)
            return (1, "", "", original_idx)

        indexed.sort(key=order_key)
        self.sweep_parameters = [sp for _, sp in indexed]

    @staticmethod
    def _table_item_text(table: QTableWidget, row: int, col: int) -> str:
        widget = table.cellWidget(row, col)
        if isinstance(widget, QComboBox):
            return widget.currentText().strip()
        item = table.item(row, col)
        return item.text().strip() if item is not None else ""

    def _next_auto_link_group(self) -> str:
        max_group_num = 0
        for sp in self.sweep_parameters:
            group = sp.link_group.strip()
            match = re.fullmatch(r"grp(\d+)", group, re.IGNORECASE)
            if match:
                max_group_num = max(max_group_num, int(match.group(1)))
        return f"grp{max_group_num + 1}"

    def on_auto_link_selected_sweeps(self) -> None:
        rows = self._selected_sweep_rows()
        if len(rows) < 2:
            self._show_warning("Select at least two sweep parameter rows to auto-link.")
            return

        link_group = self._next_auto_link_group()
        for row in rows:
            self.sweep_parameters[row].link_group = link_group

        self.sweep_link_edit.setText(link_group)
        self._commit_parameter_change(
            enforce_sync=True,
            log_sync_notes=True,
            refresh_sweep_table=True,
            refresh_constant_table=True,
            refresh_const_editor=True,
        )
        self._log(f"[sweep-link] auto assigned {link_group} to {len(rows)} parameter(s)")

    def on_set_sweep_link_group(self) -> None:
        link_group = self.sweep_link_edit.text().strip()
        if not link_group:
            self._show_warning("Enter a link-group name (for example: grp1).")
            return
        rows = self._selected_sweep_rows()
        if not rows:
            self._show_warning("Select one or more sweep parameter rows to link.")
            return
        for row in rows:
            self.sweep_parameters[row].link_group = link_group
        self._commit_parameter_change(
            enforce_sync=True,
            log_sync_notes=True,
            refresh_sweep_table=True,
            refresh_constant_table=True,
            refresh_const_editor=True,
        )
        self._log(f"[sweep-link] set link_group={link_group} for {len(rows)} parameter(s)")

    def on_clear_sweep_link_group(self) -> None:
        rows = self._selected_sweep_rows()
        if not rows:
            self._show_warning("Select one or more sweep parameter rows to clear link.")
            return
        cleared = 0
        for row in rows:
            if self.sweep_parameters[row].link_group:
                self.sweep_parameters[row].link_group = ""
                cleared += 1
        self._commit_parameter_change(
            enforce_sync=True,
            log_sync_notes=True,
            refresh_sweep_table=True,
            refresh_constant_table=True,
            refresh_const_editor=True,
        )
        self._log(f"[sweep-link] cleared link group for {cleared} parameter(s)")

    def on_remove_sweep_parameter(self) -> None:
        row = self.sweep_table.currentRow()
        if row < 0 or row >= len(self.sweep_parameters):
            self._show_warning("Select a sweep parameter row to remove.")
            return
        del self.sweep_parameters[row]
        self._commit_parameter_change(
            enforce_sync=True,
            log_sync_notes=True,
            refresh_sweep_table=True,
            refresh_constant_table=True,
            refresh_const_editor=True,
        )

    def on_sweep_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_sweep_table:
            return

        row = item.row()
        col = item.column()
        if row < 0 or row >= len(self.sweep_parameters):
            return

        sp = self.sweep_parameters[row]
        value_text = item.text().strip()

        try:
            if col == 1:
                sp.min_value = float(value_text)
            elif col == 2:
                sp.max_value = float(value_text)
            elif col == 3:
                delta = float(value_text)
                if delta <= 0:
                    raise ValueError("delta must be greater than zero")
                sp.delta = delta
            elif col == 4:
                include = parse_bool_text(value_text)
                if include is None:
                    raise ValueError("Use Yes/No (or True/False) for In Case Name")
                sp.include_in_name = include
            elif col == 5:
                sp.abbreviation = value_text
            elif col == 6:
                sp.link_group = value_text
            else:
                self._refresh_sweep_table()
                return
        except ValueError as exc:
            self._show_warning(f"Invalid sweep value for {sp.name}: {exc}")
            self._refresh_sweep_table()
            return

        self._commit_parameter_change(
            enforce_sync=True,
            log_sync_notes=True,
            refresh_sweep_table=True,
            refresh_constant_table=True,
            refresh_const_editor=True,
        )

    def on_sweep_table_include_changed(self, param_name: str, value_text: str) -> None:
        if self._updating_sweep_table:
            return

        include = parse_bool_text(value_text)
        if include is None:
            self._show_warning(f"Invalid In Case Name value for {param_name}: {value_text}")
            self._refresh_sweep_table()
            return

        updated = False
        for idx, sp in enumerate(self.sweep_parameters):
            if sp.name == param_name and sp.include_in_name != include:
                self.sweep_parameters[idx].include_in_name = include
                updated = True
                break

        if not updated:
            return

        self._commit_parameter_change(
            enforce_sync=True,
            log_sync_notes=True,
            refresh_sweep_table=True,
            refresh_constant_table=True,
            refresh_const_editor=True,
        )

    def _refresh_sweep_table(self) -> None:
        current_row = self.sweep_table.currentRow()
        current_col = self.sweep_table.currentColumn()
        current_name = ""
        if 0 <= current_row < len(self.sweep_parameters):
            current_name = self.sweep_parameters[current_row].name

        self._reorder_sweep_parameters_by_link_group()

        self._updating_sweep_table = True
        self.sweep_table.blockSignals(True)
        try:
            self.sweep_table.setRowCount(len(self.sweep_parameters))
            for i, sp in enumerate(self.sweep_parameters):
                name_item = QTableWidgetItem(sp.name)
                name_item.setFlags(name_item.flags() & ~ITEM_IS_EDITABLE)
                self.sweep_table.setItem(i, 0, name_item)
                self.sweep_table.setItem(i, 1, QTableWidgetItem(fmt_float_for_ini(sp.min_value)))
                self.sweep_table.setItem(i, 2, QTableWidgetItem(fmt_float_for_ini(sp.max_value)))
                self.sweep_table.setItem(i, 3, QTableWidgetItem(fmt_float_for_ini(sp.delta)))
                include_item = QTableWidgetItem("Yes" if sp.include_in_name else "No")
                include_item.setFlags(include_item.flags() & ~ITEM_IS_EDITABLE)
                self.sweep_table.setItem(i, 4, include_item)
                include_combo = NoWheelComboBox()
                include_combo.addItems(["Yes", "No"])
                include_combo.setCurrentText("Yes" if sp.include_in_name else "No")
                include_combo.currentTextChanged.connect(
                    lambda text, param_name=sp.name: self.on_sweep_table_include_changed(param_name, text)
                )
                self.sweep_table.setCellWidget(i, 4, include_combo)
                self.sweep_table.setItem(i, 5, QTableWidgetItem(sp.abbreviation))
                self.sweep_table.setItem(i, 6, QTableWidgetItem(sp.link_group))
        finally:
            self.sweep_table.blockSignals(False)
            self._updating_sweep_table = False

        target_row = -1
        if current_name:
            target_row = next(
                (idx for idx, sp in enumerate(self.sweep_parameters) if sp.name == current_name),
                -1,
            )
        elif 0 <= current_row < self.sweep_table.rowCount():
            target_row = current_row

        target_col = current_col if 0 <= current_col < self.sweep_table.columnCount() else 0
        if 0 <= target_row < self.sweep_table.rowCount():
            self.sweep_table.setCurrentCell(target_row, target_col)
        elif self.sweep_table.rowCount() > 0:
            fallback_row = min(max(current_row, 0), self.sweep_table.rowCount() - 1)
            self.sweep_table.setCurrentCell(fallback_row, target_col)
        self._refresh_output_views(refresh_plot_metrics=True)

    def _loaded_value_for_parameter(self, param_name: str) -> str:
        for name, value in self.loaded_parameter_items:
            if name == param_name:
                return value.strip()
        return ""

    def _existing_constant_value(self, param_name: str) -> str:
        for cp in self.constant_parameters:
            if cp.name == param_name:
                return cp.value.strip()
        return ""

    def _normalize_choice_value(self, value: str, options: Sequence[str]) -> str:
        text = value.strip()
        if not text:
            return ""

        bool_value = parse_bool_text(text)
        if bool_value is not None and "True" in options and "False" in options:
            return "True" if bool_value else "False"

        lowered = text.lower()
        for opt in options:
            if opt.lower() == lowered:
                return opt
        return text

    def _parameter_value_options(self, param_name: str, current_value: str = "") -> List[str]:
        options = list(ENUM_CONFIG_OPTIONS.get(param_name, ()))
        # Infer bool options only from textual bool literals for unknown params.
        # This avoids misclassifying numeric values like "0" and "1" as booleans.
        if not options and (
            param_name in BOOL_CONFIG_PARAMS or parse_bool_text(current_value, allow_numeric=False) is not None
        ):
            options = ["True", "False"]
        if not options:
            return []

        normalized_current = self._normalize_choice_value(current_value, options)
        if normalized_current and normalized_current not in options:
            options.append(normalized_current)
        return options

    def _current_const_value_input(self) -> str:
        if self.const_value_stack.currentWidget() is self.const_value_combo:
            return self.const_value_combo.currentText().strip()
        return self.const_value_edit.text().strip()

    def on_const_parameter_changed(self) -> None:
        param = self.const_param_combo.currentText().strip()
        if not param:
            self.const_value_stack.setCurrentWidget(self.const_value_edit)
            return

        preferred = self._existing_constant_value(param)
        if not preferred:
            preferred = self._loaded_value_for_parameter(param)
        if not preferred:
            preferred = self.const_value_edit.text().strip()

        options = self._parameter_value_options(param, preferred)
        if options:
            target_value = self._normalize_choice_value(preferred, options)
            self.const_value_combo.blockSignals(True)
            self.const_value_combo.clear()
            self.const_value_combo.addItems(options)
            idx = self.const_value_combo.findText(target_value)
            self.const_value_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.const_value_combo.blockSignals(False)
            self.const_value_edit.setText(self.const_value_combo.currentText().strip())
            self.const_value_stack.setCurrentWidget(self.const_value_combo)
            return

        self.const_value_stack.setCurrentWidget(self.const_value_edit)
        if preferred:
            self.const_value_edit.setText(preferred)

    def on_const_table_combo_value_changed(self, param_name: str, new_value: str) -> None:
        if self._updating_const_table:
            return

        updated = False
        for idx, cp in enumerate(self.constant_parameters):
            if cp.name == param_name and cp.value != new_value:
                self.constant_parameters[idx] = ConstantParameter(name=cp.name, value=new_value)
                updated = True
                break

        if not updated:
            return

        refresh_editor = self.const_param_combo.currentText().strip() == param_name
        self._commit_parameter_change(refresh_const_editor=refresh_editor)

    def on_add_constant_parameter(self) -> None:
        param = self.const_param_combo.currentText().strip()
        value = self._current_const_value_input()
        if not param:
            self._show_warning("Select a constant parameter.")
            return
        if value == "":
            self._show_warning("Constant parameter value cannot be empty.")
            return

        if any(sp.name == param for sp in self.sweep_parameters):
            self._show_warning("This parameter is already used as a sweep parameter.")
            return

        idx = next((i for i, cp in enumerate(self.constant_parameters) if cp.name == param), None)
        if idx is None:
            self.constant_parameters.append(ConstantParameter(name=param, value=value))
        else:
            self.constant_parameters[idx] = ConstantParameter(name=param, value=value)

        self._commit_parameter_change(
            refresh_constant_table=True,
            refresh_const_editor=True,
        )

    def on_remove_constant_parameter(self) -> None:
        row = self.const_table.currentRow()
        if row < 0 or row >= len(self.constant_parameters):
            self._show_warning("Select a constant parameter row to remove.")
            return
        del self.constant_parameters[row]
        self._commit_parameter_change(
            refresh_constant_table=True,
            refresh_const_editor=True,
        )

    def on_const_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_const_table:
            return

        row = item.row()
        col = item.column()
        if row < 0 or row >= len(self.constant_parameters):
            return

        if col != 1:
            self._refresh_constant_table()
            return

        value = item.text().strip()
        if value == "":
            self._show_warning("Constant parameter value cannot be empty.")
            self._refresh_constant_table()
            return

        param_name = self.constant_parameters[row].name
        self.constant_parameters[row].value = value
        refresh_editor = self.const_param_combo.currentText().strip() == param_name
        self._commit_parameter_change(
            refresh_constant_table=True,
            refresh_const_editor=refresh_editor,
        )

    def _refresh_parameter_views(
        self,
        *,
        refresh_loaded_params: bool = False,
        refresh_parameter_selectors: bool = False,
        refresh_sweep_table: bool = False,
        refresh_constant_table: bool = False,
        refresh_sweep_editor: bool = False,
        refresh_const_editor: bool = False,
        clear_generated_cases: bool = True,
    ) -> None:
        if refresh_loaded_params:
            self._refresh_loaded_params_table()
        if refresh_parameter_selectors:
            self._refresh_parameter_selectors()
        if refresh_sweep_table:
            self._refresh_sweep_table()
        if refresh_constant_table:
            self._refresh_constant_table()
        self._on_parameters_changed(
            refresh_sweep_editor=refresh_sweep_editor,
            refresh_const_editor=refresh_const_editor,
            clear_generated_cases=clear_generated_cases,
        )

    def _commit_parameter_change(
        self,
        *,
        enforce_sync: bool = False,
        log_sync_notes: bool = False,
        refresh_loaded_params: bool = False,
        refresh_parameter_selectors: bool = False,
        refresh_sweep_table: bool = False,
        refresh_constant_table: bool = False,
        refresh_sweep_editor: bool = False,
        refresh_const_editor: bool = False,
        clear_generated_cases: bool = True,
    ) -> List[str]:
        sync_notes: List[str] = []
        if enforce_sync:
            sync_notes = self._enforce_cruise_altitude_sweep_sync()
        self._refresh_parameter_views(
            refresh_loaded_params=refresh_loaded_params,
            refresh_parameter_selectors=refresh_parameter_selectors,
            refresh_sweep_table=refresh_sweep_table,
            refresh_constant_table=refresh_constant_table,
            refresh_sweep_editor=refresh_sweep_editor,
            refresh_const_editor=refresh_const_editor,
            clear_generated_cases=clear_generated_cases,
        )
        if log_sync_notes:
            for note in sync_notes:
                self._log(f"[sweep-sync] {note}")
        return sync_notes

    def _on_parameters_changed(
        self,
        *,
        refresh_sweep_editor: bool = False,
        refresh_const_editor: bool = False,
        clear_generated_cases: bool = True,
    ) -> None:
        if clear_generated_cases and self.generated_cases:
            self.generated_cases = []
            self._log(
                "[params] parameter values changed; cleared generated-case cache. "
                "Regenerate input files before execute."
            )

        self._update_diff_summary()
        if refresh_sweep_editor:
            self.on_sweep_parameter_changed()
        if refresh_const_editor:
            self.on_const_parameter_changed()

    def on_update_params_from_tables(self, checked: bool = False, *, show_success: bool = True) -> bool:
        del checked
        self.sweep_table.clearFocus()
        self.const_table.clearFocus()
        QApplication.processEvents()

        errors: List[str] = []
        warnings: List[str] = []

        updated_sweeps: List[SweepParameter] = []
        sweep_pos: Dict[str, int] = {}
        for row in range(self.sweep_table.rowCount()):
            name = self._table_item_text(self.sweep_table, row, 0)
            if not name:
                warnings.append(f"Sweep row {row + 1}: empty parameter name row ignored.")
                continue

            min_text = self._table_item_text(self.sweep_table, row, 1)
            max_text = self._table_item_text(self.sweep_table, row, 2)
            delta_text = self._table_item_text(self.sweep_table, row, 3)
            include_text = self._table_item_text(self.sweep_table, row, 4)
            abbrev = self._table_item_text(self.sweep_table, row, 5)
            link_group = self._table_item_text(self.sweep_table, row, 6)

            try:
                min_value = float(min_text)
                max_value = float(max_text)
                delta = float(delta_text)
                if delta <= 0:
                    raise ValueError("delta must be greater than zero")
            except ValueError as exc:
                errors.append(f"Sweep row {row + 1} ({name}): {exc}")
                continue

            include_flag = parse_bool_text(include_text)
            if include_flag is None:
                errors.append(
                    f"Sweep row {row + 1} ({name}): invalid In Case Name value '{include_text}' "
                    "(use Yes/No, True/False, 1/0)."
                )
                continue

            if not abbrev:
                abbrev = self._default_abbreviation(name)

            sweep_item = SweepParameter(
                name=name,
                min_value=min_value,
                max_value=max_value,
                delta=delta,
                include_in_name=include_flag,
                abbreviation=abbrev,
                link_group=link_group,
            )
            if name in sweep_pos:
                updated_sweeps[sweep_pos[name]] = sweep_item
                warnings.append(
                    f"Sweep duplicate '{name}' detected; kept last occurrence (row {row + 1})."
                )
            else:
                sweep_pos[name] = len(updated_sweeps)
                updated_sweeps.append(sweep_item)

        updated_constants: List[ConstantParameter] = []
        const_pos: Dict[str, int] = {}
        for row in range(self.const_table.rowCount()):
            name = self._table_item_text(self.const_table, row, 0)
            value = self._table_item_text(self.const_table, row, 1)
            if not name:
                warnings.append(f"Constant row {row + 1}: empty parameter name row ignored.")
                continue
            if value == "":
                errors.append(f"Constant row {row + 1} ({name}): value cannot be empty.")
                continue

            const_item = ConstantParameter(name=name, value=value)
            if name in const_pos:
                updated_constants[const_pos[name]] = const_item
                warnings.append(
                    f"Constant duplicate '{name}' detected; kept last occurrence (row {row + 1})."
                )
            else:
                const_pos[name] = len(updated_constants)
                updated_constants.append(const_item)

        if errors:
            error_preview = "\n".join(f"- {msg}" for msg in errors[:8])
            if len(errors) > 8:
                error_preview += f"\n- ... and {len(errors) - 8} more"
            self._show_warning(
                "Could not update parameters due to invalid values:\n\n"
                f"{error_preview}"
            )
            return False

        sweep_names = {sp.name for sp in updated_sweeps}
        filtered_constants: List[ConstantParameter] = []
        conflict_names: List[str] = []
        for cp in updated_constants:
            if cp.name in sweep_names:
                conflict_names.append(cp.name)
            else:
                filtered_constants.append(cp)

        if conflict_names:
            preview = ", ".join(conflict_names[:8])
            suffix = f" (+{len(conflict_names) - 8} more)" if len(conflict_names) > 8 else ""
            warnings.append(
                f"Removed {len(conflict_names)} constant(s) that conflict with sweep parameters "
                f"(sweep kept): {preview}{suffix}."
            )

        self.sweep_parameters = updated_sweeps
        self.constant_parameters = filtered_constants
        warnings.extend(
            self._commit_parameter_change(
                enforce_sync=True,
                refresh_sweep_table=True,
                refresh_constant_table=True,
                refresh_const_editor=True,
            )
        )
        self._log(
            f"[params-update] sweep={len(self.sweep_parameters)}, constants={len(self.constant_parameters)}, "
            f"warnings={len(warnings)}"
        )
        for warning_msg in warnings:
            self._log(f"[params-update] {warning_msg}")

        if warnings:
            warning_preview = "\n".join(f"- {msg}" for msg in warnings[:8])
            if len(warnings) > 8:
                warning_preview += f"\n- ... and {len(warnings) - 8} more"
            self._show_warning(
                "Parameters updated with conflict handling:\n\n"
                f"{warning_preview}"
            )
        elif show_success:
            QMessageBox.information(
                self,
                "HFCAD GUI",
                f"Parameters updated.\n\n"
                f"Sweep parameters: {len(self.sweep_parameters)}\n"
                f"Constant parameters: {len(self.constant_parameters)}",
            )
        return True

    def on_add_all_constants_except_sweep(self) -> None:
        if not self.loaded_parameter_items:
            self._show_warning("No loaded parameters available. Load a template or import a case first.")
            return

        sweep_names = {sp.name for sp in self.sweep_parameters}
        existing_values = {cp.name: cp.value for cp in self.constant_parameters}

        new_constants: List[ConstantParameter] = []
        added = 0
        kept_existing = 0
        skipped_sweep = 0

        for name, loaded_value in self.loaded_parameter_items:
            if name in sweep_names:
                skipped_sweep += 1
                continue

            if name in existing_values:
                value = existing_values[name]
                kept_existing += 1
            else:
                value = loaded_value
                added += 1

            new_constants.append(ConstantParameter(name=name, value=value))

        if not new_constants:
            self._show_warning("No non-sweep parameters available to add as constants.")
            return

        self.constant_parameters = new_constants
        self._commit_parameter_change(
            refresh_constant_table=True,
            refresh_const_editor=True,
        )
        self._log(
            f"[constants-all] total={len(new_constants)}, added={added}, "
            f"kept_existing={kept_existing}, skipped_sweep={skipped_sweep}"
        )

    def _refresh_constant_table(self) -> None:
        current_row = self.const_table.currentRow()
        current_col = self.const_table.currentColumn()
        self._updating_const_table = True
        self.const_table.blockSignals(True)
        try:
            self.const_table.setRowCount(len(self.constant_parameters))
            for i, cp in enumerate(self.constant_parameters):
                name_item = QTableWidgetItem(cp.name)
                name_item.setFlags(name_item.flags() & ~ITEM_IS_EDITABLE)
                self.const_table.setItem(i, 0, name_item)
                options = self._parameter_value_options(cp.name, cp.value)
                if options:
                    value_item = QTableWidgetItem(cp.value)
                    value_item.setFlags(value_item.flags() & ~ITEM_IS_EDITABLE)
                    self.const_table.setItem(i, 1, value_item)

                    combo = NoWheelComboBox()
                    combo.addItems(options)
                    selected = self._normalize_choice_value(cp.value, options)
                    idx = combo.findText(selected)
                    combo.setCurrentIndex(idx if idx >= 0 else 0)
                    combo.currentTextChanged.connect(
                        lambda text, param_name=cp.name: self.on_const_table_combo_value_changed(param_name, text)
                    )
                    self.const_table.setCellWidget(i, 1, combo)
                else:
                    self.const_table.setCellWidget(i, 1, None)
                    self.const_table.setItem(i, 1, QTableWidgetItem(cp.value))
        finally:
            self.const_table.blockSignals(False)
            self._updating_const_table = False

        if 0 <= current_row < self.const_table.rowCount():
            target_col = current_col if 0 <= current_col < self.const_table.columnCount() else 0
            self.const_table.setCurrentCell(current_row, target_col)

    def on_make_input_files(self) -> None:
        if not self.on_update_params_from_tables(show_success=False):
            return

        template_path = Path(self.template_edit.text().strip()).expanduser()
        out_dir = Path(self.input_out_edit.text().strip()).expanduser()

        if not template_path.exists():
            self._show_warning(f"Template INI does not exist: {template_path}")
            return

        try:
            template_text = template_path.read_text(encoding="utf-8")
        except Exception as exc:
            self._show_warning(f"Could not read template file:\n{exc}")
            return
        template_check = new_config_parser()
        try:
            template_check.read_string(template_text)
        except configparser.Error as exc:
            self._show_warning(
                "Template INI has invalid or duplicate entries.\n\n"
                f"Details: {exc}"
            )
            return

        values_per_parameter: List[List[float]] = []
        for sp in self.sweep_parameters:
            try:
                values_per_parameter.append(build_series(sp.min_value, sp.max_value, sp.delta))
            except ValueError as exc:
                self._show_warning(f"Invalid sweep for {sp.name}: {exc}")
                return

        linked_indices: Dict[str, List[int]] = {}
        for idx, sp in enumerate(self.sweep_parameters):
            group = sp.link_group.strip()
            if group:
                linked_indices.setdefault(group, []).append(idx)

        active_linked_indices = {name: idxs for name, idxs in linked_indices.items() if len(idxs) >= 2}
        ignored_link_groups = sorted(name for name, idxs in linked_indices.items() if len(idxs) < 2)

        for group_name, idxs in active_linked_indices.items():
            point_counts = {len(values_per_parameter[idx]) for idx in idxs}
            if len(point_counts) != 1:
                details = ", ".join(
                    f"{self.sweep_parameters[idx].name}:{len(values_per_parameter[idx])}" for idx in idxs
                )
                self._show_warning(
                    f"Linked group '{group_name}' requires equal number of points.\n\n"
                    f"Current points: {details}\n"
                    "Adjust min/max/delta values so all linked parameters have matching point counts."
                )
                return

        sweep_dimensions = []
        emitted_link_groups = set()
        for idx, sp in enumerate(self.sweep_parameters):
            group = sp.link_group.strip()
            if group and group in active_linked_indices:
                if group in emitted_link_groups:
                    continue
                emitted_link_groups.add(group)
                group_points = len(values_per_parameter[active_linked_indices[group][0]])
                sweep_dimensions.append(("group", group, list(range(group_points))))
            else:
                sweep_dimensions.append(("single", idx, values_per_parameter[idx]))

        total_cases = 1
        for _, _, dimension_values in sweep_dimensions:
            total_cases *= len(dimension_values)

        if not self.sweep_parameters:
            self._log("[generate] no sweep parameters set; generating a single input file.")

        if total_cases > 50_000:
            self._show_warning(
                f"Requested sweep would generate {total_cases:,} cases.\n"
                "Reduce ranges/steps before generating."
            )
            return

        confirmed = self._confirm_action(
            "Confirm Input Generation",
            f"Make input files now?\n\n"
            f"Files to create: {total_cases:,}\n"
            f"Directory: {out_dir}",
        )
        if not confirmed:
            self._log("[generate] canceled by user before file creation.")
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        if not self._maybe_clear_directory_files(
            out_dir,
            title="Input Directory Has Existing Files",
            description="input INI",
            file_glob="*.ini",
            recursive=False,
            clear_all_contents=False,
        ):
            return
        self.generated_cases = []
        used_case_names: Dict[str, int] = {}

        if ignored_link_groups:
            self._log(
                "[sweep-link] ignored groups with fewer than 2 parameters: "
                + ", ".join(ignored_link_groups)
            )

        for selection in itertools.product(*(dim[2] for dim in sweep_dimensions)):
            combo_values: List[float] = [0.0] * len(self.sweep_parameters)
            for dim, chosen_value in zip(sweep_dimensions, selection):
                if dim[0] == "single":
                    sweep_idx = dim[1]
                    combo_values[sweep_idx] = float(chosen_value)
                    continue

                group_name = dim[1]
                point_idx = int(chosen_value)
                for sweep_idx in active_linked_indices[group_name]:
                    combo_values[sweep_idx] = values_per_parameter[sweep_idx][point_idx]

            combo = tuple(combo_values)
            cp = new_config_parser()
            try:
                cp.read_string(template_text)
            except configparser.Error as exc:
                self._show_warning(
                    "Template INI could not be applied while generating cases.\n\n"
                    f"Case context: {combo}\n"
                    f"Details: {exc}"
                )
                return

            case_diff: Dict[str, str] = {}
            for constant in self.constant_parameters:
                section, key = split_parameter(constant.name)
                if not cp.has_section(section):
                    cp.add_section(section)
                cp.set(section, key, constant.value)
                case_diff[constant.name] = constant.value

            for sp, value in zip(self.sweep_parameters, combo):
                section, key = split_parameter(sp.name)
                if not cp.has_section(section):
                    cp.add_section(section)
                value_str = fmt_float_for_ini(value)
                cp.set(section, key, value_str)
                case_diff[sp.name] = value_str

            case_name = self._build_case_name(combo, used_case_names)
            case_path = out_dir / f"{case_name}.ini"
            with case_path.open("w", encoding="utf-8") as f:
                cp.write(f)

            self.generated_cases.append((case_path, case_diff))

        strict_standards = self.strict_ini_check.isChecked()
        self._log(
            f"[generate-check] strict standards mode={'on' if strict_standards else 'off'}."
        )
        check_ok, error_count, warning_count, errors, warnings = self._validate_generated_ini_files(
            strict_standards=strict_standards
        )
        for message in warnings:
            self._log(f"[generate-check] {message}")
        if warning_count > len(warnings):
            self._log(f"[generate-check] ... and {warning_count - len(warnings)} more warning(s).")

        if not check_ok:
            for message in errors:
                self._log(f"[generate-check] {message}")
            if error_count > len(errors):
                self._log(f"[generate-check] ... and {error_count - len(errors)} more error(s).")
            self.generated_cases = []
            self._update_diff_summary()
            preview = "\n".join(f"- {msg}" for msg in errors[:8]) if errors else "- Unknown validation error."
            if error_count > 8:
                preview += f"\n- ... and {error_count - 8} more"
            self._show_warning(
                "Generated INI safety check failed.\n"
                "Generated case cache has been cleared; fix issues and regenerate.\n\n"
                f"{preview}"
            )
            return

        if warning_count:
            self._log(f"[generate-check] completed with {warning_count} warning(s).")
        else:
            self._log("[generate-check] all generated INI files passed consistency checks.")

        self._update_diff_summary()
        self._log(f"[generate] wrote {len(self.generated_cases)} input files to {out_dir}")

    def _build_case_name(self, values: Sequence[float], used_names: Dict[str, int]) -> str:
        prefix = self.case_prefix_edit.text().strip()
        tokens: List[str] = []

        for sp, value in zip(self.sweep_parameters, values):
            if not sp.include_in_name:
                continue
            abbr = sp.abbreviation.strip() or self._default_abbreviation(sp.name)
            tokens.append(f"{abbr}{fmt_float_for_case(value)}")

        if not tokens:
            tokens = ["case"]

        name = "_".join(tokens)
        if prefix:
            name = f"{prefix}_{name}"

        name = CASE_NAME_SAFE.sub("-", name).strip("._-")
        if not name:
            name = "case"

        base_name = name
        if base_name in used_names:
            used_names[base_name] += 1
            name = f"{base_name}_{used_names[base_name]}"
        else:
            used_names[base_name] = 0
        return name

    def _validate_parameter_value_standard(self, param_name: str, value: str) -> str:
        if value == "":
            return f"{param_name}: empty value."

        enum_options = list(ENUM_CONFIG_OPTIONS.get(param_name, ()))
        if enum_options:
            normalized = self._normalize_choice_value(value, enum_options)
            if normalized not in enum_options:
                options_preview = ", ".join(enum_options[:6])
                if len(enum_options) > 6:
                    options_preview += ", ..."
                return (
                    f"{param_name}: non-standard value '{value}' "
                    f"(allowed: {options_preview})."
                )

        if param_name in BOOL_CONFIG_PARAMS and parse_bool_text(value) is None:
            return f"{param_name}: non-standard boolean value '{value}'."

        return ""

    def _validate_generated_ini_files(
        self,
        *,
        strict_standards: bool = False,
        max_reported_issues: int = 20,
    ) -> Tuple[bool, int, int, List[str], List[str]]:
        error_count = 0
        warning_count = 0
        errors: List[str] = []
        warnings: List[str] = []

        def add_error(message: str) -> None:
            nonlocal error_count
            error_count += 1
            if len(errors) < max_reported_issues:
                errors.append(message)

        def add_warning(message: str) -> None:
            nonlocal warning_count
            warning_count += 1
            if len(warnings) < max_reported_issues:
                warnings.append(message)

        baseline_keys: Optional[set] = None
        baseline_case = ""
        for case_path, expected_diff in self.generated_cases:
            try:
                text = case_path.read_text(encoding="utf-8")
            except Exception as exc:
                add_error(f"{case_path.name}: could not read generated file ({exc}).")
                continue

            cp = new_config_parser()
            try:
                cp.read_string(text)
            except (configparser.DuplicateSectionError, configparser.DuplicateOptionError) as exc:
                add_error(f"{case_path.name}: duplicate section/parameter detected ({exc}).")
                continue
            except configparser.Error as exc:
                add_error(f"{case_path.name}: invalid INI format ({exc}).")
                continue

            actual: Dict[str, str] = {}
            lowered_to_name: Dict[str, str] = {}
            for section in cp.sections():
                if not section.strip():
                    add_warning(f"{case_path.name}: empty section name encountered.")
                for key in cp[section].keys():
                    param_name = f"{section}.{key}"
                    value = cp[section][key].strip()
                    actual[param_name] = value

                    lowered = param_name.lower()
                    prior = lowered_to_name.get(lowered)
                    if prior is not None and prior != param_name:
                        add_warning(
                            f"{case_path.name}: parameter names differ only by case "
                            f"('{prior}' vs '{param_name}')."
                        )
                    else:
                        lowered_to_name[lowered] = param_name

                    standard_issue = self._validate_parameter_value_standard(param_name, value)
                    if standard_issue:
                        message = f"{case_path.name}: {standard_issue}"
                        if strict_standards:
                            add_error(message)
                        else:
                            add_warning(message)

            keys = set(actual.keys())
            if baseline_keys is None:
                baseline_keys = keys
                baseline_case = case_path.name
            elif keys != baseline_keys:
                missing = sorted(baseline_keys - keys)
                extra = sorted(keys - baseline_keys)
                details: List[str] = []
                if missing:
                    miss_preview = ", ".join(missing[:4])
                    miss_suffix = " ..." if len(missing) > 4 else ""
                    details.append(f"missing [{miss_preview}{miss_suffix}]")
                if extra:
                    extra_preview = ", ".join(extra[:4])
                    extra_suffix = " ..." if len(extra) > 4 else ""
                    details.append(f"extra [{extra_preview}{extra_suffix}]")
                detail_text = "; ".join(details) if details else "key mismatch"
                add_error(
                    f"{case_path.name}: parameter set inconsistent with {baseline_case} ({detail_text})."
                )

            for param_name, expected_value in expected_diff.items():
                actual_value = actual.get(param_name)
                if actual_value is None:
                    add_error(f"{case_path.name}: missing expected parameter '{param_name}'.")
                    continue
                if actual_value != expected_value:
                    add_error(
                        f"{case_path.name}: '{param_name}' expected '{expected_value}' "
                        f"but found '{actual_value}'."
                    )

        return error_count == 0, error_count, warning_count, errors, warnings

    def _update_diff_summary(self) -> None:
        lines: List[str] = []
        lines.append(f"Generated cases: {len(self.generated_cases)}")
        lines.append(f"Output directory: {Path(self.input_out_edit.text().strip()).expanduser()}")
        lines.append("")
        lines.append("Sweep parameters:")
        for sp in self.sweep_parameters:
            values = build_series(sp.min_value, sp.max_value, sp.delta)
            link_desc = sp.link_group if sp.link_group else "-"
            lines.append(
                f"- {sp.name}: {fmt_float_for_ini(sp.min_value)} .. {fmt_float_for_ini(sp.max_value)} "
                f"(delta {fmt_float_for_ini(sp.delta)}, {len(values)} points), "
                f"case-name flag={'on' if sp.include_in_name else 'off'}, "
                f"abbrev={sp.abbreviation}, link={link_desc}"
            )

        if self.constant_parameters:
            lines.append("")
            lines.append("Constant overrides:")
            for cp in self.constant_parameters:
                lines.append(f"- {cp.name} = {cp.value}")

        lines.append("")
        lines.append("Case differences preview (first 25 cases):")
        for case_path, case_diff in self.generated_cases[:25]:
            diff_items = ", ".join(f"{k}={v}" for k, v in case_diff.items())
            lines.append(f"- {case_path.name}: {diff_items}")

        if len(self.generated_cases) > 25:
            lines.append(f"... ({len(self.generated_cases) - 25} more cases omitted)")

        self.diff_text.setPlainText("\n".join(lines))

    def on_execute_cases(self) -> None:
        if self.execution_worker is not None and self.execution_worker.isRunning():
            self._show_warning("A run is already in progress.")
            return

        input_dir = Path(self.input_out_edit.text().strip()).expanduser()
        if self.generated_cases:
            case_paths = [p for p, _ in self.generated_cases]
        else:
            case_paths = sorted(input_dir.glob("*.ini"))

        if not case_paths:
            self._show_warning("No input cases found. Generate cases first or point to a folder with .ini files.")
            return

        python_bin = self.python_edit.text().strip()
        main_script = Path(self.main_script_edit.text().strip()).expanduser()
        out_dir = Path(self.results_out_edit.text().strip()).expanduser()
        log_root = Path(self.log_root_edit.text().strip()).expanduser()

        if not python_bin:
            self._show_warning("Python executable is empty.")
            return
        if not main_script.exists():
            self._show_warning(f"Main script not found: {main_script}")
            return

        mode = self.mode_combo.currentText()
        jobs = self.jobs_spin.value()
        timeout_sec = self.timeout_spin.value()
        success_artifact = self.success_artifact_edit.text().strip()
        workers = jobs if mode == "parallel" else 1

        confirmed = self._confirm_action(
            "Confirm Case Execution",
            f"Execute cases now?\n\n"
            f"Total cases: {len(case_paths):,}\n"
            f"Run mode: {mode}\n"
            f"Worker number: {workers}",
        )
        if not confirmed:
            self._log("[start] execution canceled by user.")
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        if not self._maybe_clear_directory_files(
            out_dir,
            title="Output Directory Has Existing Files",
            description="result output",
            file_glob="*",
            recursive=True,
            clear_all_contents=True,
        ):
            return
        log_root.mkdir(parents=True, exist_ok=True)
        run_dir = log_root / f"run-{time.strftime('%Y%m%d-%H%M%S')}"

        self.completed_count = 0
        self.total_count = len(case_paths)
        self.progress_label.setText(f"Running 0/{self.total_count} cases...")
        self.execute_btn.setEnabled(False)
        self._set_stop_button_running_state(True)
        self._refresh_log_views(reset_for_new_run=True)
        self._log("")
        self._log(
            f"{'=' * 20} New Run {time.strftime('%Y-%m-%d %H:%M:%S')} {'=' * 20}"
        )
        self._log(
            f"[start] mode={mode}, jobs={jobs}, timeout={timeout_sec}s, "
            f"main={main_script}, out={out_dir}, log_root={log_root}"
        )

        self.execution_worker = ExecutionWorker(
            case_paths=case_paths,
            python_bin=python_bin,
            main_script=str(main_script),
            out_dir=out_dir,
            run_dir=run_dir,
            mode=mode,
            jobs=jobs,
            timeout_sec=timeout_sec,
            success_artifact=success_artifact,
        )
        self.execution_worker.log_signal.connect(self._log)
        self.execution_worker.case_done_signal.connect(self._on_case_done)
        self.execution_worker.summary_signal.connect(self._on_run_finished)
        self.execution_worker.start()

    def on_stop_run(self) -> None:
        if self.execution_worker is None or not self.execution_worker.isRunning():
            self._show_warning("No run is currently in progress.")
            return

        confirmed = self._confirm_action(
            "Stop Current Run",
            f"Stop the running job?\n\n"
            f"Progress: {self.completed_count}/{self.total_count} cases completed.",
        )
        if not confirmed:
            return

        self.execution_worker.request_stop()
        self.stop_btn.setEnabled(False)
        self.progress_label.setText(
            f"Stopping... {self.completed_count}/{self.total_count} cases completed."
        )

    def on_mode_changed(self) -> None:
        mode = self.mode_combo.currentText()
        self.jobs_spin.setEnabled(mode == "parallel")

    def on_log_view_changed(self, _: int = 0) -> None:
        self._refresh_log_views(refresh_current=True)

    def _on_case_done(self, case_name: str, is_ok: bool, return_code: int, log_path: str) -> None:
        self.completed_count += 1
        status = "OK" if is_ok else "FAIL"
        if self.execution_worker is not None and self.execution_worker.stop_requested:
            self.progress_label.setText(f"Stopping... {self.completed_count}/{self.total_count} cases...")
        else:
            self.progress_label.setText(f"Running {self.completed_count}/{self.total_count} cases...")
        self._log(f"[done] {case_name}: {status} (rc={return_code}) log={log_path}")
        self._refresh_log_views(register_case_name=case_name, register_log_path=log_path)

    def _on_run_finished(self, total: int, ok: int, fail: int, elapsed_s: float, summary_path: str) -> None:
        stopped = self.execution_worker.stop_requested if self.execution_worker is not None else False
        self.progress_label.setText(
            f"Run finished: total={total}, ok={ok}, fail={fail}, elapsed={elapsed_s:.1f}s"
        )
        self._log(f"[summary] total={total}, ok={ok}, fail={fail}, elapsed={elapsed_s:.1f}s")
        self._log(f"[summary] details written to {summary_path}")
        self.execute_btn.setEnabled(True)
        self._set_stop_button_running_state(False)
        self.execution_worker = None
        self.on_refresh_outputs()

        if self._close_after_stop_requested:
            self._close_after_stop_requested = False
            self.close()
            return

        if stopped:
            self._show_warning(
                f"Run was stopped by user.\n\nTotal configured: {total}\nOK: {ok}\nFAIL: {fail}\n\n"
                f"See summary: {summary_path}"
            )
        elif fail > 0:
            self._show_warning(
                f"Run completed with failures.\n\nTotal: {total}\nOK: {ok}\nFAIL: {fail}\n\n"
                f"See summary: {summary_path}"
            )
        else:
            QMessageBox.information(
                self,
                "Execution Complete",
                f"All cases completed.\n\nTotal: {total}\nOK: {ok}\nFAIL: {fail}\n\nSummary: {summary_path}",
            )

    def _default_abbreviation(self, param_name: str) -> str:
        _, key = split_parameter(param_name)
        if key in {"h_cr_m", "h_to_m", "climbalt_m", "cruisealt_m"}:
            return "h"
        if "mach" in key.lower():
            return "M"
        if "range" in key.lower():
            return "R"
        if "payload" in key.lower():
            return "PL"
        tokens = [tok for tok in re.split(r"[_\W]+", key) if tok]
        if len(tokens) == 1:
            return tokens[0][:3]
        return "".join(tok[0] for tok in tokens[:3]).upper()

    def _log(self, text: str) -> None:
        self._summary_log_lines.append(text)
        if self.log_view_combo.currentData() == LOG_SUMMARY_KEY:
            self.logs_text.appendPlainText(text)

    def _reset_log_views_for_new_run(self) -> None:
        self._summary_log_lines = []
        self._case_log_paths = {}
        self.log_view_combo.blockSignals(True)
        self.log_view_combo.clear()
        self.log_view_combo.addItem("Summary", LOG_SUMMARY_KEY)
        self.log_view_combo.setCurrentIndex(0)
        self.log_view_combo.blockSignals(False)
        self.logs_text.clear()

    def _refresh_log_views(
        self,
        *,
        reset_for_new_run: bool = False,
        register_case_name: str = "",
        register_log_path: str = "",
        refresh_current: bool = False,
        refresh_registered_case: bool = True,
    ) -> None:
        if reset_for_new_run:
            self._reset_log_views_for_new_run()

        if register_case_name:
            self._register_case_log(register_case_name, register_log_path)
            if refresh_registered_case and self.log_view_combo.currentData() == register_case_name:
                refresh_current = True

        if refresh_current:
            self._refresh_log_panel()

    def _register_case_log(self, case_name: str, log_path: str) -> None:
        log_file = Path(log_path).expanduser()
        if not log_path or not log_file.exists():
            return

        self._case_log_paths[case_name] = log_file
        if self.log_view_combo.findData(case_name) < 0:
            self.log_view_combo.addItem(case_name, case_name)

    def _refresh_log_panel(self) -> None:
        selected_key = self.log_view_combo.currentData()
        if selected_key == LOG_SUMMARY_KEY:
            self.logs_text.setPlainText("\n".join(self._summary_log_lines))
            return

        case_name = str(selected_key) if selected_key is not None else ""
        if not case_name:
            self.logs_text.clear()
            return

        log_path = self._case_log_paths.get(case_name)
        if log_path is None:
            self.logs_text.setPlainText(f"No log available for case: {case_name}")
            return

        try:
            content = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            self.logs_text.setPlainText(f"[log-read-error] {case_name}: {exc}")
            return

        self.logs_text.setPlainText(content)

    def _confirm_action(self, title: str, message: str) -> bool:
        reply = QMessageBox.question(
            self,
            title,
            message,
            MESSAGEBOX_YES | MESSAGEBOX_NO,
            MESSAGEBOX_NO,
        )
        return reply == MESSAGEBOX_YES

    def _maybe_clear_directory_files(
        self,
        directory: Path,
        *,
        title: str,
        description: str,
        file_glob: str,
        recursive: bool,
        clear_all_contents: bool,
    ) -> bool:
        if not directory.exists():
            return True

        file_iter = directory.rglob(file_glob) if recursive else directory.glob(file_glob)
        existing_files = sorted((p for p in file_iter if p.is_file()), key=lambda p: str(p))
        if not existing_files:
            return True

        preview = ", ".join(p.name for p in existing_files[:5])
        if len(existing_files) > 5:
            preview += ", ..."

        remove_confirmed = self._confirm_action(
            title,
            f"The selected directory already contains {len(existing_files):,} {description} file(s).\n\n"
            f"Directory: {directory}\n"
            f"Examples: {preview}\n\n"
            "Remove existing files before continuing?",
        )
        if not remove_confirmed:
            self._log(
                f"[cleanup] kept existing files in {directory}; proceeding without deleting old {description} files."
            )
            return True

        try:
            if clear_all_contents:
                for child in directory.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
            else:
                for file_path in existing_files:
                    file_path.unlink()
        except Exception as exc:
            self._show_warning(f"Failed to clean directory before continue:\n{directory}\n\n{exc}")
            return False

        removed_note = "directory contents" if clear_all_contents else f"{description} files"
        self._log(f"[cleanup] removed existing {removed_note} from {directory}")
        return True

    def _set_stop_button_running_state(self, running: bool) -> None:
        self.stop_btn.setEnabled(running)
        if running:
            self.stop_btn.setStyleSheet(
                "QPushButton { background-color: #c62828; color: #ffffff; font-weight: 600; }"
            )
        else:
            self.stop_btn.setStyleSheet(
                "QPushButton { background-color: #d3d3d3; color: #666666; }"
            )

    def _show_warning(self, message: str) -> None:
        QMessageBox.warning(self, "HFCAD GUI", message)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        worker = self.execution_worker
        if worker is None or not worker.isRunning():
            event.accept()
            return

        first_request = not self._close_after_stop_requested
        self._close_after_stop_requested = True
        worker.request_stop()
        self.stop_btn.setEnabled(False)
        self.progress_label.setText(
            f"Stopping before close... {self.completed_count}/{self.total_count} cases completed."
        )
        self._log("[shutdown] close requested while run is active; stopping worker.")

        if first_request:
            self._show_warning("Run stop requested. The window will close automatically after active cases stop.")
        event.ignore()


def main() -> int:
    app = QApplication(sys.argv)
    window = HFCADGui()
    window.show()
    return app_exec(app)


if __name__ == "__main__":
    raise SystemExit(main())
