"""
Virtual Makeup desktop app (PySide6).

    python app.py                 start on the webcam
    python app.py --image X.png   start on an image

The makeup engine lives in utils.py, this file is only the window around it:
a Processor thread owns the camera and the models, the UI pushes Settings into it
and paints the frames that come back.
"""
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
from PySide6.QtCore import QMutex, QMutexLocker, QRect, Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from utils import (
    FEATURES,
    FaceAnalyzer,
    adapt_style,
    bgr_to_hex,
    blur_background,
    camera_format,
    camera_is_known,
    camera_resolution,
    create_face_mesh,
    create_segmenter,
    detect_landmarks,
    hex_to_bgr,
    load_presets,
    open_camera,
    probe_camera,
    remember_camera_mode,
    matched_foundation_shade,
    apply_foundation,
    render_makeup,
)

ACCENT = "#e0567a"
CAMERA_SLOTS = 3  # "Webcam 0..2" entries in the source list, an opened image goes after them
# quality dropdown: label -> open_camera resolution argument
QUALITY_OPTIONS = {
    "Auto": "auto",
    "480p": "480p",
    "720p": "720p",
    "1080p": "1080p",
    "Max": "max",
    "Auto (detect again)": "detect",
}

STYLESHEET = """
QWidget { background: #17181c; color: #e8e8ea; font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 13px; }
QMainWindow, QStatusBar { background: #101114; }
QStatusBar { color: #8a8f98; }
QStatusBar::item { border: none; }
QLabel#title { font-size: 20px; font-weight: 600; }
QLabel#dim { color: #8a8f98; }
QGroupBox { border: 1px solid #2a2c33; border-radius: 8px; margin-top: 14px; padding: 10px 10px 6px 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9aa0ab; }
QPushButton { background: #2a2c33; border: 1px solid #363944; border-radius: 6px; padding: 6px 12px; }
QPushButton:hover { background: #343744; }
QPushButton:pressed { background: #1f2127; }
QPushButton#accent { background: #e0567a; border-color: #e0567a; color: white; font-weight: 600; padding: 8px 12px; }
QPushButton#accent:hover { background: #ea6b8b; }
QComboBox { background: #2a2c33; border: 1px solid #363944; border-radius: 6px; padding: 5px 8px; }
QComboBox QAbstractItemView { background: #2a2c33; border: 1px solid #363944; selection-background-color: #e0567a; }
QSlider::groove:horizontal { height: 4px; background: #2a2c33; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #e0567a; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; height: 14px; margin: -5px 0; background: #ffffff; border-radius: 7px; }
QSlider::sub-page:horizontal:disabled { background: #3a3d47; }
QSlider::handle:horizontal:disabled { background: #6a6e7a; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; border: 1px solid #4a4e5a; background: #2a2c33; }
QCheckBox::indicator:checked { background: #e0567a; border-color: #e0567a; }
QScrollArea { border: none; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #3a3d47; border-radius: 4px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: #4a4e5a; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QToolTip { background: #2a2c33; color: #e8e8ea; border: 1px solid #363944; }
QTableWidget { background: #101114; gridline-color: #2a2c33; border: 1px solid #2a2c33; border-radius: 6px; selection-background-color: #e0567a; selection-color: white; }
QHeaderView::section { background: #2a2c33; color: #9aa0ab; border: none; padding: 6px; font-weight: 600; }
QDialog { background: #17181c; }
"""


@dataclass(frozen=True)
class Settings:
    """everything the UI controls, handed to the Processor as one immutable snapshot"""

    style: dict  # {feature: {"color": (b, g, r), "alpha": 0..1, "enabled": bool}}
    adaptive: bool = True  # move the shades to the skin tone and the light
    foundation: bool = False
    foundation_shade: tuple | None = None  # (b, g, r), None matches the skin
    coverage: float = 0.5  # foundation colour evening 0..1
    smoothing: float = 0.4  # foundation texture smoothing 0..1
    blur_background: bool = False
    blur_strength: float = 0.05
    compare: bool = False


def split_view(before, after):
    """left half of the original next to the right half of the result, with a divider"""
    h, w = before.shape[:2]
    mid = w // 2
    out = after.copy()
    out[:, :mid] = before[:, :mid]
    cv2.line(out, (mid, 0), (mid, h), (255, 255, 255), 2)
    for text, x in (("Before", 12), ("After", mid + 12)):
        cv2.putText(out, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def to_qimage(bgr):
    h, w = bgr.shape[:2]
    # copy so the QImage owns its pixels once the numpy frame is gone
    return QImage(bgr.data, w, h, bgr.strides[0], QImage.Format.Format_BGR888).copy()


class Processor(QThread):
    """
    owns the camera and the mediapipe models and runs the whole pipeline off the UI thread.
    the UI pushes Settings and sources in, rendered frames come back through frame_ready.
    """

    frame_ready = Signal(QImage, bool)  # rendered frame, face found
    fps_changed = Signal(float)
    source_changed = Signal(str)  # human readable description of the active source
    failed = Signal(str)
    test_progress = Signal(str)  # webcam test: what is being measured right now
    test_finished = Signal(list)  # webcam test: rows from utils.probe_camera
    analysis_changed = Signal(str)  # what the shades are adapted to, empty when no face
    shade_matched = Signal(str)  # foundation shade matched to the skin, as #rrggbb

    def __init__(self, settings: Settings):
        super().__init__()
        self._lock = QMutex()
        self._settings = settings
        self._source = ("camera", (0, "auto"))  # ("camera", (index, resolution)) or ("image", ndarray)
        self._version = 0  # bumped on every settings / source change
        self._reopen = False  # the webcam must be opened again even if the source looks the same
        self._running = True
        self._last_error = None

    # ---- called from the UI thread
    def update_settings(self, settings: Settings):
        with QMutexLocker(self._lock):
            self._settings = settings
            self._version += 1

    def use_camera(self, index: int = 0, resolution: str = "auto", reopen: bool = False):
        """reopen forces a fresh open, needed when the remembered auto mode changed"""
        with QMutexLocker(self._lock):
            self._source = ("camera", (index, resolution))
            self._reopen = self._reopen or reopen
            self._version += 1

    def use_image(self, image):
        with QMutexLocker(self._lock):
            self._source = ("image", image)
            self._version += 1

    def test_camera(self, index: int = 0):
        """measure every mode of the webcam, then go back to it in auto mode"""
        with QMutexLocker(self._lock):
            self._source = ("test", index)
            self._version += 1

    def stop(self):
        self._running = False
        self.wait(3000)

    # ---- worker thread
    def run(self):
        face_video = create_face_mesh(static_image_mode=False)
        face_static = create_face_mesh(static_image_mode=True)
        segmenter = create_segmenter()
        analyzer = FaceAnalyzer()
        capture, capture_index = None, None
        image_id, image_landmarks, image_analysis = None, None, None
        rendered_version = -1
        last_description, last_shade = None, None
        frames, t_fps = 0, time.perf_counter()

        while self._running:
            with QMutexLocker(self._lock):
                settings, (kind, payload), version = self._settings, self._source, self._version
                reopen, self._reopen = self._reopen, False

            if kind == "test":
                # the test needs the webcam to itself
                if capture is not None:
                    capture.release()
                    capture, capture_index = None, None
                self.failed.emit(f"Testing webcam {payload}…")
                rows = probe_camera(payload, progress=self.test_progress.emit)
                self.test_finished.emit(rows)
                with QMutexLocker(self._lock):
                    if self._source == ("test", payload):
                        self._source = ("camera", (payload, "auto"))
                        self._version += 1
                continue

            if kind == "camera":
                index, resolution = payload
                if capture is None or capture_index != payload or reopen:
                    if capture is not None:
                        capture.release()
                    if resolution in ("detect", "max") or (resolution == "auto" and not camera_is_known(index)):
                        self.failed.emit(f"Measuring which modes webcam {index} runs smoothly, about 15 s, only once…")
                    else:
                        self.failed.emit(f"Opening webcam {index}…")
                    capture, capture_index = open_camera(index, resolution), payload
                    if capture is None:
                        self._report(f"Could not open webcam {index}")
                        self.msleep(500)
                        continue
                    width, height = camera_resolution(capture)
                    self.source_changed.emit(f"Webcam {index}  {width}x{height} {camera_format(capture)}")
                    analyzer.reset()
                ok, frame = capture.read()
                if not ok:
                    self._report(f"Webcam {index} stopped delivering frames")
                    capture.release()
                    capture = None
                    self.msleep(500)
                    continue
                frame = cv2.flip(frame, 1)
                landmarks = detect_landmarks(frame, face_video)
            else:
                if capture is not None:
                    capture.release()
                    capture, capture_index = None, None
                if rendered_version == version:
                    # a still image only needs re-rendering when something changed
                    self.msleep(15)
                    continue
                frame = payload
                if id(payload) != image_id:
                    image_landmarks, image_id = detect_landmarks(frame, face_static), id(payload)
                    image_analysis = analyzer.analyze(frame, image_landmarks) if image_landmarks is not None else None
                    self.source_changed.emit(f"Image  {frame.shape[1]}x{frame.shape[0]}")
                landmarks = image_landmarks
                rendered_version = version

            if landmarks is None:
                output = frame
                description = ""
            else:
                style = settings.style
                description = ""
                match_shade = settings.foundation and settings.foundation_shade is None
                analysis = None
                if settings.adaptive or match_shade:
                    analysis = analyzer.update(frame, landmarks) if kind == "camera" else image_analysis
                if settings.adaptive:
                    style = adapt_style(style, analysis)
                    description = analysis.describe()
                output = frame
                if settings.foundation:
                    shade = settings.foundation_shade or matched_foundation_shade(analysis)
                    output = apply_foundation(frame, landmarks, settings.coverage, settings.smoothing, shade, analysis)
                    if match_shade and bgr_to_hex(shade) != last_shade:
                        last_shade = bgr_to_hex(shade)
                        self.shade_matched.emit(last_shade)
                output = render_makeup(output, landmarks, style)
            if description != last_description:
                self.analysis_changed.emit(description)
                last_description = description
            if settings.blur_background:
                output = blur_background(output, segmenter, settings.blur_strength)
            if settings.compare:
                output = split_view(frame, output)
            self.frame_ready.emit(to_qimage(output), landmarks is not None)
            self._last_error = None

            if kind == "camera":
                frames += 1
                now = time.perf_counter()
                if now - t_fps >= 0.5:
                    self.fps_changed.emit(frames / (now - t_fps))
                    frames, t_fps = 0, now
            else:
                self.fps_changed.emit(0.0)

        if capture is not None:
            capture.release()
        face_video.close()
        face_static.close()
        segmenter.close()

    def _report(self, message):
        # only tell the UI once per distinct failure, the loop retries every half second
        if message != self._last_error:
            self._last_error = message
            self.failed.emit(message)



class VideoView(QWidget):
    """paints the latest frame scaled to fit, or a message when there is nothing to show"""

    def __init__(self):
        super().__init__()
        self._image = None
        self._message = "Starting webcam…"
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, image: QImage):
        self._image = image
        self.update()

    def set_message(self, text: str):
        self._image, self._message = None, text
        self.update()

    def image(self):
        return self._image

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0f1013"))
        if self._image is None:
            painter.setPen(QColor("#8a8f98"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
            return
        size = self._image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        target = QRect(0, 0, size.width(), size.height())
        target.moveCenter(self.rect().center())
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(target, self._image)


class FeatureRow(QWidget):
    """enable checkbox, colour swatch and intensity slider for one makeup feature"""

    changed = Signal()

    def __init__(self, label: str):
        super().__init__()
        self._color = (0, 0, 0)
        self.enabled = QCheckBox(label)
        self.swatch = QPushButton()
        self.swatch.setFixedSize(34, 24)
        self.swatch.setToolTip("Pick colour")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.value = QLabel("0%")
        self.value.setObjectName("dim")
        self.value.setFixedWidth(36)
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.enabled.setFixedWidth(96)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(8)
        row.addWidget(self.enabled)
        row.addWidget(self.slider, 1)
        row.addWidget(self.value)
        row.addWidget(self.swatch)

        self.enabled.toggled.connect(self._on_toggle)
        self.swatch.clicked.connect(self._pick_color)
        self.slider.valueChanged.connect(self._on_slider)

    def set_state(self, color, alpha: float, enabled: bool):
        """fill the controls without emitting changed"""
        for widget in (self.enabled, self.slider):
            widget.blockSignals(True)
        self.enabled.setChecked(enabled)
        self.slider.setValue(round(alpha * 100))
        self.slider.setEnabled(enabled)
        self.value.setText(f"{round(alpha * 100)}%")
        self._set_color(color)
        for widget in (self.enabled, self.slider):
            widget.blockSignals(False)

    def state(self) -> dict:
        return {"color": self._color, "alpha": self.slider.value() / 100, "enabled": self.enabled.isChecked()}

    def _set_color(self, color):
        self._color = tuple(int(c) for c in color)
        self.swatch.setStyleSheet(
            f"background: {bgr_to_hex(self._color)}; border: 1px solid #4a4e5a; border-radius: 6px;"
        )

    def _on_toggle(self, checked: bool):
        self.slider.setEnabled(checked)
        self.changed.emit()

    def _on_slider(self, value: int):
        self.value.setText(f"{value}%")
        self.changed.emit()

    def _pick_color(self):
        initial = QColor(bgr_to_hex(self._color))
        color = QColorDialog.getColor(initial, self, f"{self.enabled.text()} colour")
        if color.isValid():
            self._set_color(hex_to_bgr(color.name()))
            self.changed.emit()


class CameraTestDialog(QDialog):
    """
    runs utils.probe_camera through the Processor and shows every measured mode.
    the user can take the recommendation or pick any row.
    """

    COLUMNS = ("Mode", "Delivered", "Format", "Frame rate", "Verdict")

    def __init__(self, processor: Processor, index: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Webcam {index} test")
        self.resize(560, 360)
        self.processor = processor
        self.index = index
        self.rows = []
        self.changed = False

        self.status = QLabel("Measuring every mode, native and MJPG. Each switch takes a few seconds…")
        self.status.setWordWrap(True)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_selection)

        self.buttons = QDialogButtonBox()
        self.use_button = self.buttons.addButton("Use selected", QDialogButtonBox.ButtonRole.AcceptRole)
        self.use_button.setObjectName("accent")
        self.use_button.setEnabled(False)
        self.buttons.addButton(QDialogButtonBox.StandardButton.Close)
        self.buttons.accepted.connect(self._use_selected)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.buttons)

        processor.test_progress.connect(self.status.setText)
        processor.test_finished.connect(self._on_finished)
        processor.test_camera(index)

    def _on_finished(self, rows: list):
        self.rows = rows
        self.table.setRowCount(len(rows))
        recommended = None
        for i, row in enumerate(rows):
            verdict = "recommended" if row["recommended"] else "smooth" if row["smooth"] else "too slow"
            cells = (
                row["mode"],
                f"{row['size'][0]}x{row['size'][1]}",
                row["format"],
                f"{row['fps']:.0f} fps",
                verdict,
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if row["recommended"]:
                    item.setForeground(QColor(ACCENT))
                elif not row["smooth"]:
                    item.setForeground(QColor("#8a8f98"))
                self.table.setItem(i, column, item)
            if row["recommended"]:
                recommended = i
        if not rows:
            self.status.setText(f"Webcam {self.index} could not be opened.")
        elif recommended is None:
            self.status.setText("No mode runs smoothly. The webcam stays in its default mode.")
        else:
            self.status.setText(
                "Done. The recommended mode is the largest one that runs smoothly and is now used for Auto. "
                "Select another row and press Use selected to override it."
            )
            self.table.selectRow(recommended)

    def _on_selection(self):
        self.use_button.setEnabled(bool(self.table.selectedItems()))

    def done(self, result: int):
        # the test keeps running in the processor if the dialog is closed early, stop listening
        self.processor.test_progress.disconnect(self.status.setText)
        self.processor.test_finished.disconnect(self._on_finished)
        super().done(result)

    def _use_selected(self):
        selected = self.table.selectedItems()
        if selected:
            row = self.rows[selected[0].row()]
            remember_camera_mode(self.index, row["size"], row["mjpg"])
            # the processor already went back to the recommended mode, a different choice needs a reopen
            self.changed = not row["recommended"]
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self, presets: dict, image_path: str | None = None):
        super().__init__()
        self.setWindowTitle("Virtual Makeup")
        self.resize(1200, 840)
        self.presets = presets
        self._image = None  # the currently opened still image, if any
        self._loading = False  # True while a preset fills the rows, suppresses pushes

        self.view = VideoView()
        panel = self._build_panel()

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(self.view, 1)
        layout.addWidget(panel, 0)
        self.setCentralWidget(central)

        self.face_label = QLabel("")
        self.source_label = QLabel("")
        self.source_label.setObjectName("dim")
        self.fps_label = QLabel("")
        self.fps_label.setObjectName("dim")
        self.statusBar().addPermanentWidget(self.face_label)
        self.statusBar().addPermanentWidget(self.source_label)
        self.statusBar().addPermanentWidget(self.fps_label)
        self.statusBar().showMessage("B blur   C compare   Ctrl+O open image   Ctrl+S snapshot   Q quit")

        self._load_preset(0)
        self.processor = Processor(self._settings())
        self.processor.frame_ready.connect(self._on_frame)
        self.processor.fps_changed.connect(self._on_fps)
        self.processor.source_changed.connect(self.source_label.setText)
        self.processor.analysis_changed.connect(self._on_analysis)
        self.processor.shade_matched.connect(self._on_shade_matched)
        self.processor.failed.connect(self.view.set_message)
        self.processor.start()
        self._add_shortcuts()

        if image_path:
            self._open_image(image_path)

    # ---- layout
    def _build_panel(self):
        panel = QWidget()
        column = QVBoxLayout(panel)
        column.setContentsMargins(0, 0, 8, 0)
        column.setSpacing(8)

        title = QLabel("Virtual Makeup")
        title.setObjectName("title")
        subtitle = QLabel("MediaPipe face mesh  ·  OpenCV")
        subtitle.setObjectName("dim")
        column.addWidget(title)
        column.addWidget(subtitle)

        source_box = QGroupBox("Source")
        self.source_combo = QComboBox()
        self.source_combo.addItems([f"Webcam {i}" for i in range(CAMERA_SLOTS)])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        open_button = QPushButton("Open image…")
        open_button.clicked.connect(self._open_image_dialog)
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(list(QUALITY_OPTIONS))
        self.quality_combo.setToolTip(
            "Auto measures the frame rate and picks the largest mode that still runs smoothly. "
            "Max takes the largest size the camera has, whatever the frame rate."
        )
        self.quality_combo.currentIndexChanged.connect(self._on_source_changed)
        quality_label = QLabel("Quality")
        quality_label.setObjectName("dim")
        test_button = QPushButton("Test webcam…")
        test_button.setToolTip("Measure every mode of the webcam and pick the one to use")
        test_button.clicked.connect(self._test_camera)
        grid = QGridLayout(source_box)
        grid.addWidget(self.source_combo, 0, 0, 1, 2)
        grid.addWidget(open_button, 0, 2)
        grid.addWidget(quality_label, 1, 0)
        grid.addWidget(self.quality_combo, 1, 1)
        grid.addWidget(test_button, 1, 2)
        column.addWidget(source_box)

        preset_box = QGroupBox("Preset")
        row = QHBoxLayout(preset_box)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.presets))
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        reset_button = QPushButton("Reset")
        reset_button.setToolTip("Back to the preset values")
        reset_button.clicked.connect(lambda: self._apply_preset(self.preset_combo.currentIndex()))
        row.addWidget(self.preset_combo, 1)
        row.addWidget(reset_button)
        column.addWidget(preset_box)

        makeup_box = QGroupBox("Makeup")
        rows = QVBoxLayout(makeup_box)
        rows.setSpacing(4)
        self.rows = {}
        for key, spec in FEATURES.items():
            feature_row = FeatureRow(spec["label"])
            feature_row.changed.connect(self._push_settings)
            self.rows[key] = feature_row
            rows.addWidget(feature_row)
        self.adaptive_check = QCheckBox("Adapt shades to skin and light")
        self.adaptive_check.setChecked(True)
        self.adaptive_check.setToolTip(
            "Moves every shade toward the skin undertone, deepens it on deeper skin, gives it more "
            "colour where a pale shade would vanish, and scales the intensity with the brightness of "
            "the image. Brows take your own brow colour."
        )
        self.adaptive_check.toggled.connect(self._push_settings)
        self.analysis_label = QLabel("")
        self.analysis_label.setObjectName("dim")
        self.analysis_label.setWordWrap(True)
        rows.addSpacing(4)
        rows.addWidget(self.adaptive_check)
        rows.addWidget(self.analysis_label)
        column.addWidget(makeup_box)

        foundation_box = QGroupBox("Foundation")
        grid = QGridLayout(foundation_box)
        self.foundation_check = QCheckBox("Foundation")
        self.foundation_check.setToolTip("Evens out the skin colour and softens texture, eyes, brows and lips stay untouched")
        self.foundation_check.toggled.connect(self._on_foundation_toggle)
        self.match_check = QCheckBox("Match skin")
        self.match_check.setChecked(True)
        self.match_check.setToolTip("Use a shade measured from the skin, untick to pick one")
        self.match_check.toggled.connect(self._on_match_toggle)
        self.shade_swatch = QPushButton()
        self.shade_swatch.setFixedSize(34, 24)
        self.shade_swatch.setToolTip("Foundation shade")
        self.shade_swatch.clicked.connect(self._pick_shade)
        self._shade = (160, 190, 220)
        self._set_shade_swatch(self._shade)
        self.coverage_slider = QSlider(Qt.Orientation.Horizontal)
        self.coverage_slider.setRange(0, 100)
        self.coverage_slider.setValue(50)
        self.coverage_slider.setToolTip("How much redness, blotches and dark patches are evened out")
        self.smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self.smooth_slider.setRange(0, 100)
        self.smooth_slider.setValue(40)
        self.smooth_slider.setToolTip("How much fine skin texture is softened")
        self.coverage_value, self.smooth_value = QLabel("50%"), QLabel("40%")
        labels = {}
        for key, text in (("coverage", "Coverage"), ("smooth", "Smooth")):
            labels[key] = QLabel(text)
            labels[key].setObjectName("dim")
        for value in (self.coverage_value, self.smooth_value):
            value.setObjectName("dim")
            value.setFixedWidth(36)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.coverage_slider.valueChanged.connect(lambda v: (self.coverage_value.setText(f"{v}%"), self._push_settings()))
        self.smooth_slider.valueChanged.connect(lambda v: (self.smooth_value.setText(f"{v}%"), self._push_settings()))
        grid.addWidget(self.foundation_check, 0, 0)
        grid.addWidget(self.match_check, 0, 1, alignment=Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self.shade_swatch, 0, 2, alignment=Qt.AlignmentFlag.AlignRight)
        grid.addWidget(labels["coverage"], 1, 0)
        grid.addWidget(self.coverage_slider, 1, 1)
        grid.addWidget(self.coverage_value, 1, 2)
        grid.addWidget(labels["smooth"], 2, 0)
        grid.addWidget(self.smooth_slider, 2, 1)
        grid.addWidget(self.smooth_value, 2, 2)
        self._set_foundation_enabled(False)
        column.addWidget(foundation_box)

        background_box = QGroupBox("Background")
        grid = QGridLayout(background_box)
        self.blur_check = QCheckBox("Blur background")
        self.blur_check.toggled.connect(self._on_blur_toggle)
        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setRange(1, 15)
        self.blur_slider.setValue(5)
        self.blur_slider.setEnabled(False)
        self.blur_slider.valueChanged.connect(self._push_settings)
        strength_label = QLabel("Strength")
        strength_label.setObjectName("dim")
        grid.addWidget(self.blur_check, 0, 0, 1, 2)
        grid.addWidget(strength_label, 1, 0)
        grid.addWidget(self.blur_slider, 1, 1)
        column.addWidget(background_box)

        self.compare_check = QCheckBox("Compare before / after")
        self.compare_check.toggled.connect(self._push_settings)
        column.addWidget(self.compare_check)

        snapshot_button = QPushButton("Save snapshot…")
        snapshot_button.setObjectName("accent")
        snapshot_button.clicked.connect(self._save_snapshot)
        column.addWidget(snapshot_button)
        column.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(340)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    def _add_shortcuts(self):
        bindings = (
            ("B", lambda: self.blur_check.toggle()),
            ("C", lambda: self.compare_check.toggle()),
            ("Ctrl+O", self._open_image_dialog),
            ("Ctrl+S", self._save_snapshot),
            ("Q", self.close),
            ("Esc", self.close),
        )
        for keys, slot in bindings:
            action = QAction(self)
            action.setShortcut(QKeySequence(keys))
            action.triggered.connect(slot)
            self.addAction(action)

    # ---- settings flow: controls -> Settings -> Processor
    def _settings(self) -> Settings:
        return Settings(
            style={key: row.state() for key, row in self.rows.items()},
            adaptive=self.adaptive_check.isChecked(),
            foundation=self.foundation_check.isChecked(),
            foundation_shade=None if self.match_check.isChecked() else self._shade,
            coverage=self.coverage_slider.value() / 100,
            smoothing=self.smooth_slider.value() / 100,
            blur_background=self.blur_check.isChecked(),
            blur_strength=self.blur_slider.value() / 100,
            compare=self.compare_check.isChecked(),
        )

    def _push_settings(self, *_):
        if not self._loading:
            self.processor.update_settings(self._settings())

    def _load_preset(self, index: int):
        style = self.presets[self.preset_combo.itemText(index)]
        self._loading = True
        for key, row in self.rows.items():
            entry = style[key]
            row.set_state(entry["color"], entry["alpha"], entry["enabled"])
        self._loading = False

    def _apply_preset(self, index: int):
        self._load_preset(index)
        self._push_settings()

    def _set_foundation_enabled(self, enabled: bool):
        for widget in (self.match_check, self.coverage_slider, self.smooth_slider):
            widget.setEnabled(enabled)
        self.shade_swatch.setEnabled(enabled and not self.match_check.isChecked())

    def _on_foundation_toggle(self, checked: bool):
        self._set_foundation_enabled(checked)
        self._push_settings()

    def _on_match_toggle(self, checked: bool):
        self.shade_swatch.setEnabled(not checked and self.foundation_check.isChecked())
        self._push_settings()

    def _set_shade_swatch(self, color):
        self._shade = tuple(int(c) for c in color)
        self.shade_swatch.setStyleSheet(f"background: {bgr_to_hex(self._shade)}; border: 1px solid #4a4e5a; border-radius: 6px;")

    def _on_shade_matched(self, hex_color: str):
        # show the measured shade, it becomes the starting point if the user picks one
        if self.match_check.isChecked():
            self._set_shade_swatch(hex_to_bgr(hex_color))

    def _pick_shade(self):
        color = QColorDialog.getColor(QColor(bgr_to_hex(self._shade)), self, "Foundation shade")
        if color.isValid():
            self._set_shade_swatch(hex_to_bgr(color.name()))
            self._push_settings()

    def _on_analysis(self, description: str):
        self.analysis_label.setText(f"Adapted for {description}" if description else "")

    def _on_blur_toggle(self, checked: bool):
        self.blur_slider.setEnabled(checked)
        self._push_settings()

    # ---- sources
    def _on_source_changed(self, *_):
        index = self.source_combo.currentIndex()
        if index < CAMERA_SLOTS:
            self.view.set_message(f"Starting webcam {index}…")
            self.processor.use_camera(index, QUALITY_OPTIONS[self.quality_combo.currentText()])
        elif self._image is not None:
            self.processor.use_image(self._image)

    def _test_camera(self):
        index = min(self.source_combo.currentIndex(), CAMERA_SLOTS - 1)
        dialog = CameraTestDialog(self.processor, index, self)
        dialog.exec()
        # the test leaves the webcam in auto mode, whatever was chosen in the dialog is what auto opens
        self.quality_combo.blockSignals(True)
        self.quality_combo.setCurrentText("Auto")
        self.quality_combo.blockSignals(False)
        self.source_combo.blockSignals(True)
        self.source_combo.setCurrentIndex(index)
        self.source_combo.blockSignals(False)
        self.processor.use_camera(index, "auto", reopen=dialog.changed)

    def _open_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if path:
            self._open_image(path)

    def _open_image(self, path: str):
        image = cv2.imread(path)
        if image is None:
            QMessageBox.warning(self, "Open image", f"Could not read {path}")
            return
        self._image = image
        # keep a single "Image: …" entry after the webcam entries
        if self.source_combo.count() == CAMERA_SLOTS:
            self.source_combo.addItem("")
        self.source_combo.setItemText(CAMERA_SLOTS, f"Image: {Path(path).name}")
        self.source_combo.blockSignals(True)
        self.source_combo.setCurrentIndex(CAMERA_SLOTS)
        self.source_combo.blockSignals(False)
        self.processor.use_image(image)

    # ---- processor feedback
    def _on_frame(self, image: QImage, face_found: bool):
        self.view.set_image(image)
        self.face_label.setText("Face detected" if face_found else "No face")
        self.face_label.setStyleSheet("color: #8fd18f;" if face_found else f"color: {ACCENT};")

    def _on_fps(self, fps: float):
        self.fps_label.setText(f"{fps:.0f} fps" if fps > 0 else "")

    def _save_snapshot(self):
        image = self.view.image()
        if image is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save snapshot", "makeup.png", "PNG image (*.png);;JPEG image (*.jpg)")
        if path:
            image.save(path)

    def closeEvent(self, event):
        self.processor.stop()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="Virtual Makeup desktop app")
    parser.add_argument("--image", help="Start with an image instead of the webcam.")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    window = MainWindow(load_presets(), image_path=args.image)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
