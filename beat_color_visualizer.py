#!/usr/bin/env python3
"""
Beat Color Visualizer
====================
သီချင်းထဲက Beat (စည်းချက်) အတိုင်း Window တစ်ခုလုံး အရောင်များ အလိုအလျောက် ပြောင်းလဲပေးသည့် Python Script။

Features:
- Live Microphone (မိုက်ခရိုဖုန်းမှတစ်ဆင့် အသံဖမ်းယူခြင်း)
- Audio File Player (MP3, WAV, FLAC, OGG သီချင်းဖွင့်ခြင်း)
- 5 Different Color Pulse Modes (မီးရောင်စုံ Mode ၅ မျိုး)
- Fullscreen Support (F11 သို့မဟုတ် Double Click)
- Auto-hide controls for pure visual immersion
"""

import sys
import os
import math
import time
import colorsys
from collections import deque

import numpy as np
import sounddevice as sd
import soundfile as sf
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt6.QtGui import (
    QPainter, QColor, QRadialGradient, QLinearGradient,
    QFont, QPen, QBrush, QKeySequence, QShortcut
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider,
    QComboBox, QGraphicsDropShadowEffect
)

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024


class AudioEngine:
    """Handles both real-time microphone input and audio file playback with FFT analysis."""
    def __init__(self, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE):
        self.sample_rate = sample_rate
        self.block_size = block_size
        
        # Audio state
        self.mode = "mic"  # "mic" or "file"
        self.stream = None
        self.file_data = None
        self.file_pos = 0
        self.file_total_samples = 0
        self.is_playing = True
        
        # Audio feature extraction
        self.current_buffer = np.zeros(self.block_size, dtype=np.float32)
        self.bass_level = 0.0
        self.mid_level = 0.0
        self.treble_level = 0.0
        self.overall_volume = 0.0
        self.is_beat = False
        
        # Beat detection history & adaptive threshold
        self.energy_history = deque(maxlen=43)  # ~1 second of history at 43 blocks/sec
        self.beat_decay = 0.0
        self.sensitivity = 1.3  # Multiplier for beat threshold
        self.last_beat_time = 0
        self.min_beat_interval = 0.16  # Minimum 160ms between beats (~375 BPM max)
        
        # Hanning window for FFT
        self.window = np.hanning(self.block_size)
        
        self.start_mic()

    def start_mic(self):
        self.stop()
        self.mode = "mic"
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                callback=self._mic_callback
            )
            self.stream.start()
            self.is_playing = True
            return True, "Microphone connected"
        except Exception as e:
            return False, f"Mic error: {e}"

    def load_file(self, file_path):
        self.stop()
        try:
            data, sr = sf.read(file_path, dtype='float32')
            # Convert to mono if stereo or multi-channel
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            
            self.file_data = data
            self.sample_rate = sr
            self.file_total_samples = len(data)
            self.file_pos = 0
            self.mode = "file"
            self.window = np.hanning(self.block_size)

            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                callback=self._file_callback
            )
            self.stream.start()
            self.is_playing = True
            return True, os.path.basename(file_path)
        except Exception as e:
            return False, f"File load error: {e}"

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def toggle_play(self):
        self.is_playing = not self.is_playing
        return self.is_playing

    def _mic_callback(self, indata, frames, time_info, status):
        if not self.is_playing:
            return
        audio_chunk = indata[:, 0]
        self._process_audio(audio_chunk)

    def _file_callback(self, outdata, frames, time_info, status):
        if not self.is_playing or self.file_data is None:
            outdata.fill(0)
            return
            
        remaining = self.file_total_samples - self.file_pos
        if remaining <= 0:
            # Loop file
            self.file_pos = 0
            remaining = self.file_total_samples

        chunk_size = min(frames, remaining)
        audio_chunk = self.file_data[self.file_pos:self.file_pos + chunk_size]
        self.file_pos += chunk_size
        
        if chunk_size < frames:
            outdata[:chunk_size, 0] = audio_chunk
            outdata[chunk_size:, 0] = 0
        else:
            outdata[:, 0] = audio_chunk

        self._process_audio(outdata[:, 0])

    def _process_audio(self, chunk):
        if len(chunk) < self.block_size:
            chunk = np.pad(chunk, (0, self.block_size - len(chunk)))
        else:
            chunk = chunk[:self.block_size]

        # RMS overall volume
        volume = float(np.sqrt(np.mean(chunk**2)))
        self.overall_volume = volume

        # FFT Analysis
        windowed = chunk * self.window
        fft = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(self.block_size, 1.0 / self.sample_rate)

        # Bass band: 20Hz - 160Hz (Kick drums, sub-bass)
        bass_idx = np.where((freqs >= 20) & (freqs <= 160))[0]
        bass_energy = float(np.mean(fft[bass_idx])) if len(bass_idx) > 0 else 0.0

        # Mid band: 160Hz - 2500Hz (Vocals, snare, chords)
        mid_idx = np.where((freqs > 160) & (freqs <= 2500))[0]
        mid_energy = float(np.mean(fft[mid_idx])) if len(mid_idx) > 0 else 0.0

        # Treble band: 2500Hz - 16000Hz (Hi-hats, cymbals, shimmer)
        treble_idx = np.where((freqs > 2500) & (freqs <= 16000))[0]
        treble_energy = float(np.mean(fft[treble_idx])) if len(treble_idx) > 0 else 0.0

        # Normalize with soft saturation
        self.bass_level = math.tanh(bass_energy * 0.15)
        self.mid_level = math.tanh(mid_energy * 0.25)
        self.treble_level = math.tanh(treble_energy * 0.4)

        # Beat detection logic using adaptive threshold
        now = time.time()
        self.energy_history.append(bass_energy)

        if len(self.energy_history) >= 10:
            avg_energy = float(np.mean(self.energy_history))
            std_energy = float(np.std(self.energy_history))
            threshold = avg_energy + (self.sensitivity * std_energy)
            
            # Check beat condition
            if (bass_energy > threshold and 
                bass_energy > 0.05 and 
                (now - self.last_beat_time) > self.min_beat_interval):
                self.is_beat = True
                self.last_beat_time = now
                self.beat_decay = 1.0
            else:
                self.is_beat = False
        else:
            self.is_beat = False


class VisualizerCanvas(QWidget):
    """Modern Canvas that paints responsive, vibrant visuals matching the music beat."""
    
    MODES = [
        "Solid Strobe (မီးရောင်စုံ ဖလက်ရှ်)",
        "Radial Pulse (ဗဟို အလင်းလှိုင်း)",
        "Rainbow Wave (သက်တံရောင်စဉ်)",
        "Cyberpunk Neon (ခရမ်းစိမ်းနီယွန်)",
        "Ambient Breathing (အေးဆေး အလင်းစုံ)",
    ]

    def __init__(self, audio_engine: AudioEngine, parent=None):
        super().__init__(parent)
        self.audio = audio_engine
        self.current_mode_index = 0
        
        # Color & animation states
        self.base_hue = 0.0
        self.target_hue = 0.0
        self.flash_intensity = 0.0
        self.pulse_radius = 0.0
        self.wave_phase = 0.0
        self.color_palette_idx = 0
        
        # Cyberpunk palette presets
        self.cyberpunk_colors = [
            (320 / 360.0, 1.0, 1.0),  # Neon Pink / Magenta
            (180 / 360.0, 1.0, 1.0),  # Bright Cyan
            (275 / 360.0, 1.0, 1.0),  # Electric Purple
            (30 / 360.0, 1.0, 1.0),   # Neon Orange
            (130 / 360.0, 1.0, 1.0),  # Acid Green
        ]

        # 60 FPS update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)  # ~60 FPS

    def next_mode(self):
        self.current_mode_index = (self.current_mode_index + 1) % len(self.MODES)
        return self.MODES[self.current_mode_index]

    def set_mode(self, index):
        self.current_mode_index = index % len(self.MODES)

    def update_animation(self):
        # Update beat decay & flash
        if self.audio.is_beat:
            self.flash_intensity = 1.0
            self.pulse_radius = 0.1
            
            # Step hue on beat
            self.base_hue = (self.base_hue + 0.12 + (self.audio.bass_level * 0.1)) % 1.0
            self.color_palette_idx = (self.color_palette_idx + 1) % len(self.cyberpunk_colors)
        else:
            # Smooth exponential decay
            self.flash_intensity *= 0.88
            self.pulse_radius += 0.015

        # Ambient continuous hue flow
        self.base_hue = (self.base_hue + 0.0015) % 1.0
        self.wave_phase += 0.04

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        cx = w / 2.0
        cy = h / 2.0
        diag = math.hypot(w, h)

        mode = self.current_mode_index

        if mode == 0:
            # Mode 0: Solid Strobe / Club Light (စတိတ်ရှိုး မီးရောင်စုံ ခတ်သည့်ပုံစံ)
            # Hue rotates on beat; brightness spikes with beat flash & bass level
            sat = 0.95
            val = min(1.0, 0.25 + (self.flash_intensity * 0.65) + (self.audio.bass_level * 0.3))
            r, g, b = colorsys.hsv_to_rgb(self.base_hue, sat, val)
            painter.fillRect(0, 0, w, h, QColor.fromRgbF(r, g, b))

        elif mode == 1:
            # Mode 1: Radial Pulse (ဗဟို အလင်းလှိုင်း)
            # Center pulses outward reacting to bass and mids
            r1, g1, b1 = colorsys.hsv_to_rgb(self.base_hue, 0.9, min(1.0, 0.4 + self.flash_intensity * 0.6))
            r2, g2, b2 = colorsys.hsv_to_rgb((self.base_hue + 0.35) % 1.0, 0.95, 0.12 + self.audio.mid_level * 0.25)
            
            radius = max(50.0, (diag * 0.3) + (self.flash_intensity * diag * 0.4) + (self.audio.bass_level * 150))
            gradient = QRadialGradient(QPointF(cx, cy), radius)
            gradient.setColorAt(0.0, QColor.fromRgbF(r1, g1, b1))
            gradient.setColorAt(0.6, QColor.fromRgbF(r2, g2, b2))
            gradient.setColorAt(1.0, QColor(10, 10, 15))

            painter.fillRect(0, 0, w, h, gradient)

            # Expanding ring waves on beats
            if self.pulse_radius < 1.0:
                ring_rad = self.pulse_radius * (diag * 0.65)
                alpha = int(max(0, (1.0 - self.pulse_radius) * 200))
                ring_pen = QPen(QColor(255, 255, 255, alpha), 4 + (self.audio.treble_level * 8))
                painter.setPen(ring_pen)
                painter.drawEllipse(QPointF(cx, cy), ring_rad, ring_rad)

        elif mode == 2:
            # Mode 2: Rainbow Spectrum Wave (သက်တံရောင်စဉ်)
            # Multi-angle linear gradient waving smoothly across the window
            angle = self.wave_phase * 0.5
            dx = math.cos(angle) * (w / 2)
            dy = math.sin(angle) * (h / 2)
            
            grad = QLinearGradient(cx - dx, cy - dy, cx + dx, cy + dy)
            for i in range(5):
                pos = i / 4.0
                h_val = (self.base_hue + (pos * 0.6)) % 1.0
                v_val = min(1.0, 0.35 + (self.flash_intensity * 0.5) + (self.audio.overall_volume * 0.4))
                cr, cg, cb = colorsys.hsv_to_rgb(h_val, 0.9, v_val)
                grad.setColorAt(pos, QColor.fromRgbF(cr, cg, cb))
                
            painter.fillRect(0, 0, w, h, grad)

        elif mode == 3:
            # Mode 3: Cyberpunk Neon Beats (ခရမ်းစိမ်းနီယွန်)
            h1, s1, v1 = self.cyberpunk_colors[self.color_palette_idx]
            h2, s2, v2 = self.cyberpunk_colors[(self.color_palette_idx + 1) % len(self.cyberpunk_colors)]
            
            v1_boosted = min(1.0, 0.2 + (self.flash_intensity * 0.75) + (self.audio.bass_level * 0.2))
            v2_boosted = min(1.0, 0.15 + (self.audio.mid_level * 0.5))

            r1, g1, b1 = colorsys.hsv_to_rgb(h1, s1, v1_boosted)
            r2, g2, b2 = colorsys.hsv_to_rgb(h2, s2, v2_boosted)

            grad = QRadialGradient(QPointF(cx, cy), diag * 0.6)
            grad.setColorAt(0.0, QColor.fromRgbF(r1, g1, b1))
            grad.setColorAt(0.7, QColor.fromRgbF(r2, g2, b2))
            grad.setColorAt(1.0, QColor(5, 5, 12))

            painter.fillRect(0, 0, w, h, grad)

            # Central pulse shape (polygon / reactive equalizer diamond)
            poly_size = 60 + (self.audio.bass_level * 180) + (self.flash_intensity * 60)
            painter.save()
            painter.translate(cx, cy)
            painter.rotate(self.wave_phase * 20)
            diamond_pen = QPen(QColor(255, 255, 255, int(150 + self.flash_intensity * 105)), 3)
            painter.setPen(diamond_pen)
            painter.setBrush(QBrush(QColor(int(r1 * 255), int(g1 * 255), int(b1 * 255), 80)))
            painter.drawRect(QRectF(-poly_size / 2, -poly_size / 2, poly_size, poly_size))
            painter.restore()

        elif mode == 4:
            # Mode 4: Ambient Breathing (အေးဆေး အလင်းစုံ)
            # Smooth and gentle color shifting, pulsing softly with the song
            b_val = min(1.0, 0.3 + (self.audio.bass_level * 0.4) + (self.flash_intensity * 0.3))
            r, g, b = colorsys.hsv_to_rgb(self.base_hue, 0.75, b_val)
            painter.fillRect(0, 0, w, h, QColor.fromRgbF(r, g, b))

        # Bottom Beat Reactive Bar (Frequency visualizer strip)
        self._draw_frequency_strip(painter, w, h)

    def _draw_frequency_strip(self, painter: QPainter, w, h):
        """Draws subtle modern reactive sound indicators at bottom."""
        bar_h = 8 + (self.audio.bass_level * 16)
        y = h - bar_h
        
        # Bass bar (Left)
        b_color = QColor(255, 60, 100, int(180 + self.flash_intensity * 75))
        painter.fillRect(QRectF(0, y, w * 0.33, bar_h), b_color)
        
        # Mid bar (Center)
        m_color = QColor(60, 220, 255, int(150 + self.audio.mid_level * 100))
        painter.fillRect(QRectF(w * 0.33, y, w * 0.34, bar_h), m_color)
        
        # Treble bar (Right)
        t_color = QColor(255, 220, 60, int(150 + self.audio.treble_level * 100))
        painter.fillRect(QRectF(w * 0.67, y, w * 0.33, bar_h), t_color)


class MainWindow(QMainWindow):
    """Main Application Window with Fullscreen and Controls Overlay."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beat Color Visualizer")
        self.resize(1000, 650)
        self.setMinimumSize(600, 400)

        # Audio engine
        self.audio = AudioEngine()

        # Canvas
        self.canvas = VisualizerCanvas(self.audio, self)
        self.setCentralWidget(self.canvas)

        # Control Overlay Panel
        self.init_overlay()

        # Shortcuts
        QShortcut(QKeySequence("F11"), self, self.toggle_fullscreen)
        QShortcut(QKeySequence("F"), self, self.toggle_fullscreen)
        QShortcut(QKeySequence("Space"), self, self.toggle_playback)
        QShortcut(QKeySequence("M"), self, self.cycle_mode)
        QShortcut(QKeySequence("O"), self, self.open_audio_file)
        QShortcut(QKeySequence("L"), self, self.switch_to_mic)
        QShortcut(QKeySequence("H"), self, self.toggle_controls)
        QShortcut(QKeySequence("Escape"), self, self.on_escape)
        QShortcut(QKeySequence("Up"), self, self.increase_sensitivity)
        QShortcut(QKeySequence("Down"), self, self.decrease_sensitivity)

        # Mouse activity tracking for auto-hiding overlay
        self.setMouseTracking(True)
        self.canvas.setMouseTracking(True)
        self.last_mouse_move = time.time()
        
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.check_auto_hide)
        self.ui_timer.start(500)

    def init_overlay(self):
        """Build translucent modern control bar."""
        self.overlay = QWidget(self)
        self.overlay.setStyleSheet("""
            QWidget#overlay {
                background-color: rgba(18, 18, 24, 0.78);
                border-radius: 14px;
                border: 1px solid rgba(255, 255, 255, 0.15);
            }
            QLabel {
                color: #FFFFFF;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton {
                background-color: rgba(255, 255, 255, 0.12);
                color: #FFFFFF;
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                padding: 6px 14px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.25);
                border-color: rgba(255, 255, 255, 0.4);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.35);
            }
            QComboBox {
                background-color: rgba(255, 255, 255, 0.12);
                color: #FFFFFF;
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                padding: 4px 10px;
                font-size: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e24;
                color: #FFFFFF;
                selection-background-color: #6366f1;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #6366f1;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #FFFFFF;
                border: 1px solid #6366f1;
                width: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
        """)
        self.overlay.setObjectName("overlay")

        layout = QHBoxLayout(self.overlay)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(12)

        # Source Button
        self.btn_source = QPushButton("🎤 Mic Mode")
        self.btn_source.clicked.connect(self.toggle_source)
        layout.addWidget(self.btn_source)

        # File Select Button
        self.btn_open = QPushButton("📂 Open Song")
        self.btn_open.clicked.connect(self.open_audio_file)
        layout.addWidget(self.btn_open)

        # Mode Selection Combobox
        layout.addWidget(QLabel("Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(VisualizerCanvas.MODES)
        self.combo_mode.currentIndexChanged.connect(self.canvas.set_mode)
        layout.addWidget(self.combo_mode)

        # Sensitivity Slider
        layout.addWidget(QLabel("Sensitivity:"))
        self.slider_sens = QSlider(Qt.Orientation.Horizontal)
        self.slider_sens.setRange(5, 30)  # 0.5x to 3.0x
        self.slider_sens.setValue(int(self.audio.sensitivity * 10))
        self.slider_sens.setFixedWidth(110)
        self.slider_sens.valueChanged.connect(self.on_sens_changed)
        layout.addWidget(self.slider_sens)

        # Fullscreen Button
        self.btn_fs = QPushButton("⛶ Fullscreen")
        self.btn_fs.clicked.connect(self.toggle_fullscreen)
        layout.addWidget(self.btn_fs)

        # Status / Help Label
        self.lbl_status = QLabel("Press 'H' to hide controls | 'M' next mode | 'F' fullscreen")
        self.lbl_status.setStyleSheet("color: rgba(255, 255, 255, 0.65); font-size: 11px;")
        layout.addWidget(self.lbl_status)

        # Shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 160))
        shadow.setOffset(0, 4)
        self.overlay.setGraphicsEffect(shadow)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Position overlay centered at top
        ov_w = min(self.width() - 40, 920)
        ov_h = 56
        ov_x = (self.width() - ov_w) // 2
        ov_y = 20
        self.overlay.setGeometry(ov_x, ov_y, ov_w, ov_h)

    def mouseMoveEvent(self, event):
        self.last_mouse_move = time.time()
        if not self.overlay.isVisible():
            self.overlay.show()
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.toggle_fullscreen()
        super().mouseDoubleClickEvent(event)

    def check_auto_hide(self):
        # Auto-hide overlay after 3.5 seconds of inactivity in fullscreen or normal mode
        if time.time() - self.last_mouse_move > 3.5 and self.overlay.isVisible():
            self.overlay.hide()

    def toggle_controls(self):
        if self.overlay.isVisible():
            self.overlay.hide()
        else:
            self.overlay.show()
            self.last_mouse_move = time.time()

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self.btn_fs.setText("⛶ Fullscreen")
        else:
            self.showFullScreen()
            self.btn_fs.setText("❐ Exit Fullscreen")
        self.last_mouse_move = time.time()

    def on_escape(self):
        if self.isFullScreen():
            self.showNormal()
            self.btn_fs.setText("⛶ Fullscreen")
        else:
            self.close()

    def toggle_playback(self):
        is_playing = self.audio.toggle_play()
        status = "Playing" if is_playing else "Paused"
        self.lbl_status.setText(f"Status: {status}")

    def cycle_mode(self):
        next_name = self.canvas.next_mode()
        self.combo_mode.setCurrentIndex(self.canvas.current_mode_index)
        self.lbl_status.setText(f"Mode: {next_name}")

    def on_sens_changed(self, value):
        self.audio.sensitivity = value / 10.0
        self.lbl_status.setText(f"Beat Sensitivity: {self.audio.sensitivity:.1f}x")

    def increase_sensitivity(self):
        new_val = min(30, self.slider_sens.value() + 2)
        self.slider_sens.setValue(new_val)

    def decrease_sensitivity(self):
        new_val = max(5, self.slider_sens.value() - 2)
        self.slider_sens.setValue(new_val)

    def toggle_source(self):
        if self.audio.mode == "mic":
            self.open_audio_file()
        else:
            self.switch_to_mic()

    def switch_to_mic(self):
        ok, msg = self.audio.start_mic()
        if ok:
            self.btn_source.setText("🎤 Mic Mode")
            self.lbl_status.setText("Source: Live Microphone (listening to room music)")
        else:
            self.lbl_status.setText(msg)

    def open_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "သီချင်းဖိုင် ရွေးချယ်ပါ (Select Audio File)",
            "",
            "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a)"
        )
        if file_path:
            ok, msg = self.audio.load_file(file_path)
            if ok:
                self.btn_source.setText("🎵 Song File")
                self.lbl_status.setText(f"Playing: {msg}")
            else:
                self.lbl_status.setText(msg)

    def closeEvent(self, event):
        self.audio.stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    
    # If audio file passed as CLI argument: python beat_color_visualizer.py song.mp3
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        window.audio.load_file(sys.argv[1])
        window.btn_source.setText("🎵 Song File")
        window.lbl_status.setText(f"Playing: {os.path.basename(sys.argv[1])}")

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
