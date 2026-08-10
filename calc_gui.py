import sys
import os
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QTabWidget, QWidget, QGridLayout, QComboBox, QLineEdit,
    QLabel, QPushButton
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

FILTERS = ['F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F410M', 'F444W']


def load_coefficients():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "coefficients.csv")
    return pd.read_csv(csv_path)


def load_shape_params():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "shape_params.csv")
    return pd.read_csv(csv_path)


def erfc_curve(x, d0, p, p1, p2, p3):
    from scipy import special
    return 0.5 * d0 * special.erfc(p * (x - p1)) * p2 + p3


def completeness_percent(filt, exposure_time, background, size, mag, coef_df, shape_df,
                         remove_confusion=False):
    """Completeness percentage for an object of given magnitude.

    Midpoint (p1) comes from the model; the other erfc shape parameters are
    interpolated across radius from shape_params.csv.

    If remove_confusion is True, the percentage is divided by the bright-end
    plateau height, so the plateau becomes 100%. This reports detection
    completeness against noise alone, with the confusion (field-crowding)
    suppression removed.
    """
    p1 = completeness_mag(filt, exposure_time, background, size, coef_df)
    radii = shape_df['radius'].values
    lr = np.log10(np.clip(size, radii.min(), radii.max()))
    d0 = np.interp(lr, np.log10(radii), shape_df['d0'].values)
    p  = np.interp(lr, np.log10(radii), shape_df['p'].values)
    p2 = np.interp(lr, np.log10(radii), shape_df['p2'].values)
    p3 = np.interp(lr, np.log10(radii), shape_df['p3'].values)

    pct = erfc_curve(mag, d0, p, p1, p2, p3)

    if remove_confusion:
        # plateau = curve evaluated at a very bright magnitude
        plateau = erfc_curve(p1 - 10, d0, p, p1, p2, p3)
        if plateau > 0:
            pct = pct / plateau * 100.0

    return max(0.0, min(100.0, pct))


def completeness_percent_err(filt, exposure_time, background, size, mag, coef_df, shape_df):
    """1-sigma uncertainty on the completeness percentage.

    Monte Carlo over both the midpoint uncertainty (dominates near the steep
    part of the curve) and the shape-parameter spreads (dominate near the
    plateau, where field-to-field confusion makes the height uncertain)."""
    radii = shape_df['radius'].values
    lr = np.log10(np.clip(size, radii.min(), radii.max()))
    logr = np.log10(radii)

    p1 = completeness_mag(filt, exposure_time, background, size, coef_df)
    p1_err = completeness_mag_err(filt, exposure_time, background, size, coef_df)

    d0 = np.interp(lr, logr, shape_df['d0'].values)
    p  = np.interp(lr, logr, shape_df['p'].values)
    p2 = np.interp(lr, logr, shape_df['p2'].values)
    p3 = np.interp(lr, logr, shape_df['p3'].values)
    d0_s = np.interp(lr, logr, shape_df['d0_std'].values)
    p_s  = np.interp(lr, logr, shape_df['p_std'].values)
    p2_s = np.interp(lr, logr, shape_df['p2_std'].values)
    p3_s = np.interp(lr, logr, shape_df['p3_std'].values)

    rng = np.random.default_rng(0)
    n = 2000
    samples = erfc_curve(
        mag,
        rng.normal(d0, d0_s, n),
        rng.normal(p,  p_s,  n),
        rng.normal(p1, p1_err, n),
        rng.normal(p2, p2_s, n),
        rng.normal(p3, p3_s, n),
    )
    samples = np.clip(samples, 0.0, 100.0)
    return np.std(samples)


def completeness_mag(filt, exposure_time, background, size, df):
    """50% completeness magnitude for a given filter, exposure, background, size.

    Evaluates p = a(log10(e) - e_avg) + b(g - g_avg) + c_combined at each of the
    9 radii, then interpolates across log10(radius) at the requested size.
    """
    sub = df[df['filter'] == filt].sort_values('radius')

    radii = sub['radius'].values
    a = sub['a'].values
    b = sub['b'].values
    c = sub['c_combined'].values
    e_avg = sub['e_avg'].values
    g_avg = sub['g_avg'].values

    p_per_radius = (
        a * (np.log10(exposure_time) - e_avg)
        + b * (background - g_avg)
        + c
    )

    # interpolate across log10(radius)
    size = np.clip(size, radii.min(), radii.max())
    return np.interp(np.log10(size), np.log10(radii), p_per_radius)


def completeness_mag_err(filt, exposure_time, background, size, df):
    """1-sigma uncertainty on the 50% completeness magnitude.

    Combines the coefficient-propagated error (how well the fit line is
    determined) with the fit RMS (scatter of points around the line) in
    quadrature, giving the total predictive uncertainty."""
    sub = df[df['filter'] == filt].sort_values('radius')
    radii = sub['radius'].values
    e_avg = sub['e_avg'].values
    g_avg = sub['g_avg'].values
    a_err = sub['a_err'].values
    b_err = sub['b_err'].values
    c_err = sub['c_combined_err'].values
    rms = sub['rms'].values

    coeff_var = (
        ((np.log10(exposure_time) - e_avg) ** 2) * a_err ** 2
        + ((background - g_avg) ** 2) * b_err ** 2
        + c_err ** 2
    )
    total_sigma = np.sqrt(coeff_var + rms ** 2)

    size = np.clip(size, radii.min(), radii.max())
    return np.interp(np.log10(size), np.log10(radii), total_sigma)


class MagnitudeCalculatorTab(QWidget):
    def __init__(self, df):
        super().__init__()
        self.df = df
        layout = QGridLayout()
        self.setLayout(layout)

        font = QFont("Arial", 12)
        font.setBold(True)
        self.setFont(font)

        filter_label = QLabel("Filter:")
        size_label = QLabel("Object Size (arcsec):")
        background_label = QLabel("Background (MJy/sr):")
        exposure_label = QLabel("Exposure Time (sec):")

        self.filter_box = QComboBox()
        self.filter_box.addItems(FILTERS)

        self.object_size = QLineEdit()
        self.object_size.setPlaceholderText("0.01 to 1.0")

        self.background = QLineEdit()
        self.background.setPlaceholderText("Enter background in MJy/sr")

        self.exposure_time = QLineEdit()
        self.exposure_time.setPlaceholderText("Enter exposure time in seconds")

        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.clicked.connect(self.calculate_output)

        self.output_label = QLabel("Output will be displayed here")
        self.output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #FFFFFF;")

        layout.addWidget(filter_label,        0, 0)
        layout.addWidget(self.filter_box,     0, 1)
        layout.addWidget(size_label,          1, 0)
        layout.addWidget(self.object_size,    1, 1)
        layout.addWidget(background_label,    2, 0)
        layout.addWidget(self.background,     2, 1)
        layout.addWidget(exposure_label,      3, 0)
        layout.addWidget(self.exposure_time,  3, 1)
        layout.addWidget(self.calculate_button, 4, 0, 1, 2)
        layout.addWidget(self.output_label,     5, 0, 1, 2)

    def calculate_output(self):
        try:
            filt = self.filter_box.currentText()
            size = float(self.object_size.text())
            background = float(self.background.text())
            exposure_time = float(self.exposure_time.text())

            if not (0.01 <= size <= 1.0):
                self.output_label.setText("Object size must be between 0.01 and 1.0 arcsec.")
                return
            if exposure_time <= 0:
                self.output_label.setText("Exposure time must be positive.")
                return

            result = completeness_mag(filt, exposure_time, background, size, self.df)
            err = completeness_mag_err(filt, exposure_time, background, size, self.df)
            self.output_label.setText(f"50% Completeness Mag: {result:.2f} \u00b1 {err:.2f}")

        except ValueError:
            self.output_label.setText("Please enter valid numbers.")


class ProbabilityCalculatorTab(QWidget):
    def __init__(self, coef_df, shape_df):
        super().__init__()
        self.coef_df = coef_df
        self.shape_df = shape_df
        layout = QGridLayout()
        self.setLayout(layout)

        font = QFont("Arial", 12)
        font.setBold(True)
        self.setFont(font)

        filter_label = QLabel("Filter:")
        size_label = QLabel("Object Size (arcsec):")
        background_label = QLabel("Background (MJy/sr):")
        exposure_label = QLabel("Exposure Time (sec):")
        magnitude_label = QLabel("Object Magnitude (AB):")

        self.filter_box = QComboBox()
        self.filter_box.addItems(FILTERS)

        self.object_size = QLineEdit()
        self.object_size.setPlaceholderText("0.01 to 1.0")

        self.background = QLineEdit()
        self.background.setPlaceholderText("Enter background in MJy/sr")

        self.exposure_time = QLineEdit()
        self.exposure_time.setPlaceholderText("Enter exposure time in seconds")

        self.magnitude = QLineEdit()
        self.magnitude.setPlaceholderText("Enter object magnitude")

        confusion_label = QLabel("Confusion:")
        self.confusion_box = QComboBox()
        self.confusion_box.addItems(["With confusion", "Without confusion"])

        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.clicked.connect(self.calculate_output)

        self.output_label = QLabel("Output will be displayed here")
        self.output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #FFFFFF;")

        layout.addWidget(filter_label,        0, 0)
        layout.addWidget(self.filter_box,     0, 1)
        layout.addWidget(size_label,          1, 0)
        layout.addWidget(self.object_size,    1, 1)
        layout.addWidget(background_label,    2, 0)
        layout.addWidget(self.background,     2, 1)
        layout.addWidget(exposure_label,      3, 0)
        layout.addWidget(self.exposure_time,  3, 1)
        layout.addWidget(magnitude_label,     4, 0)
        layout.addWidget(self.magnitude,      4, 1)
        layout.addWidget(confusion_label,     5, 0)
        layout.addWidget(self.confusion_box,  5, 1)
        layout.addWidget(self.calculate_button, 6, 0, 1, 2)
        layout.addWidget(self.output_label,     7, 0, 1, 2)

    def calculate_output(self):
        try:
            filt = self.filter_box.currentText()
            size = float(self.object_size.text())
            background = float(self.background.text())
            exposure_time = float(self.exposure_time.text())
            mag = float(self.magnitude.text())
            remove_confusion = self.confusion_box.currentText() == "Without confusion"

            if not (0.01 <= size <= 1.0):
                self.output_label.setText("Object size must be between 0.01 and 1.0 arcsec.")
                return
            if exposure_time <= 0:
                self.output_label.setText("Exposure time must be positive.")
                return

            pct = completeness_percent(filt, exposure_time, background, size, mag,
                                       self.coef_df, self.shape_df,
                                       remove_confusion=remove_confusion)
            pct_err = completeness_percent_err(filt, exposure_time, background, size, mag,
                                               self.coef_df, self.shape_df)

            if mag < 20 or mag > 32:
                self.output_label.setText(
                    f"Completeness: {pct:.1f}% \u00b1 {pct_err:.1f}%\n"
                    f"(Note: magnitude {mag:g} is outside the tested range of 20–32;\n"
                    f"result is extrapolated.)"
                )
            else:
                self.output_label.setText(f"Completeness: {pct:.1f}% \u00b1 {pct_err:.1f}%")

        except ValueError:
            self.output_label.setText("Please enter valid numbers.")


class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NIRCam Completeness Calculator")
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
        self.setWindowIcon(QIcon(icon_path))
        df = load_coefficients()
        shape_df = load_shape_params()
        self.magnitude_tab = MagnitudeCalculatorTab(df)
        self.probability_tab = ProbabilityCalculatorTab(df, shape_df)
        self.addTab(self.magnitude_tab, "Magnitude Calculation")
        self.addTab(self.probability_tab, "Completeness Calculation")
        self.setMinimumSize(900, 500)
        self.resize(900, 500)
        self.setStyleSheet("""
            QWidget { background-color: #2E2E2E; color: #FFFFFF; font-size: 14px; }
            QLineEdit, QComboBox {
                background-color: #4D4D4D; border: 1px solid #5E5E5E;
                padding: 8px; font-size: 14px; color: #FFFFFF;
            }
            QLabel { font-size: 14px; color: #FFFFFF; }
            QPushButton {
                background-color: #1E90FF; color: #FFFFFF; padding: 10px;
                font-size: 16px; border-radius: 5px;
            }
            QPushButton:hover { background-color: #63B8FF; }
            QPushButton:pressed { background-color: #104E8B; }
            QComboBox QAbstractItemView {
                background-color: #4D4D4D; selection-background-color: #1E90FF; color: #FFFFFF;
            }
            QTabWidget::pane { background-color: #2E2E2E; border: 1px solid #444444; }
            QTabBar::tab { background-color: #4D4D4D; color: #FFFFFF; padding: 8px; margin-right: 2px; }
            QTabBar::tab:selected { background-color: #1E90FF; }
            QTabBar::tab:hover { background-color: #63B8FF; }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())