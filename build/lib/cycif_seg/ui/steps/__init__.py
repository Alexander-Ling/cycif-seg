"""UI step panels.

This package contains small Qt widgets used as panels for the main application tabs.
Keeping these in separate files drastically reduces the size of `main_widget.py` and
helps avoid indentation-related regressions when iterating on UI code.
"""

from .step1_preprocess_panel import Step1PreprocessPanel
from .step2a_nuclei_panel import Step2aNucleiPanel
from .step2b_edit_panel import Step2bEditPanel
