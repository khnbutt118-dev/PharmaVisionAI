
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from datetime import datetime

OUTPUT_FILE = "PharmaSight_AI_Project_Master_Document.docx"


def shade_cell(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Shading Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = str(header)
        shade_cell(cell, "1F4E79")

        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)

    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = str(value)

    doc.add_paragraph()
    return table


def add_bullets(doc, items):
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def add_numbered(doc, items):
    for item in items:
        doc.add_paragraph(item, style="List Number")


def add_code(doc, text):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Inches(0.2)
    run = paragraph.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(8)


def add_section(doc, title, content):
    doc.add_heading(title, level=1)
    doc.add_paragraph(content)


def create_document():
    doc = Document()

    # Page formatting
    section = doc.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

    # Default fonts
    doc.styles["Normal"].font.name = "Aptos"
    doc.styles["Normal"].font.size = Pt(9.5)

    doc.styles["Heading 1"].font.color.rgb = RGBColor(
        31, 78, 121
    )
    doc.styles["Heading 2"].font.color.rgb = RGBColor(
        46, 116, 181
    )

    # Header
    header = section.header.paragraphs[0]
    header.text = "PHARMASIGHT AI | TECHNICAL DOCUMENTATION"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    # Footer
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run(
        "Project Master Document | Technical Handoff"
    )

    # Cover page
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run("\n\n")

    title = paragraph.add_run("PHARMASIGHT AI")
    title.bold = True
    title.font.size = Pt(30)
    title.font.color.rgb = RGBColor(31, 78, 121)

    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = paragraph.add_run(
        "Project Master Document & Technical Handoff"
    )
    subtitle.font.size = Pt(19)

    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run(
        "\nArchitecture, Modules, Datasets, Training,\n"
        "Evaluation, Implementation, and Roadmap"
    )

    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run(
        f"\nPrepared: {datetime.now().strftime('%d %B %Y')}"
    )

    doc.add_page_break()

    # AI instructions
    doc.add_heading(
        "How AI Assistants Should Use This Document", level=1
    )

    doc.add_paragraph(
        "This document is intended to provide persistent project "
        "context to Claude, ChatGPT, GPT models, and other AI "
        "assistants. Use it as a technical reference when reviewing "
        "code, debugging, extending features, or preparing reports."
    )

    add_bullets(doc, [
        "Distinguish implemented features from prototypes and roadmap items.",
        "Never invent experimental metrics or dataset details.",
        "Use the uploaded source code as the authority for implementation.",
        "Explain code changes with filenames and verification steps.",
        "Preserve the human-in-the-loop QA safety approach.",
        "Do not claim GMP validation or pharmaceutical certification.",
    ])

    # Executive summary
    doc.add_heading("1. Executive Summary", level=1)

    doc.add_paragraph(
        "PharmaSight AI is a computer-vision and interaction-oriented "
        "research prototype designed to explore pharmaceutical "
        "visual inspection, PPE awareness, gesture-based interaction, "
        "molecular visualization, and QA-oriented decision support."
    )

    add_table(doc, ["Module", "Description"], [
        ("Module 1", "Hand Gesture and Air Drawing"),
        ("Module 2", "Molecular Visualization"),
        ("Module 3", "Capsule & Tablet Insights"),
        ("Module 4", "PPE / Mask Detection"),
        ("Module 5", "Tablet Detection and Inspection"),
    ])

    # Project background
    doc.add_heading("2. Project Background and Objectives", level=1)

    doc.add_paragraph(
        "The project combines interactive computer vision with "
        "pharmaceutical quality-awareness workflows. It aims to "
        "demonstrate real-time visual interaction and prototype "
        "inspection assistance while retaining human review."
    )

    add_bullets(doc, [
        "Explore real-time webcam-based computer vision.",
        "Develop tablet localization and abnormality classification.",
        "Implement PPE mask detection and compliance awareness.",
        "Enable gesture-controlled air drawing and visualization.",
        "Explore molecular visualization interaction.",
        "Develop conservative QA interpretation and review logic.",
    ])

    # Development pathway
    doc.add_heading("3. Development Pathway", level=1)

    add_numbered(doc, [
        "Project foundation, scope definition, and environment setup.",
        "Dataset collection, validation, and manifest generation.",
        "Duplicate detection and image quality auditing.",
        "Tablet binary classification development.",
        "Tablet localization using YOLO.",
        "PillQC multiclass experimentation.",
        "PPE dataset preparation and detector training.",
        "PPE compliance gate experiments.",
        "Webcam application and specialist module integration.",
        "Gesture interaction, voice introductions, and interface improvements.",
        "Evaluation, documentation, and future validation planning.",
    ])

    # Architecture
    doc.add_heading("4. System Architecture", level=1)

    add_code(doc, """
Webcam / OpenCV
       |
       v
PharmaSightLive Application
       |
       +-- DRAW       -> MediaPipe Hands -> Air Drawing
       |
       +-- MOLECULE   -> Gesture Interaction -> Molecular Visualization
       |
       +-- CAPSULE    -> Capsule Visualization / Insights
       |
       +-- PPE        -> YOLO Mask Detection -> Compliance Gate
       |
       +-- TABLET     -> YOLO Localization -> ResNet18 Classification
       |
       v
QA Interpretation / Manual Review / Visual Output
""")

    add_table(doc, ["Layer", "Primary responsibility"], [
        ("Camera", "Webcam acquisition and frame processing"),
        ("Application", "Mode routing, HUD, keyboard, lifecycle"),
        ("Gesture", "Hand landmarks and drawing interaction"),
        ("PPE", "Mask detection and temporal gate"),
        ("Tablet", "Tablet localization and binary classification"),
        ("QA", "Conservative interpretation and review prompts"),
        ("Evaluation", "Metrics, confusion matrices, audits"),
    ])

    # Modules
    doc.add_heading("5. Detailed Module Documentation", level=1)

    modules = [
        (
            "Module 1 — Hand Gesture and Air Drawing",
            "DRAW mode, keyboard D. Uses MediaPipe hand tracking, "
            "landmark visualization, gesture classification, drawing "
            "canvas, shape snapping, and canvas interactions."
        ),
        (
            "Module 2 — Molecular Visualization",
            "MOLECULE mode, keyboard V. Provides interactive molecular "
            "visualization, including aspirin-oriented interaction. "
            "It is a visualization experience, not a chemistry simulator."
        ),
        (
            "Module 3 — Capsule & Tablet Insights",
            "CAPSULE mode, keyboard K. Provides capsule visualization "
            "and interaction. Automated blister completeness inspection "
            "is a separate roadmap capability."
        ),
        (
            "Module 4 — PPE Mask Detection",
            "PPE mode, keyboard P. Uses YOLOv8 detection for with_mask, "
            "without_mask, and mask_weared_incorrect, followed by a "
            "temporal compliance gate."
        ),
        (
            "Module 5 — Tablet Detection",
            "TABLET mode, keyboard T. YOLO detects tablet regions; "
            "ResNet18 classifies crops as normal or defective. "
            "Uncertain outputs require manual review."
        ),
    ]

    for title, description in modules:
        doc.add_heading(title, level=2)
        doc.add_paragraph(description)

    # Dataset methodology
    doc.add_heading("6. Dataset Management and Preparation", level=1)

    add_numbered(doc, [
        "Preserve original datasets in immutable raw folders.",
        "Validate file readability and image dimensions.",
        "Build dataset manifests and record provenance.",
        "Verify labels against original source annotations.",
        "Identify exact and near-duplicate images.",
        "Group splits by physical sample or capture session.",
        "Audit class imbalance and image quality.",
        "Maintain separate training, validation, and test splits.",
        "Document licensing and redistribution constraints.",
    ])

    doc.add_paragraph(
        "The dataset_manifest.csv file is an inventory and metadata "
        "record. Suggested labels derived from folders must be "
        "verified manually before being treated as ground truth."
    )

    # Models
    doc.add_heading("7. Model Architectures and Training", level=1)

    add_table(doc, ["Pipeline", "Model / approach"], [
        ("Tablet classification", "ResNet18 binary classifier"),
        ("PillQC classification", "ResNet18 multiclass classifier"),
        ("Tablet localization", "YOLOv8n object detector"),
        ("PPE mask detection", "YOLOv8n object detector"),
        ("Gesture interaction", "MediaPipe Hand Landmarker"),
    ])

    doc.add_heading("7.1 Tablet Binary Classification", level=2)
    doc.add_paragraph(
        "The binary classifier uses normal and defective classes. "
        "Tablet detector boxes are expanded and cropped, then "
        "preprocessed to 224x224 with ImageNet normalization. "
        "The model returns class probabilities including P(defective)."
    )

    doc.add_heading("7.2 PPE Detection and Gate", level=2)
    doc.add_paragraph(
        "The PPE detector identifies mask status. The temporal gate "
        "aggregates frame-level detections and can return compliant, "
        "not compliant, uncertain, or no face. Gate behavior is "
        "conservative and should be validated on unseen sessions."
    )

    # Evaluation
    doc.add_heading("8. Evaluation and Results", level=1)

    doc.add_paragraph(
        "The following values are reported in the supplied project "
        "evidence. They are experimental results and must not be "
        "presented as production guarantees."
    )

    add_table(doc, ["Evaluation", "Reported result"], [
        ("Binary source-balanced test", "250 images; accuracy 0.756"),
        ("Binary defective recall", "0.5748"),
        ("Binary false-negative rate", "0.4252"),
        ("Final binary threshold test", "False-pass rate 0.44"),
        ("PillQC multiclass", "50 test images; reported metrics 1.00"),
        ("Tablet webcam baseline", "101 images; decisive accuracy 0.7553"),
        ("Tablet webcam defective recall", "0.8947"),
        ("PPE gate experiment", "False-accept rates varied by sensitivity"),
    ])

    doc.add_paragraph(
        "The binary model's false-pass rate is a significant limitation. "
        "It must not automatically approve pharmaceutical samples. "
        "The PillQC perfect score is based on a small test population "
        "and requires independent-source validation."
    )

    # QA
    doc.add_heading("9. QA Decision Support and Safety", level=1)

    add_bullets(doc, [
        "Uncertain model predictions should trigger manual review.",
        "No visible defect is not equivalent to proof of quality.",
        "Possible defect factors are investigation prompts, not causes.",
        "AI outputs cannot replace SOPs or qualified QA decisions.",
        "The prototype is not GMP-validated or certified.",
        "Automatic batch release must not be based on model outputs.",
    ])

    # Voice
    doc.add_heading("10. Voice Assistant and User Experience", level=1)

    doc.add_paragraph(
        "The voice subsystem uses a separate Python environment "
        "and Kokoro-based narration. Welcome audio and module "
        "introductions are generated or pre-generated and played "
        "asynchronously. Module transitions stop or queue audio "
        "to prevent overlapping narration."
    )

    add_table(doc, ["Key", "Mode"], [
        ("D", "Hand Gesture / Air Drawing"),
        ("V", "Molecular Visualization"),
        ("K", "Capsule Insights"),
        ("P", "PPE Detection"),
        ("T", "Tablet Detection"),
    ])

    # Repo structure
    doc.add_heading("11. Repository Structure", level=1)

    add_code(doc, """
PharmaSight-AI/
├── config/
├── docs/
├── src/
│   ├── camera/
│   ├── hand_gesture/
│   ├── modules/
│   ├── ppe_mask/
│   ├── training/
│   ├── evaluation/
│   ├── data/
│   └── voice/
├── PharmaVision-Module2-PPE-Mask/
├── datasets/
├── models/
├── runs/
├── reports/
├── notebooks/
└── tests/
""")

    # Installation
    doc.add_heading("12. Environment and Installation", level=1)

    add_code(doc, """
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt
python -m pip install -r PharmaVision-Module2-PPE-Mask\\requirements_ppe_mask.txt
python -m pytest
python webcam_test.py
python src\\camera\\pharmasight_live.py
""")

    doc.add_paragraph(
        "The project requirements include PyTorch, torchvision, "
        "OpenCV, NumPy, pandas, scikit-learn, Matplotlib, "
        "Albumentations, PyYAML, Pillow, tqdm, pytest, Jupyter, "
        "and Streamlit. Optional modules require additional packages."
    )

    # Risks
    doc.add_heading("13. Limitations and Risks", level=1)

    add_table(doc, ["Limitation", "Implication"], [
        ("False-pass tablet predictions", "Do not automatically approve samples."),
        ("Dataset source bias", "External validation is necessary."),
        ("Class imbalance", "Rare defects may be poorly learned."),
        ("PillQC test size", "Perfect metrics do not establish generalization."),
        ("Webcam domain shift", "Lighting and camera differences affect results."),
        ("Incomplete blister module", "Automated blister inspection remains unfinished."),
        ("Missing model weights", "Uploaded documentation archives are not a complete runnable release."),
        ("Regulatory scope", "Prototype is not a validated QA instrument."),
    ])

    # Roadmap
    doc.add_heading("14. Future Development Roadmap", level=1)

    add_numbered(doc, [
        "Reproduce all training and evaluation runs from versioned configurations.",
        "Reduce tablet false-pass rate through validation-driven threshold tuning.",
        "Perform independent-source and physical-sample-separated testing.",
        "Expand webcam data across tablet types, lighting, and sessions.",
        "Close PPE false-accept validation using unseen sessions.",
        "Implement the blister inspection pipeline.",
        "Improve UI recovery, logging, and model lifecycle handling.",
        "Conduct QA expert review and usability testing.",
        "Prepare model cards, dataset cards, and a reproducible release.",
    ])

    # AI handoff
    doc.add_heading("15. AI Assistant Handoff Instructions", level=1)

    doc.add_paragraph(
        "When continuing this project, another AI assistant should "
        "first inspect the actual repository files and compare them "
        "against this document. The document is context, not a "
        "substitute for source code."
    )

    add_bullets(doc, [
        "Do not invent files, metrics, dataset sources, or model results.",
        "Use actual code as the authority for implementation details.",
        "Keep implemented, experimental, and planned features separate.",
        "Explain all modifications and provide verification commands.",
        "Preserve human review and conservative QA semantics.",
        "Report metrics with sample counts, split details, and thresholds.",
        "Never claim GMP certification or automatic pharmaceutical release.",
    ])

    doc.add_heading("16. Source Files and Evidence Reference", level=1)

    add_bullets(doc, [
        "README.md and CHANGELOG.md",
        "docs/PROJECT_SCOPE_AND_ROADMAP.md",
        "docs/DATA_COLLECTION_PROTOCOL.md",
        "docs/VALIDATION_PLAN.md",
        "src/camera/pharmasight_live.py",
        "src/camera/live_tablet_inspection.py",
        "src/hand_gesture/air_canvas_live.py",
        "src/modules/qa_engine.py",
        "src/modules/tablet_inspection.py",
        "src/training/",
        "src/evaluation/",
        "src/data/",
        "PharmaVision-Module2-PPE-Mask/",
        "reports/evaluation/",
        "reports/dataset_audit/",
        "reports/training/",
        "runs/ppe_mask/",
        "runs/tablet_detection/",
        "datasets/dataset_manifest.csv",
    ])

    doc.add_heading("17. Document Maintenance", level=1)

    doc.add_paragraph(
        "Update this document whenever major modules, datasets, "
        "training configurations, checkpoints, evaluation protocols, "
        "or deployment assumptions change. Record the source commit, "
        "document revision, and date for every substantial update."
    )

    # Save
    doc.save(OUTPUT_FILE)

    print(f"Created successfully: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_document()
