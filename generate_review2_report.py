from fpdf import FPDF


class ReviewReportPDF(FPDF):

  def header(self):
    if self.page_no() > 1:
      self.set_font("Helvetica", "I", 8)
      self.set_text_color(120, 130, 140)
      self.cell(
          0,
          8,
          "B.Tech Project Review II - AI-Based Network Traffic Classifier",
          0,
          0,
          "L",
      )
      self.cell(0, 8, f"Page {self.page_no()}", 0, 1, "R")
      self.set_draw_color(210, 215, 220)
      self.line(10, 16, 200, 16)
      self.ln(6)

  def footer(self):
    self.set_y(-14)
    self.set_font("Helvetica", "I", 8)
    self.set_text_color(150, 150, 150)
    self.cell(
        0, 10, "Department of Computer Science & Engineering", 0, 0, "C"
    )

  def chapter_title(self, title):
    self.set_font("Helvetica", "B", 13)
    self.set_text_color(15, 30, 65)
    self.cell(0, 8, title, 0, 1, "L")
    self.set_draw_color(56, 189, 248)
    self.set_line_width(0.5)
    self.line(10, self.get_y(), 200, self.get_y())
    self.ln(4)

  def section_sub(self, title):
    self.set_font("Helvetica", "B", 10)
    self.set_text_color(40, 60, 90)
    self.cell(0, 6, title, 0, 1, "L")
    self.ln(1)

  def body_paragraph(self, text):
    self.set_font("Helvetica", "", 9.5)
    self.set_text_color(50, 50, 50)
    self.multi_cell(0, 5, text)
    self.ln(3)

  def key_val(self, key, value):
    self.set_font("Helvetica", "B", 9.5)
    self.set_text_color(30, 45, 70)
    self.cell(42, 5, f"{key}:", 0, 0, "L")
    self.set_font("Helvetica", "", 9.5)
    self.set_text_color(60, 60, 60)
    self.multi_cell(0, 5, value)
    self.ln(1)


pdf = ReviewReportPDF(orientation="P", unit="mm", format="A4")
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Title Section
pdf.set_font("Helvetica", "B", 18)
pdf.set_text_color(15, 30, 65)
pdf.cell(
    0, 10, "AI-BASED NETWORK TRAFFIC & VPN CLASSIFIER", 0, 1, "C"
)
pdf.set_font("Helvetica", "B", 11)
pdf.set_text_color(56, 189, 248)
pdf.cell(
    0,
    6,
    "REVIEW II EVALUATION & IMPLEMENTATION REPORT",
    0,
    1,
    "C",
)
pdf.ln(6)

# Metadata Box
pdf.set_fill_color(248, 250, 252)
pdf.set_draw_color(226, 232, 240)
pdf.rect(10, pdf.get_y(), 190, 26, "DF")
pdf.set_y(pdf.get_y() + 2)
pdf.key_val("Project Scope", "Encrypted & Non-VPN Network Flow Classification")
pdf.key_val(
    "Architecture", "Random Forest Ensemble (100 Trees) + Streamlit UI"
)
pdf.key_val(
    "Target Classes", "14 Network Applications (7 Standard, 7 Encrypted VPN)"
)
pdf.key_val("Holdout Test Acc.", "91.0% (11,942 holdout flows) | Batch Acc: 97.62%")
pdf.ln(5)

# 1. Project Abstract & Motivation
pdf.chapter_title("1. Problem Statement & Motivation")
pdf.body_paragraph(
    "Modern internet protocols heavily adopt end-to-end payload encryption"
    " (TLS 1.3, HTTPS, QUIC, and VPN tunnels like WireGuard/OpenVPN)."
    " Consequently, traditional Deep Packet Inspection (DPI) methods fail"
    " because packet contents are unreadable. This project builds a high-speed,"
    " machine learning classification engine that categorizes network sessions"
    " entirely through statistical flow metadata (inter-arrival intervals,"
    " volume dynamics, and burst distributions) without decrypting or"
    " inspecting packet payloads."
)

# 2. System Architecture & Dataset
pdf.chapter_title("2. Dataset & Feature Engineering")
pdf.body_paragraph(
    "The model is trained on the benchmark ISCX VPN-nonVPN dataset, covering 14"
    " distinct application scenarios: BROWSING, CHAT, FT, MAIL, P2P, STREAMING,"
    " VOIP, and their corresponding VPN-tunneled variants. A total of 23"
    " statistical flow parameters are extracted per flow session:"
)

# Feature summary table
pdf.set_font("Helvetica", "B", 8.5)
pdf.set_fill_color(241, 245, 249)
pdf.cell(50, 6, "Feature Category", 1, 0, "L", fill=True)
pdf.cell(140, 6, "Included Statistical Metrics", 1, 1, "L", fill=True)

pdf.set_font("Helvetica", "", 8.5)
pdf.cell(50, 6, "Timing / Inter-Arrival", 1, 0, "L")
pdf.cell(140, 6, "duration, total_fiat, total_biat, min/max/mean fiat & biat", 1, 1, "L")
pdf.cell(50, 6, "Flow Rates & Volumetric", 1, 0, "L")
pdf.cell(140, 6, "flowPktsPerSecond, flowBytesPerSecond, min/max/mean/std flowiat", 1, 1, "L")
pdf.cell(50, 6, "Active & Idle Windows", 1, 0, "L")
pdf.cell(140, 6, "min, mean, max, and std deviation of active & idle intervals", 1, 1, "L")
pdf.ln(5)

# 3. System Implementation & UI
pdf.chapter_title("3. Implementation & Real-Time Dashboard")
pdf.body_paragraph(
    "A full-stack inference environment was constructed utilizing Python and"
    " Streamlit. The system supports two primary operating modes:"
)
pdf.body_paragraph(
    "1. Single Flow Inspection: Allows live parameter experimentation via"
    " canonical presets (VoIP call, Chat message, P2P transfer, VPN audio)."
    " Outputs predicted class, VPN security pill, and a confidence percentage"
    " derived from the voting consensus of 100 decision trees.\n"
    "2. Batch CSV Classification: Ingests batch network captures,"
    " automatically detects ground truth ('class1') to compute dynamic batch"
    " accuracy, plots class distribution charts, and exports annotated CSVs."
)

# Page Break for Evaluation
pdf.add_page()
pdf.chapter_title("4. Experimental Results & Performance Evaluation")
pdf.body_paragraph(
    "The classifier was validated on holdout benchmark samples across all 14"
    " classes. Performance was evaluated using Accuracy, Precision, Recall, and"
    " Macro F1-Score."
)

# Benchmark Table
pdf.set_font("Helvetica", "B", 8.5)
pdf.set_fill_color(241, 245, 249)
pdf.cell(50, 6, "Traffic Category", 1, 0, "L", fill=True)
pdf.cell(35, 6, "Precision", 1, 0, "C", fill=True)
pdf.cell(35, 6, "Recall", 1, 0, "C", fill=True)
pdf.cell(35, 6, "F1-Score", 1, 0, "C", fill=True)
pdf.cell(35, 6, "Support (Flows)", 1, 1, "C", fill=True)

pdf.set_font("Helvetica", "", 8.5)
perf_data = [
    ("VOIP / VPN-VOIP", "1.00 / 1.00", "0.75 / 1.00", "0.86 / 1.00", "7"),
    ("CHAT / VPN-CHAT", "1.00 / 1.00", "1.00 / 1.00", "1.00 / 1.00", "5"),
    ("FT / VPN-FT", "1.00 / 1.00", "1.00 / 1.00", "1.00 / 1.00", "6"),
    ("P2P / VPN-P2P", "1.00 / 1.00", "1.00 / 1.00", "1.00 / 1.00", "7"),
    ("MAIL / VPN-MAIL", "1.00 / 1.00", "1.00 / 1.00", "1.00 / 1.00", "6"),
    ("STREAMING / VPN-STREAMING", "1.00 / 1.00", "1.00 / 1.00", "1.00 / 1.00", "6"),
    ("BROWSING / VPN-BROWSING", "0.67 / 1.00", "1.00 / 1.00", "0.80 / 1.00", "5"),
    ("Macro Average Aggregate", "0.98", "0.98", "0.98", "42"),
]

for row in perf_data:
  if "Macro" in row[0]:
    pdf.set_font("Helvetica", "B", 8.5)
  pdf.cell(50, 5.5, row[0], 1, 0, "L")
  pdf.cell(35, 5.5, row[1], 1, 0, "C")
  pdf.cell(35, 5.5, row[2], 1, 0, "C")
  pdf.cell(35, 5.5, row[3], 1, 0, "C")
  pdf.cell(35, 5.5, row[4], 1, 1, "C")

pdf.ln(5)
pdf.body_paragraph(
    "Summary of Metrics:\n"
    "* Holdout Baseline Accuracy : 91.0% (across 11,942 flows)\n"
    "* Batch Evaluation Accuracy : 97.62% (41 of 42 correct flows)\n"
    "* Macro F1-Score            : 97.55% balanced performance across all classes"
)

# 5. Verification Cases
pdf.chapter_title("5. Review II Verification & Demo Test Cases")
pdf.body_paragraph(
    "Two specific test datasets were created to prove system robustness:\n"
    "* test_labeled_evaluation.csv: Contains 42 flows with ground truth"
    " ('class1'). Confirms active batch accuracy reporting of 97.62%.\n"
    "* test_unlabeled_live_traffic.csv: Contains pure statistical features"
    " simulating uninspected live network streams, validating automatic label"
    " assignment and metric fallback."
)

# 6. Future Scope (Review III)
pdf.chapter_title("6. Review III Scope (Deep Learning Transition)")
pdf.body_paragraph(
    "To address subtle boundary overlaps between short chat bursts and web"
    " browsing, the next milestone involves transitioning to deep sequence"
    " architectures:\n"
    "1. 1D Convolutional Neural Network (1D-CNN) for automated spatial pattern"
    " extraction.\n"
    "2. Bidirectional LSTM (BiLSTM) network to model bidirectional inter-packet"
    " timing dependencies.\n"
    "3. Benchmarking Matrix: Quantifying trade-offs between Random Forest and"
    " Deep Learning across Accuracy, Latency (ms), and Model Size (MB)."
)

output_path = "Review_II_Project_Report.pdf"
pdf.output(output_path)
print(f"[+] Successfully generated: '{output_path}'")