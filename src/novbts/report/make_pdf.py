#!/usr/bin/env python3
"""Render the full Phase-3 report (incl. technical obstacles) to PDF via fpdf2."""
from fpdf import FPDF

from novbts.paths import DOCS, ensure

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONTM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
INK = (20, 20, 20)
MUT = (90, 90, 90)
ACC = (30, 70, 140)


class PDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("D", "", 7); self.set_text_color(*MUT)
        self.cell(0, 6, "Giai đoạn 3 — Neural Operator cho marker displacement VBTS", align="L")
        self.cell(0, 6, f"tr. {self.page_no()}", align="R"); self.ln(8)
        self.set_text_color(*INK)

    def footer(self):
        pass


def setup():
    pdf = PDF(format="A4")
    pdf.add_font("D", "", FONT); pdf.add_font("D", "B", FONTB); pdf.add_font("M", "", FONTM)
    pdf.set_auto_page_break(True, margin=16)
    pdf.set_margins(18, 16, 18)
    return pdf


def _mc(pdf, h, t, w=0):
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(w, h, t, new_x="LMARGIN", new_y="NEXT")

def h1(pdf, t):
    pdf.ln(2); pdf.set_font("D", "B", 14); pdf.set_text_color(*ACC)
    _mc(pdf, 7, t); pdf.set_text_color(*INK); pdf.ln(1)

def h2(pdf, t):
    pdf.ln(1); pdf.set_font("D", "B", 11); pdf.set_text_color(*INK)
    _mc(pdf, 6, t); pdf.ln(0.5)

def body(pdf, t):
    pdf.set_font("D", "", 9.5); pdf.set_text_color(*INK)
    _mc(pdf, 5, t); pdf.ln(0.5)

def bullet(pdf, t):
    pdf.set_font("D", "", 9.5); pdf.set_text_color(*INK)
    pdf.set_x(pdf.l_margin)
    pdf.cell(5, 5, "•")
    pdf.multi_cell(0, 5, t, new_x="LMARGIN", new_y="NEXT")

def code(pdf, t):
    pdf.set_font("M", "", 8); pdf.set_fill_color(244, 244, 244); pdf.set_text_color(*INK)
    for line in t.split("\n"):
        pdf.cell(0, 4.4, "  " + line, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

def table(pdf, headers, rows, widths):
    pdf.set_font("D", "B", 8.5); pdf.set_fill_color(*ACC); pdf.set_text_color(255, 255, 255)
    for h, w in zip(headers, widths):
        pdf.cell(w, 6, h, border=0, align="C", fill=True)
    pdf.ln()
    pdf.set_text_color(*INK)
    for i, row in enumerate(rows):
        fill = i % 2 == 1
        pdf.set_fill_color(238, 242, 248)
        # row height: measure max lines
        pdf.set_font("D", "", 8.3)
        y0 = pdf.get_y(); x0 = pdf.get_x()
        # compute needed height
        heights = []
        for c, w in zip(row, widths):
            n = pdf.multi_cell(w, 4.6, str(c), dry_run=True, output="LINES")
            heights.append(len(n))
        rh = max(heights) * 4.6
        if y0 + rh > pdf.h - pdf.b_margin:
            pdf.add_page(); y0 = pdf.get_y(); x0 = pdf.get_x()
        for c, w in zip(row, widths):
            x = pdf.get_x(); y = pdf.get_y()
            pdf.multi_cell(w, rh, "", border=0, fill=fill, new_x="RIGHT", new_y="TOP")
            pdf.set_xy(x, y)
            pdf.multi_cell(w, 4.6, str(c), border=0, align="L", new_x="RIGHT", new_y="TOP")
            pdf.set_xy(x + w, y)
        pdf.set_xy(x0, y0 + rh)
    pdf.ln(2)


def build():
    pdf = setup(); pdf.add_page()
    # ---- title ----
    pdf.set_font("D", "B", 18); pdf.set_text_color(*ACC)
    _mc(pdf, 9, "Báo cáo Giai đoạn 3 — Hệ thống đầy đủ & tổng quát hóa")
    pdf.set_font("D", "", 11); pdf.set_text_color(*INK)
    _mc(pdf, 6, "Neural operator học trường dịch chuyển marker của VBTS như surrogate tốc độ cao cho RL")
    pdf.set_font("D", "", 9); pdf.set_text_color(*MUT)
    _mc(pdf, 5, "Ngày: 2026-06-15   ·   Ràng buộc: 1 GPU RTX 2000 Ada 16GB, 15GB RAM   ·   RQ1–RQ3 + FEM Gate 3 (§6c) + so sánh VBTS sim (§6d)")
    pdf.set_text_color(*INK); pdf.ln(3)

    # ---- 0 framing ----
    h1(pdf, "0. Khung & vai trò các thành phần")
    body(pdf, "Trade-off cốt lõi: neural operator thay solver lúc inference — kém chính xác hơn FEM một chút nhưng nhanh hơn nhiều bậc.")
    table(pdf, ["Thành phần", "Vai trò", "Đặc tính"], [
        ["Neural operator (FNO)", "Thay solver lúc inference (surrogate cho RL)", "nhanh, xấp xỉ — ĐÓNG GÓP"],
        ["Hertz–Mindlin (giải tích)", "GT huấn luyện hiện tại + validator", "exact closed-form, ~12000 fps"],
        ["PhysX FEM (Isaac Sim)", "GT vật lý thật + mốc tốc độ solver", "FEM, ~7 fps"],
    ], [42, 80, 52])
    body(pdf, "RQ1 độ chính xác · RQ2 tổng quát hóa · RQ3 tốc độ. Quyết định GO/PIVOT/NO-GO theo từng RQ.")

    # ---- 1 GT ----
    h1(pdf, "1. Ground truth")
    h2(pdf, "Hertz–Mindlin (giải tích)")
    body(pdf, "Hertz pháp tuyến (bán kính tiếp xúc a=√(Rd), áp suất, u_z exact) + Cattaneo–Mindlin tiếp tuyến (bán kính dính c=a(1−Q/μP)^(1/3), vành slip, full slip khi Q→μP). Nhãn mode từ g=Q/μP. Giả định: bán-không-gian đàn hồi tuyến tính, biến dạng nhỏ.")
    h2(pdf, "PhysX FEM (Isaac Sim, không TacEx, không IPC)")
    body(pdf, "Gel block deformable (lưới tetrahedral co-rotational, E=1e5 Pa, ν=0.45) + indentor cầu rigid; pin node đáy (kinematic target); đọc nodal_pos_w → marker grid. NORMAL (isaac_extract_groundtruth.py) + SHEAR (isaac_extract_shear.py — đã phá deadlock, xem mục 3c). GIPC không dùng vì nó vào Isaac qua TacEx — đã bỏ TacEx; PhysX deformable là FEM native.")

    # ---- 2 dataset ----
    h1(pdf, "2. Dataset")
    bullet(pdf, "Hertz–Mindlin (data/phase3_gt/): train 16k, val 2k, test_id 2k, test_slip 1k, + 8 OOD (radius/depth/material/friction/geometry/resolution). Marker 32×32.")
    bullet(pdf, "FEM normal (data/phase3_gt_fem/): 40 frame normal-only, marker 24×24, solve-time log.")
    bullet(pdf, "FEM shear (data/fem_gt_shear.npz): 40 frame có kéo tiếp tuyến, drive-ratio g 0.04–1.28 (stick/partial/full), marker 24×24.")
    pdf.ln(1)

    # ---- 3 validation ----
    h1(pdf, "3. Kiểm chứng ground truth (validate_gt.py)")
    table(pdf, ["Kiểm tra", "Kết quả"], [
        ["Hertz–Mindlin: contact radius a (qua u_z(a)=0.5·u_z(0))", "rel err 1.3% (res32) / 0.34% (res64)  ✓"],
        ["Hertz–Mindlin: convergence peak |u_z| (res32 vs 64)", "0.53% (<5%)  ✓"],
        ["Hertz–Mindlin: stick radius c (đọc field)", "48%→32% (res32→64); c enforce exact khi sinh"],
        ["FEM vs Hertz: contact radius", "lệch ~37% → hiệu ứng gel dày/rộng HỮU HẠN mà half-space bỏ qua"],
        ["FEM: peak|u_z|/depth", "~1.16 (hợp lý: lún + phồng quanh tiếp xúc)"],
    ], [95, 79])
    body(pdf, "→ Hertz–Mindlin chính xác ở giới hạn chuẩn; FEM lệch ~37% do hình học thật — định lượng đúng lý do cần FEM.")

    # ---- 3c FEM shear ----
    h1(pdf, "3c. FEM shear — slip nảy sinh tự ma sát (OPEN PROBLEM ĐÃ GIẢI QUYẾT)")
    h2(pdf, "Chẩn đoán nguyên nhân gốc")
    body(pdf, "Triệu chứng cũ: kéo indentor (rigid, kinematic_enabled=True) trượt ngang KHI ĐÃ LÚN SÂU → PhysX treo >1h, GPU IDLE (kẹt thật, không phải chậm). Phase lún thẳng thì chạy bình thường. Nguyên nhân: body kinematic là VÔ HẠN CỨNG — solver buộc nó đi đúng pose áp đặt bất kể phản lực. Khi cắm sâu rồi ép NGANG qua khối tet → loạt ràng buộc tiếp xúc OVER-CONSTRAINED mà vòng lặp position-iteration của FEM không hội tụ: iteration đẩy node ra, ràng buộc kinematic kéo lại → luẩn quẩn.")
    h2(pdf, "Cách phá deadlock (isaac_extract_shear.py)")
    body(pdf, "Giữ indentor kinematic (không đổi sang drive lực/vận tốc — phức tạp), nhưng GIẢM cường độ over-constraint bằng 3 thay đổi, mỗi cái nhắm đúng cơ chế kẹt:")
    table(pdf, ["Thay đổi", "Vì sao hiệu quả"], [
        ["1. Lún NÔNG ~5mm cho frame shear", "tiếp xúc nhẹ → ít ràng buộc xung đột; 'deep contact' là điều kiện cần của deadlock cũ"],
        ["2. Kéo ngang 60 bước CỰC NHỎ + settle mỗi bước", "≈velocity control: mỗi bước xê dịch ràng buộc một chút → solver bám kịp, không phải giải cú nhảy lớn"],
        ["3. solver_position_iteration_count=30, contact_offset=0.002", "nhiều iteration hơn để hội tụ contact; offset rộng để bắt tiếp xúc sớm, mượt"],
    ], [70, 104])
    body(pdf, "→ Mỗi micro-step ổn định ~0.022s, KHÔNG kẹt; smoke 1 frame + sweep 40 frame đều chạy trơn. Theo dõi qua fem_progress.txt (Isaac nuốt stdout); kill container sau khi xong (app.close() treo).")
    h2(pdf, "Kết quả — tín hiệu slip định tính (đánh giá TRUNG THỰC)")
    body(pdf, "Slip KHÔNG bị áp đặt (như Cattaneo–Mindlin) mà nảy từ contact ma sát của solver. Tín hiệu CHẮC CHẮN nhất: dịch chuyển tiếp tuyến bề mặt BÃO HÒA ~0.85±0.26mm dù indentor kéo ngang tới 7.65mm (travel trải 29×; corr(peak_tang,travel)=−0.47). → indentor TRƯỢT trên gel chứ không kéo gel đi mãi — đúng bản chất slip. Contact giữ vững (peak|uz| 6.5mm > lún 5.5mm). Solve 1.57s/frame ≈ 0.64 fps (chủ yếu do 240 step/frame; per-step ~0.022s như normal) → mốc solver RQ3.")
    body(pdf, "KHÔNG over-claim 'khớp Cattaneo–Mindlin' — soi kỹ cho thấy validation định lượng CHƯA đứng vững:")
    table(pdf, ["Hạn chế", "Hệ quả"], [
        ["Lưới deformable thô (~605 node → ~5 node ngang vết tiếp xúc, cần >=8-10)",
         "trường tiếp tuyến under-resolved; KHÔNG đọc được bán kính dính c; mịn lưới hơn → cooking treo"],
        ["Trường radial của lún tạo sàn ~0.85mm trên peak_tang",
         "ρ=peak_tang/travel giảm theo g phần lớn là ARTIFACT chia cho travel∝g, không phải dấu hiệu slip"],
        ["Một điểm vận hành (1 R,μ,E), depth 4–7mm, n=40", "chưa phủ tham số; không kết luận tổng quát"],
    ], [70, 104])
    body(pdf, "→ Chứng minh METHOD/PIPELINE shear chạy được (deadlock giải quyết + tín hiệu slip thô từ ma sát), CHƯA phải GT slip độ-phân-giải-cao đã validate định lượng. Nâng cấp: lưới mịn hơn, quét R/μ, tách radial khỏi tiếp tuyến. Dữ liệu: data/fem_gt_shear.npz. Validate: validate_shear.py.")

    # ---- 3d convergence + PhysX stability ceiling ----
    h1(pdf, "3d. Hội tụ lưới & trần ổn định PhysX (vs IPC/TacEx)")
    body(pdf, "'Lưới mịn có chính xác hơn không' KHÔNG đo bằng Cattaneo (half-space = sai thước đo) mà bằng CONVERGENCE: cố định geometry, đổi mesh, xem nghiệm hội tụ. Ở gel 50×50×2mm (field simulation_hexahedral_resolution):")
    table(pdf, ["hex-res", "nodes", "peak_uz", "trạng thái"], [
        ["24", "1250", "0.700mm", "ổn định"],
        ["32", "2196", "0.568mm", "ổn định"],
        ["48", "7248", "0.551mm", "ổn định (hội tụ)"],
        ["64", "16926", "(nổ)", "phân kỳ — kể cả DT 5ms→1ms"],
    ], [26, 30, 34, 84])
    bullet(pdf, "peak_uz HỘI TỤ 0.70→0.57→0.55mm (số gia nhỏ dần) → lưới mịn cho nghiệm pháp tuyến đáng tin; lưới thô đo hụt. (Tiếp tuyến hội tụ chậm hơn — qua bề dày mới 1–3 phần tử.)")
    bullet(pdf, "'68% lệch Cattaneo' của bản mịn KHÔNG phải kém chính xác — do thước đo sai (half-space ≠ gel hữu hạn) + confound kích thước gel (đã sửa: cùng 50×50×2mm).")
    bullet(pdf, "Trần ổn định PhysX (ĐIỀU KIỆN ~ element_size/DT): gel nhỏ mọi chiều (5×5×2mm) NỔ; lưới quá mịn (res64) cũng NỔ dù giảm DT → PhysX có trần mịn (~res48).")
    bullet(pdf, "TacEx KHÔNG vướng: gel biến dạng dùng IPC (sapienipc.IPCSystem / UIPC) — ổn định VÔ ĐIỀU KIỆN; đường PhysX của TacEx chỉ là gelpad RIGID. Nổ ở gel nhỏ/mịn = đúng cái giá của việc chọn PhysX deformable thay IPC/GIPC (bỏ TacEx).")

    # ---- 3b framing ----
    h1(pdf, "3b. Framing là quyết định — headline phải là field→field")
    body(pdf, "Cùng vật lý, 2 cách đóng gói input, và CHÍNH cách này quyết định FNO thắng hay thua baseline:")
    bullet(pdf, "param→field (CŨ): input = vector 9 số → field 32×32. Mỗi điểm được 'mớm' đủ 9 params → MLP coordinate giải CỤC BỘ được → FNO mất lợi thế.")
    bullet(pdf, "field→field (HEADLINE): input = field bản đồ lún 32×32 (penetration + shear·mask) + 2 scalar (mu,E) → field chuyển vị. Điểm ngoài tiếp xúc nhận penetration=0 → MLP per-point KHÔNG biết tiếp xúc ở đâu. Chuyển vị là hàm PHI CỤC BỘ (Green's function) → chỉ operator (FNO) giải được.")
    body(pdf, "→ Toàn bộ RQ1–RQ3 dưới đây chạy trong framing field→field (phase3_field2field_full.py, 16k train/40 epoch). Framing param→field cũ chuyển xuống §4b ablation phản chứng.")

    # ---- 4 RQ1 ----
    h1(pdf, "4. RQ1 — Độ chính xác (test_id, field→field)")
    table(pdf, ["Model", "params", "rel L2", "normal", "stick", "partial", "full", "dir err"], [
        ["MLP (per-point)", "134K", "0.743", "0.782", "0.759", "0.661", "0.807", "62.8°"],
        ["FNO (operator)", "2.67M", "0.111", "0.090", "0.091", "0.123", "0.147", "4.2°"],
        ["FNO+head a", "2.67M", "0.109", "0.093", "0.092", "0.114", "0.143", "3.8°"],
    ], [34, 18, 18, 20, 18, 20, 16, 20])
    bullet(pdf, "FNO THẮNG MLP 6.7× (0.111 vs 0.743) — setup operator đúng nghĩa, không phải artifact framing.")
    bullet(pdf, "test_slip (slip-only, khó hơn): FNO rel L2 0.162. MLP per-point sụp đổ (74% lỗi, hướng sai 63°) vì thiếu ngữ cảnh toàn cục.")
    bullet(pdf, "FNO trong dải Gate ~11%; full_slip khó nhất; direction error ~3.8–4.2° (xuất sắc).")
    h2(pdf, "Slip detection (mode-F1)")
    table(pdf, ["Head", "macro-F1", "normal", "stick", "partial", "full", "slip-binary"], [
        ["a (multitask, gắn FNO)", "0.985", "1.00", "1.00", "0.98", "0.96", "1.00"],
        ["b (classifier riêng)", "0.856", "0.98", "0.85", "0.79", "0.81", "0.94"],
    ], [46, 22, 20, 18, 20, 16, 24])
    body(pdf, "→ Cả hai vượt ngưỡng 0.75 (heuristic cũ Gate 3 chỉ 0.67). Multitask >> separate. ĐÓNG điều kiện slip của Gate 3.")

    # ---- 4b field2field ablation ----
    h1(pdf, "4b. Ablation framing — vì sao param→field gây hiểu lầm (phản chứng)")
    body(pdf, "Cùng pipeline, đổi cách đặt bài toán, kết quả lật ngược:")
    table(pdf, ["Framing", "FNO", "MLP", "Ghi chú"], [
        ["param→field (vector 9 số → field)", "0.079", "0.066", "MLP thắng — mỗi điểm được 'mớm' full params → giải cục bộ (artifact)"],
        ["field→field (bản đồ lún → field)", "0.111", "0.743", "FNO THẮNG 6.7× (headline §4)"],
    ], [54, 18, 18, 84])
    body(pdf, "Trong param→field, mọi điểm lưới nhận đủ 9 params (biết chính xác tâm/độ sâu) nên MLP chỉ cần khớp công thức cục bộ → không cần operator. Đây là lý do KHÔNG dùng param→field cho paper: nó che mất giá trị operator learning. Ablation modes 12→16 cho thấy low-pass góp phần nhỏ nhưng không đủ lật ngược — chỉ FRAMING mới lật được. Script: phase3_field2field.py (PoC) → phase3_field2field_full.py (đầy đủ).")

    # ---- 5 RQ2 ----
    h1(pdf, "5. RQ2 — Tổng quát hóa (FNO, OOD, field→field)")
    table(pdf, ["OOD split", "rel L2", "degradation"], [
        ["deep_indent", "0.075", "0.67× (tốt hơn!)"],
        ["large_radius", "0.107", "0.96×"],
        ["res64 (upsample)", "0.112", "1.01×"],
        ["soft_material", "0.133", "1.19×"],
        ["low_friction", "0.138", "1.24×"],
        ["small_radius", "0.205", "1.84×"],
        ["flat_geom (hình học chưa train)", "0.690", "6.21×"],
        ["res16", "—", "không eval được (FNO modes=12 > grid 16)"],
    ], [60, 30, 84])
    body(pdf, "→ Tổng quát tốt với OOD tham số (<2×) + bất biến phân giải lên (res64 1.01×). flat_geom chỉ 6.2× — GIẢM MẠNH từ 19× của param→field: mã hóa hình học vào field lún (thay one-hot geom) tổng quát tốt hơn nhiều. Vẫn không xuống dưới phân giải mode (res16).")

    # ---- 6 RQ3 ----
    h1(pdf, "6. RQ3 — Tốc độ (field→field)")
    table(pdf, ["Hệ", "throughput", "/ frame"], [
        ["FNO inference", "8087 fps", "0.124 ms"],
        ["FNO+slip(a)", "8031 fps", "0.124 ms"],
        ["MLP inference", "11780 fps", "0.085 ms"],
        ["PhysX FEM solver", "7.2 fps", "139 ms"],
        ["Hertz–Mindlin analytic", "13114 fps", "0.076 ms"],
    ], [70, 50, 54])
    body(pdf, "→ FNO nhanh hơn FEM solver ≈ 1123× (8087/7.2). RQ3 speedup chỉ có nghĩa khi đối chiếu SOLVER CHẬM (FEM), không phải công thức analytic (vốn nhanh hơn cả FNO). Biểu đồ: runs/phase3_f2f_full/fidelity_speed.png.")

    # ---- 6c FEM Gate 3 ----
    pdf.add_page()
    h1(pdf, "6c. RQ1–RQ3 trên GT FEM THẬT — đóng Gate 3")
    body(pdf, "Toàn bộ RQ ở trên đứng trên GT analytic (lý tưởng hoá). Gate 3 yêu cầu làm lại trên VẬT LÝ THẬT. Ta train+đo lại trên tập FEM swept 2000 frame (res-24, quét R/μ/E) qua novbts.operator.fem_benchmark → runs/phase3_fem/benchmark.json.")
    table(pdf, ["Chỉ số", "analytic GT (16k)", "FEM GT (2000)"], [
        ["RQ1 FNO rel L2", "0.111", "0.146"],
        ["RQ1 MLP rel L2", "0.743", "0.328"],
        ["FNO thắng MLP", "6.7×", "2.24×"],
        ["RQ1 hướng tiếp tuyến (FNO)", "3.8°", "14.8°"],
        ["slip-F1 head-a (multitask)", "0.985", "0.904"],
        ["RQ3 FNO vs solver", "1123× (vs 7.2 fps)", "≈23.000× (vs 0.341 fps)"],
    ], [62, 56, 56])
    bullet(pdf, "Luận điểm phi-cục-bộ GIỮ trên vật lý thật: FNO thắng MLP 2.24× (hướng 35.7°→14.8°). Biên hẹp hơn analytic (6.7×) vì FEM nhiễu/ít lý tưởng.")
    bullet(pdf, "RQ1 per-mode FNO: normal 0.123 · stick 0.134 · partial 0.150 · full_slip 0.167 (sai số tăng dần theo slip).")
    bullet(pdf, "RQ2 ngoại suy đuôi R/μ/E: ~1.3× cả ba trục (mượt; ngoại suy TRONG hộp, chưa phải OOD ngoài-dải).")
    bullet(pdf, "RQ3 ≈23.000× so với solver PhysX-FEM shear thật (0.341 fps = 2.9s/frame).")
    body(pdf, "→ Gate 3 ĐÓNG trên GT vật lý thật: FNO là surrogate phi-cục-bộ thắng baseline, phân loại slip đạt ngưỡng, nhanh hơn solver ~4 bậc.")
    h2(pdf, "Cứu slip-classifier head-b bằng frame normal thuần")
    body(pdf, "Head-b ban đầu suy biến (normal F1=0, macro 0.595) vì sweep g∈[0,1.3] chỉ có 3.1% frame normal. Bổ sung 400 frame g=0 (mode normal) span cùng hộp R/μ/E (infra/gen_fem_normal_sweep.sh), merge+shuffle thành 2400 frame (normal 19.3%, data/fem/shear_fine_swept_normaug.npz).")
    table(pdf, ["Chỉ số", "committed (normal 3.1%)", "normaug (19.3%)"], [
        ["head-b normal F1", "0.0", "0.851"],
        ["head-b macro-F1", "0.595", "0.753 (>0.75)"],
        ["head-a normal F1", "0.769", "0.951"],
        ["FNO overall rel L2", "0.146", "0.144 (không suy giảm)"],
        ["FNO thắng MLP", "2.24×", "2.24×"],
    ], [56, 60, 58])
    body(pdf, "→ Xác nhận head-b suy biến do MẤT CÂN BẰNG LỚP, không phải lỗi mô hình; regression không suy giảm. Lưu: runs/phase3_fem/benchmark_normaug.json.")

    # ---- 6d VBTS baselines ----
    pdf.add_page()
    h1(pdf, "6d. So sánh với mô hình marker của các mô phỏng VBTS tiêu biểu")
    body(pdf, "Số liệu liên-paper KHÔNG so trực tiếp được (cảm biến khác, đầu ra ảnh RGB vs trường marker của ta). Nên ta cài lại LÕI mô hình chuyển động marker của từng phương pháp rồi fit + đo trên CHÍNH GT FEM (cùng data/split/metric, novbts.operator.vbts_baselines → runs/phase3_fem/vbts_baselines.json).")
    bullet(pdf, "TACTO (Wang RA-L 2022): động học, KHÔNG ma sát — kéo cứng cả vùng, không stick-slip.")
    bullet(pdf, "Cattaneo–Mindlin giải tích (vật lý first-principles): Hertz + Cattaneo-Mindlin, hiệu chỉnh affine per-channel (cơ hội công bằng nhất).")
    bullet(pdf, "Taxim (Si&Yuan RA-L 2022) + FOTS (Zhao 2023): đàn hồi tuyến tính chồng chập = một conv tuyến tính bất biến dịch (không phi tuyến).")
    bullet(pdf, "MLP per-point: học nhưng cục bộ (cận dưới). FNO (ours): operator phổ phi-cục-bộ.")
    table(pdf, ["Phương pháp (lõi marker)", "overall", "full", "hướng°", "FNO hơn"], [
        ["TACTO-style (động học, no friction)", "0.504", "0.625", "65.8", "3.51×"],
        ["Cattaneo–Mindlin giải tích (calib)", "0.435", "0.604", "39.0", "3.03×"],
        ["Taxim/FOTS-style (tuyến tính chồng chập)", "0.295", "0.363", "26.2", "2.05×"],
        ["MLP per-point (cục bộ)", "0.321", "0.467", "33.6", "2.24×"],
        ["FNO (ours)", "0.144", "0.168", "14.6", "—"],
    ], [78, 22, 22, 24, 28])
    bullet(pdf, "TACTO-style sụp trên tiếp tuyến (hướng 65.8°≈ngẫu nhiên): không tái tạo được stick-slip → FNO hơn 3.51×.")
    bullet(pdf, "Vật lý giải tích kinh điển (Cattaneo–Mindlin) dù đã hiệu chỉnh biên độ chỉ đạt 0.435 — TỆ HƠN cả mô hình tuyến tính fit-data (0.295): affine chỉ chỉnh biên độ, không đổi được hình dạng profile; trường FEM lệch khỏi Hertz–Mindlin lý tưởng. FNO hơn 3.03×.")
    bullet(pdf, "Taxim/FOTS-style là baseline MẠNH NHẤT (0.295) nhưng FNO vẫn hơn 2.05× vì chuyển stick→partial→full slip là PHI TUYẾN — mô hình tuyến tính không biểu diễn được.")
    bullet(pdf, "MLP cục bộ tốt ở normal/stick nhưng sụp ở partial/full (cần ngữ cảnh toàn cục). FNO thắng ở MỌI mode, cách biệt lớn nhất ở slip.")
    body(pdf, "Trung thực: cài lại LÕI marker (không phải renderer quang học đầy đủ); chạy trọng số hiệu-chỉnh-sẵn của bản gốc lên gel FEM của ta = sai cảm biến, KHÔNG công bằng hơn. Mục đích: cô lập giá trị của việc mô hình hóa trường slip phi tuyến/phi-cục-bộ mà các sim VBTS tuyến tính/động học/giải tích không nắm được.")

    # ---- 7 obstacles ----
    pdf.add_page()
    h1(pdf, "7. VƯỚNG MẮC KỸ THUẬT (đã gặp & cách xử lý)")
    h2(pdf, "A. Huấn luyện / mô hình")
    table(pdf, ["Vấn đề", "Triệu chứng", "Cách xử lý"], [
        ["torch bản CPU", "cuda=False dù có GPU", "Cài lại torch 2.11+cu126 (force-reinstall, kéo nvidia cudnn/cublas)"],
        ["Thiếu chuẩn hóa", "rel L2 ~0.31 (rất cao)", "Chuẩn hóa params per-dim + disp per-channel → 0.066 (giảm 5×)"],
        ["flat_geom nổ 3.3 triệu lần", "rel L2 = 261857 ở OOD geom", "Cột 'geom' hằng trong train (std≈0) → chuẩn hóa làm OOD value=1/eps; fix: cột hằng pass-through (mean0,std1)"],
        ["FNO modes vượt lưới", "einsum crash modes=20", "rfft 32-grid chỉ 17 tần số/trục → modes ≤ 16"],
        ["res16 không eval FNO", "RuntimeError", "FNO modes=12 > grid 16 (giảm phân giải dưới mode → không khả thi, đã log trung thực)"],
        ["Multitask phá field", "rel L2 0.61 khi λ_cls=0.5", "Hạ λ_cls=0.1 → field 0.078 + slip-F1 0.975 (cân bằng)"],
    ], [40, 52, 82])

    h2(pdf, "B. Isaac Sim / FEM (Docker)")
    table(pdf, ["Vấn đề", "Triệu chứng", "Cách xử lý"], [
        ["isaaclab không import", "ModuleNotFoundError", "Editable install thiếu; thêm /workspace/isaaclab/source/* vào sys.path"],
        ["Thiếu dependency flatdict", "ImportError trong isaaclab.app", "Build image dẫn xuất isaac-lab-fem (pip install -e isaaclab core)"],
        ["Isaac nuốt stdout", "log đứng ở 32s → tưởng treo", "Ghi tiến trình ra file mount (fem_progress.txt), KHÔNG theo stdout"],
        ["app.close() treo", "container không thoát dù xong việc", "Lưu npz incremental + waiter tự kill container khi đủ frame"],
        ["Container mồ côi tranh RAM", "2 Isaac đồng thời treo cả giờ", "Chỉ chạy 1 container/lúc (15GB RAM); kill theo ID"],
        ["Bug pin node đáy", "|u_z|/depth=2.6 (gel trôi, phi lý)", "Cờ kinematic ngược: 1.0=pinned (không phải 0.0) → fix → ratio 1.16"],
        ["Shear/slip deadlock", "phase2 treo >1h, GPU idle", "ĐÃ GIẢI QUYẾT (mục 3c): lún nông + kéo micro-step + iters=30 → không kẹt"],
        ["Nhãn mode shear sai", "40 frame đều = normal", "label_mode nhận TRAVEL(m) thay drive-ratio; gán từ g sample + relabel npz"],
        ["npz container quyền root", "PermissionError ghi đè", "Ghi bản relabeled sang path user (data/fem_gt_shear.npz)"],
    ], [44, 52, 78])
    body(pdf, "Bài học vận hành Isaac headless: theo file log mount (không stdout); app.close() treo nên lưu incremental + kill chủ động; 1 container/lúc.")

    # ---- 8 open ----
    h1(pdf, "8. Vấn đề mở & hạn chế (trung thực)")
    bullet(pdf, "ĐÃ ĐÓNG: Gate 3 trên GT FEM thật (§6c — 2000 frame swept, FNO thắng MLP 2.24×, slip-F1 0.90, ~23.000× solver). FEM shear deadlock đã phá (§3c), đã mịn lưới + scale + quét tham số.")
    bullet(pdf, "Trần tiếp tuyến ~0.146 rel L2 / 14.8° là giới hạn NỘI TẠI (đã thử res-32 + nhiều Fourier modes: KHÔNG hạ trần → bác bỏ giả thuyết GT-fidelity). Đòn bẩy còn lại: đổi biểu diễn đầu vào (lịch sử tiếp xúc, marker mịn hơn) hoặc kiến trúc — không phải mesh.")
    bullet(pdf, "RQ2 trên FEM mới là ngoại suy TRONG hộp (R∈[15,25]mm, μ∈[0.4,0.8], E∈[0.5,2]e5), chưa phải OOD ngoài-dải (không sinh FEM ngoài hộp rẻ được). flat-geom OOD vẫn 6.2× (trục hình học).")
    bullet(pdf, "GT FEM (2400) và analytic (16k) khác thang đơn vị, train riêng — chưa hợp nhất (transfer learning là hướng để analytic bootstrap FEM).")
    bullet(pdf, "Hertz–Mindlin là half-space tuyến tính ≠ gel thật (lệch ~37%) — khe hở để Giai đoạn 4 sim-to-real đóng.")

    # ---- 9 gate ----
    h1(pdf, "9. Quyết định Gate (paper-scale)")
    table(pdf, ["RQ", "Phán quyết", "Ghi chú"], [
        ["RQ1 accuracy + slip", "GO", "FNO ~11% (analytic) / 0.146 (FEM); slip-F1 0.985/0.904 đóng Gate 3; head-b cứu 0.753"],
        ["RQ1 operator > baseline", "GO", "field→field FNO thắng MLP 6.7× (analytic) / 2.24× (FEM); hơn cả TACTO/Taxim/FOTS/Cattaneo-Mindlin 2.05–3.51× (§6d)"],
        ["RQ2 generalization", "một phần", "tốt param OOD (<2×) + res-up; FEM ngoại suy trong-hộp 1.3×; kém geometry (6.2×)"],
        ["RQ3 speed", "GO", "FNO ≈1123× (analytic solver) / ≈23.000× (FEM shear solver thật)"],
    ], [48, 30, 96])
    body(pdf, "Kết luận: Giai đoạn 3 đạt PROOF-OF-MACHINERY hoàn chỉnh + ĐÓNG GATE 3 TRÊN GT VẬT LÝ THẬT (§6c): pipeline field→field chạy, operator thắng baseline chính danh (6.7× analytic / 2.24× FEM) và thắng cả các mô hình marker của mô phỏng VBTS tiêu biểu 2.05–3.51× (§6d), slip-discontinuity giải quyết (head-a 0.90 + head-b cứu 0.75), nhanh hơn solver FEM ~23.000×. Còn lại trước paper: hợp nhất analytic↔FEM (transfer learning), cải thiện flat-geom OOD, task-level validation in-loop.")

    # ---- 10 assets ----
    h1(pdf, "10. Tài sản")
    body(pdf, "src/novbts/: groundtruth/{hertz_mindlin,data_gen,isaac_extract_normal,isaac_extract_shear,aggregate_sweep}.py · operator/{field2field (HEADLINE),param2field (ablation),eval_rq,fem_train_compare,fem_benchmark (§6c),vbts_baselines (§6d)}.py · validation/{validate_gt,validate_shear,compare_shear}.py · models.py · report/make_pdf.py · infra/{Dockerfile.fem,setup_isaac.sh,gen_fem_sweep.sh,gen_fem_normal_sweep.sh}")
    body(pdf, "runs/phase3_f2f_full/: results.json + fidelity_speed.png (HEADLINE analytic) · runs/phase3_fem/{benchmark.json, benchmark_normaug.json (head-b cứu), vbts_baselines.json (§6d)} · data/fem/{shear_fine_swept.npz, shear_fine_swept_normaug.npz (2400, +400 normal), normal.npz, shear_fine.npz, shear_coarse.npz}")

    ensure(DOCS)
    out = str(DOCS / "bao_cao_giai_doan3.pdf")
    pdf.output(out)
    print("WROTE", out)


if __name__ == "__main__":
    build()
