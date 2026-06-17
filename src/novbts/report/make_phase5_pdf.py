#!/usr/bin/env python3
"""Render the Phase-5 marker-dot sensor report to PDF via fpdf2."""
from __future__ import annotations

import json
from pathlib import Path

from fpdf import FPDF

from novbts.paths import DOCS, FEM, ROOT, RUNS, ensure

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONTM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

INK = (22, 24, 28)
MUT = (90, 96, 104)
ACC = (32, 92, 130)
SOFT = (232, 240, 245)


class PDF(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("D", "", 7)
        self.set_text_color(*MUT)
        self.cell(0, 6, "Giai đoạn 5 - Marker-dot VBTS sensor model", align="L")
        self.cell(0, 6, f"tr. {self.page_no()}", align="R")
        self.ln(8)
        self.set_text_color(*INK)

    def footer(self) -> None:
        pass


def setup() -> PDF:
    pdf = PDF(format="A4")
    pdf.add_font("D", "", FONT)
    pdf.add_font("D", "B", FONTB)
    pdf.add_font("M", "", FONTM)
    pdf.set_auto_page_break(True, margin=16)
    pdf.set_margins(18, 16, 18)
    return pdf


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text())


def mc(pdf: PDF, h: float, text: str, width: float = 0) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(width, h, text, new_x="LMARGIN", new_y="NEXT")


def h1(pdf: PDF, text: str) -> None:
    pdf.ln(2)
    pdf.set_font("D", "B", 14)
    pdf.set_text_color(*ACC)
    mc(pdf, 7, text)
    pdf.set_text_color(*INK)
    pdf.ln(1)


def h2(pdf: PDF, text: str) -> None:
    pdf.ln(1)
    pdf.set_font("D", "B", 11)
    pdf.set_text_color(*INK)
    mc(pdf, 6, text)


def body(pdf: PDF, text: str) -> None:
    pdf.set_font("D", "", 9.5)
    pdf.set_text_color(*INK)
    mc(pdf, 5, text)
    pdf.ln(0.5)


def bullet(pdf: PDF, text: str) -> None:
    pdf.set_font("D", "", 9.5)
    pdf.set_text_color(*INK)
    pdf.set_x(pdf.l_margin)
    pdf.cell(5, 5, "-")
    pdf.multi_cell(0, 5, text, new_x="LMARGIN", new_y="NEXT")


def code(pdf: PDF, text: str) -> None:
    pdf.set_font("M", "", 8)
    pdf.set_fill_color(246, 247, 248)
    for line in text.splitlines():
        pdf.cell(0, 4.5, "  " + line, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def table(pdf: PDF, headers: list[str], rows: list[list[object]], widths: list[float]) -> None:
    pdf.set_font("D", "B", 8.4)
    pdf.set_fill_color(*ACC)
    pdf.set_text_color(255, 255, 255)
    for head, width in zip(headers, widths):
        pdf.cell(width, 6, str(head), align="C", fill=True)
    pdf.ln()
    pdf.set_text_color(*INK)
    for i, row in enumerate(rows):
        pdf.set_font("D", "", 8.2)
        y0 = pdf.get_y()
        x0 = pdf.get_x()
        line_counts = [
            len(pdf.multi_cell(width, 4.5, str(cell), dry_run=True, output="LINES"))
            for cell, width in zip(row, widths)
        ]
        row_h = max(line_counts) * 4.5 + 1.0
        if y0 + row_h > pdf.h - pdf.b_margin:
            pdf.add_page()
            y0 = pdf.get_y()
            x0 = pdf.get_x()
        fill = i % 2 == 1
        pdf.set_fill_color(*SOFT)
        for cell, width in zip(row, widths):
            x = pdf.get_x()
            y = pdf.get_y()
            pdf.multi_cell(width, row_h, "", fill=fill, new_x="RIGHT", new_y="TOP")
            pdf.set_xy(x, y + 0.5)
            pdf.multi_cell(width, 4.5, str(cell), new_x="RIGHT", new_y="TOP")
            pdf.set_xy(x + width, y)
        pdf.set_xy(x0, y0 + row_h)
    pdf.ln(2)


def mm(value: float) -> str:
    return f"{value:.3f} mm"


def image_fit(pdf: PDF, path: Path, max_w: float, max_h: float) -> None:
    try:
        from PIL import Image
        w_px, h_px = Image.open(path).size
        scale = min(max_w / w_px, max_h / h_px)
        pdf.image(str(path), x=pdf.l_margin, w=w_px * scale, h=h_px * scale)
    except Exception:
        pdf.image(str(path), x=pdf.l_margin, w=max_w)


def build() -> Path:
    phase = RUNS / "phase5"
    build_rep = load_json(phase / "sensor_build.json")
    compat = load_json(phase / "sensor_compat.json")
    compare_rep = load_json(phase / "sensor_compare.json")
    inv = load_json(phase / "sensor_inverse.json")

    pdf = setup()
    pdf.add_page()

    pdf.set_font("D", "B", 18)
    pdf.set_text_color(*ACC)
    mc(pdf, 9, "Báo cáo Giai đoạn 5 - Marker-dot VBTS sensor model")
    pdf.set_font("D", "", 11)
    pdf.set_text_color(*INK)
    mc(pdf, 6, "Differentiable camera + Gaussian dot renderer cho tactile marker observation")
    pdf.set_font("D", "", 9)
    pdf.set_text_color(*MUT)
    mc(
        pdf,
        5,
        "Ngày: 2026-06-16   |   Input: PhysX-FEM marker displacement   |   Output: marker image + pixel flow",
    )
    pdf.set_text_color(*INK)
    pdf.ln(3)

    h1(pdf, "1. Mục tiêu")
    body(
        pdf,
        "Phase 5 thêm tầng cảm biến cho pipeline VBTS. Trước Phase 5, hệ thống đã có "
        "FEM/FNO dự đoán trường dịch chuyển marker disp[N,M,3], nhưng observation vẫn là "
        "field vật lý nội bộ. Phase 5 biến field đó thành tín hiệu giống cảm biến marker-dot "
        "thật: ảnh camera của lưới chấm và pixel-flow 2D của từng marker.",
    )
    code(pdf, "FEM/FNO -> disp marker 3D -> pinhole camera -> dot image / pixel flow")
    bullet(pdf, "Không cần thêm Isaac solve: sensor là phép chiếu + render từ field đã có.")
    bullet(pdf, "Toàn bộ renderer viết bằng torch, nên image <- renderer <- FNO <- action khả vi end-to-end.")
    bullet(pdf, "Scope Phase 5 hiện tại là sensor model + demo tương thích FNO; các phần object mesh, calibration DIY và RL env để phase sau.")

    h1(pdf, "2. Thành phần đã triển khai")
    table(
        pdf,
        ["File", "Vai trò", "Trạng thái"],
        [
            [
                "src/novbts/sensor/markercam.py",
                "PinholeCamera, deformed_marker_xyz, render_dots, flow tracking helper.",
                "Đã có; torch differentiable.",
            ],
            [
                "src/novbts/sensor/build_sensor_dataset.py",
                "Đọc FEM npz, chiếu marker sang pixel, lưu pix_def/pix_flow/camera config, tạo preview.",
                "Đã chạy ra sensor dataset.",
            ],
            [
                "src/novbts/sensor/sensor_inverse_demo.py",
                "Train/freeze FNO, so FEM-render vs FNO-render, rồi recover shear từ rendered image bằng autograd.",
                "Đã chạy ra compat + inverse JSON.",
            ],
        ],
        [52, 92, 30],
    )

    h1(pdf, "3. Pipeline dữ liệu")
    body(
        pdf,
        "Input chính là tập FEM đã cân bằng thêm frame normal: "
        f"{build_rep.get('data', rel(FEM / 'shear_fine_swept_normaug.npz'))}. "
        "Mỗi frame có params, mode, coords[M,2] và disp[M,3]. Phase 5 không thay đổi FNO; "
        "sensor được gắn sau field displacement như một observation layer xác định. "
        "Bản hiện tại tách dense field grid khỏi visible tracking-marker grid để giống gel marker thật hơn.",
    )
    table(
        pdf,
        ["Hạng mục", "Giá trị"],
        [
            ["Số frame", build_rep.get("N", "n/a")],
            ["Số điểm field FEM/FNO", build_rep.get("field_M", build_rep.get("M", "n/a"))],
            ["Số marker nhìn thấy", build_rep.get("sensor_M", build_rep.get("M", "n/a"))],
            ["Grid marker nhìn thấy", f"{build_rep.get('sensor_marker_side', 'n/a')} x {build_rep.get('sensor_marker_side', 'n/a')}"],
            ["Marker inset", build_rep.get("marker_inset", "n/a")],
            ["Kích thước ảnh", f"{build_rep.get('px', 'n/a')} x {build_rep.get('px', 'n/a')} px"],
            ["Working distance", f"{build_rep.get('camera', {}).get('working_dist', 'n/a')} m"],
            ["fx = fy", f"{build_rep.get('camera', {}).get('fx', 0):.3f}" if "missing" not in build_rep else "n/a"],
            ["Dot sigma", build_rep.get("sigma", "n/a")],
            ["Dot style", build_rep.get("dot_style", {}).get("polarity", "n/a")],
            ["Sensor npz", rel(build_rep.get("sensor_npz", FEM / "shear_fine_swept_normaug_sensor.npz"))],
        ],
        [48, 126],
    )

    h1(pdf, "4. Kết quả 5a - Sensor build")
    round_trip = build_rep.get("round_trip", {})
    table(
        pdf,
        ["Metric", "Kết quả", "Diễn giải"],
        [
            [
                "flow-disp cosine mean",
                f"{build_rep.get('flow_disp_cos_mean', 0):.3f}",
                "Pixel flow bám sát hướng dịch chuyển XY của marker.",
            ],
            [
                "round-trip overall",
                f"{round_trip.get('overall_px', 0):.2f} px",
                "Sai số centroid tracking từ ảnh render về marker deformed.",
            ],
            [
                "round-trip stick",
                f"{round_trip.get('stick_px', 0):.2f} px",
                "Vùng stick dễ track hơn một chút.",
            ],
            [
                "round-trip slip",
                f"{round_trip.get('slip_px', 0):.2f} px",
                "Slip biến dạng mạnh hơn nên lỗi tracking cao hơn.",
            ],
            [
                "round-trip p95",
                f"{round_trip.get('p95_px', 0):.2f} px",
                "Đuôi lỗi còn đáng chú ý, chủ yếu quanh vùng contact biến dạng mạnh.",
            ],
        ],
        [50, 34, 90],
    )
    body(
        pdf,
        "Kết quả 5a đạt mục tiêu: renderer tạo được ảnh marker-dot và pixel-flow có quan hệ hình học "
        "đúng với displacement field. Sai số round-trip không bằng 0 vì centroid tracker là kiểm thử "
        "image-level đơn giản, không dùng correspondence ground-truth.",
    )

    preview = Path(build_rep.get("preview", phase / "preview.png"))
    if preview.exists():
        h2(pdf, "Preview")
        max_w = pdf.w - pdf.l_margin - pdf.r_margin
        pdf.image(str(preview), x=pdf.l_margin, w=max_w)
        pdf.ln(2)
        body(
            pdf,
            "Preview cho thấy rest dots đều, deformed dots biến dạng tại vùng contact, và vector flow "
            "tập trung đúng theo hướng shear của frame full_slip.",
        )

    samples = Path(build_rep.get("test_samples", phase / "test_samples.png"))
    if samples.exists():
        pdf.add_page()
        h2(pdf, "Nhiều mẫu test")
        max_w = pdf.w - pdf.l_margin - pdf.r_margin
        max_h = pdf.h - pdf.get_y() - pdf.b_margin - 28
        image_fit(pdf, samples, max_w, max_h)
        pdf.ln(2)
        body(
            pdf,
            "Montage chọn các frame đại diện theo độ lớn pixel-flow trong từng mode, giúp kiểm tra "
            "marker layout và biến dạng sensor không chỉ trên một frame full_slip đơn lẻ.",
        )

    h1(pdf, "5. Kết quả 5b - FNO + renderer compatibility")
    table(
        pdf,
        ["Metric trong marker-flow space", "Rel L2"],
        [
            ["overall", f"{compat.get('flow_rel_l2_overall', 0):.3f}"],
            ["stick", f"{compat.get('flow_rel_l2_stick', 0):.3f}"],
            ["slip", f"{compat.get('flow_rel_l2_slip', 0):.3f}"],
        ],
        [100, 74],
    )
    body(
        pdf,
        "FNO + renderer tái tạo observation sensor ở mức rel-L2 khoảng 0.26 trong không gian marker-flow. "
        "Đây không phải metric renderer thuần; nó chứa cả sai số FNO trên FEM và sai số chiếu camera. "
        "Slip khó hơn stick một chút, phù hợp với kết quả FEM benchmark trước đó.",
    )

    compare_plot = Path(compare_rep.get("plot", phase / "gt_vs_fno_samples.png"))
    if compare_plot.exists():
        h2(pdf, "So sánh ảnh sinh từ GT và FNO")
        mse = compare_rep.get("image_mse", [])
        fl2 = compare_rep.get("flow_rel_l2", [])
        if mse and fl2:
            body(
                pdf,
                f"Trên các frame minh họa: image MSE trung bình {sum(mse)/len(mse):.2e}; "
                f"flow residual rel-L2 trung bình {sum(fl2)/len(fl2):.3f}.",
            )
        max_w = pdf.w - pdf.l_margin - pdf.r_margin
        max_h = pdf.h - pdf.get_y() - pdf.b_margin - 12
        image_fit(pdf, compare_plot, max_w, max_h)
        pdf.ln(2)

    h1(pdf, "6. Kết quả inverse từ ảnh sensor")
    true_s = inv.get("true_shear_mm", [0.0, 0.0])
    rec_s = inv.get("recovered_shear_mm", [0.0, 0.0])
    table(
        pdf,
        ["Hạng mục", "Giá trị"],
        [
            ["Frame", inv.get("frame", "n/a")],
            ["Mode", inv.get("mode", "n/a")],
            ["True shear (sx, sy)", f"({mm(true_s[0])}, {mm(true_s[1])})"],
            ["Recovered shear (sx, sy)", f"({mm(rec_s[0])}, {mm(rec_s[1])})"],
            ["Relative error", f"{inv.get('rel_err', 0):.3f}"],
            ["Direction error", f"{inv.get('dir_err_deg', 0):.2f} deg"],
            ["Final image loss", f"{inv.get('final_image_loss', 0):.3e}"],
            ["Optimization", f"{inv.get('steps', 'n/a')} steps, {inv.get('wall_s', 'n/a')} s"],
        ],
        [58, 116],
    )
    body(
        pdf,
        "Đây là bằng chứng mạnh nhất của Phase 5: từ ảnh marker-dot render, hệ thống recover shear "
        "với lỗi tương đối khoảng 2.1% và sai hướng khoảng 1.1 độ. Gradient đã đi qua chuỗi "
        "image <- renderer <- FNO <- action, nên sensor layer thực sự compose được với Track B.",
    )

    h1(pdf, "7. Đánh giá")
    table(
        pdf,
        ["Mốc", "Phán quyết", "Ghi chú"],
        [
            [
                "5a sensor renderer",
                "GO",
                "Ảnh marker-dot, pixel-flow, preview và sensor npz đã sinh thành công.",
            ],
            [
                "5a geometry faithfulness",
                "GO",
                "flow-disp cosine 0.973; round-trip overall 1.94 px.",
            ],
            [
                "5b FNO compatibility",
                "GO",
                "FNO-render vs FEM-render rel-L2 0.264 trong marker-flow space.",
            ],
            [
                "5b differentiable inverse",
                "GO",
                "Recover full_slip shear từ ảnh với rel_err 0.021, dir_err 1.09 deg.",
            ],
            [
                "Production sensor realism",
                "CHƯA",
                "Chưa có optics/noise/camera calibration thật; renderer hiện là Gaussian dot model sạch.",
            ],
        ],
        [48, 26, 100],
    )

    h1(pdf, "8. Hạn chế và hướng tiếp")
    bullet(pdf, "Renderer hiện là pinhole + Gaussian splat lý tưởng; chưa mô phỏng blur, lighting, lens distortion, marker occlusion hay noise của camera thật.")
    bullet(pdf, "Round-trip image tracker là centroid trong cửa sổ cục bộ, dùng để sanity-check; production tracking cần thuật toán robust hơn hoặc dùng known correspondence khi train.")
    bullet(pdf, "Compatibility rel-L2 0.264 còn chịu sai số FNO FEM; cải thiện FNO hoặc input representation sẽ kéo sensor error xuống.")
    bullet(pdf, "Phase 5c nên thêm object mesh/geometry trong Isaac hoặc dùng GT từ IPC/TacEx/Taccel để sensor thấy nhiều hình học tiếp xúc hơn.")
    bullet(pdf, "Phase 5d cần calibration DIY: fit intrinsics/extrinsics, dot layout, scale pixel-meter và noise model từ hardware.")
    bullet(pdf, "Phase 5e có thể đóng thành RL env: observation = marker image/flow, transition = FNO surrogate, reward/task từ tactile manipulation.")

    h1(pdf, "9. Tài sản sinh ra")
    bullet(pdf, f"Sensor dataset: {rel(build_rep.get('sensor_npz', FEM / 'shear_fine_swept_normaug_sensor.npz'))}")
    bullet(pdf, f"Preview: {rel(preview)}")
    bullet(pdf, f"Build report JSON: {rel(phase / 'sensor_build.json')}")
    bullet(pdf, f"Compatibility JSON: {rel(phase / 'sensor_compat.json')}")
    bullet(pdf, f"Inverse JSON: {rel(phase / 'sensor_inverse.json')}")
    bullet(pdf, "Regenerate: python -m novbts.sensor.build_sensor_dataset && python -m novbts.sensor.sensor_inverse_demo")

    ensure(DOCS)
    out = DOCS / "bao_cao_giai_doan5_sensor.pdf"
    pdf.output(str(out))
    print("WROTE", out)
    return out


if __name__ == "__main__":
    build()
