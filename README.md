# TTS diffusion: SatisDive và SGI

Triển khai benchmark theo [hướng dẫn gốc](reproduce_satisdive_sgi.md), chỉ dùng **SatisDive và Scaling Group Inference (SGI), FLUX.1-dev**.

Mặc định ban đầu: **100 prompt Pick-a-Pic + 100 GenEval + 100 HPDv2**. Mỗi phương pháp trả 4 ảnh/prompt: E1 có 600 nhóm, 2.400 ảnh cuối. Các candidate bị SGI loại, pilot và sweep không nằm trong số này.

Theo yêu cầu, **chưa tải dataset/model, chưa chạy CPU smoke test hoặc sinh ảnh trên máy local**. Đã chuẩn bị mã nguồn, cấu hình, tests để chạy sau và hướng dẫn VM. Kiểm tra hiện tại chỉ là kiểm tra tĩnh; chưa có kết quả tái lập thực nghiệm.

- [Hướng dẫn chạy trên VM](reproduction/README.md)
- [Trạng thái triển khai và các điểm chưa xác minh](reproduction/REPRODUCTION_STATUS.md)
- [Định nghĩa metric và cách đọc báo cáo](reproduction/METRICS.md)
- [Đối chiếu paper với code công khai](reproduction/reports/PAPER_CODE_AUDIT.md)

Hai repo gốc được giữ tại `reproduction/third_party/`, kèm license, commit gốc và các patch đã sửa. Paper PDF được lưu tại `papers/`. Mã nguồn benchmark bổ sung nằm riêng trong `reproduction/benchmark`, `methods`, `evaluation`, `scripts` và `tests`.

Phạm vi: native verification, unified benchmark, sweep Δ/λ, scaling số candidate, evaluator chung và báo cáo. Không bổ sung baseline hay phương pháp sinh ảnh khác. Xem [attribution](THIRD_PARTY.md) trước khi phân phối các thành phần upstream.
