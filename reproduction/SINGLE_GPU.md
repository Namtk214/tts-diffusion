# Chạy trên một A100 80 GB

Cấu hình `single_gpu_768` dùng FLUX.1-dev BF16, 768 × 768, 28 bước, một seed và bốn ảnh/prompt. SatisDive dùng Δ = 1.5. SGI dùng CLIP/DINO, λ = 1.5, M = 128 và giữ một nửa số candidate ở mỗi vòng cho tới K = 4. Hai phương pháp chạy lần lượt. Lượt chính gồm 30 prompt Pick-a-Pic và 30 prompt GenEval, tổng cộng 480 ảnh. Lượt pilot riêng tạo thêm 8 ảnh.

Đây là chế độ mới, chưa chạy trên A100. Với SatisDive, cả hai tiến trình cùng thấy một GPU. Các text encoder của FLUX nằm trong RAM ngoài lúc mã hóa prompt. Worker đưa HPSv3 lên GPU khi cần gradient theo ảnh, trả model về RAM trước khi FLUX tiếp tục. Đỉnh VRAM còn phụ thuộc activation, hai CUDA context và bộ nhớ do từng tiến trình giữ. Cần đủ RAM hệ thống cho checkpoint, encoder và trọng số được chuyển về CPU. Thời gian truyền qua PCIe được tính vào runtime.

Trong thư mục `reproduction`, cài môi trường, xác thực quyền FLUX và chuẩn bị model theo [README](README.md). Có thể dùng `setup_environments.py --only controller sgi satisdive hpsv3 eval imagereward geneval` và chọn các model khi tải:

```bash
source scripts/env.sh
.venvs/controller/bin/python scripts/pin_models.py
.venvs/controller/bin/python scripts/download_models.py --keys flux tiny_vae clip dino qwen hpsv3 image_reward bert_tokenizer pickscore pickscore_processor
.venvs/controller/bin/python datasets/prepare_pickapic.py --size 30
.venvs/controller/bin/python datasets/prepare_geneval.py --size 30
.venvs/satisdive/bin/python scripts/preflight.py --method satisdive --protocol single_gpu_768
.venvs/sgi/bin/python scripts/preflight.py --method sgi --protocol single_gpu_768 --check-solver
bash scripts/run_768.sh pilot
```

Nếu nguồn Pick-a-Pic gốc không truy cập được, thêm `--source-repo pickapic-anonymous/pickapic_v1` vào lệnh chuẩn bị Pick-a-Pic ở trên. Kiểm tra bốn ảnh 768 × 768, gradient hữu hạn và khác 0 của SatisDive, candidate ID của SGI, runtime và VRAM của cả hai trong `outputs/single_gpu_768/`. Trường `resident_gpu_memory.peak_sum_mib` đo bộ nhớ thiết bị khi cả hai tiến trình cùng hoạt động. Nếu SatisDive bị OOM, dừng ở pilot và xem lỗi trong `records/`. Sau khi kiểm tra, chạy:

```bash
bash scripts/run_768.sh main
```

Tính số giờ sinh ảnh trên một GPU bằng `60 × (giây/nhóm SatisDive + giây/nhóm SGI) / 3600`. Một prompt chỉ cho ước lượng thô. Chuẩn bị LPIPS, DreamSim và GenEval như trong README rồi dùng `scripts/evaluate.py --run-dir RUN_DIR --resume` cho từng run. Evaluator tự chọn cấu hình 768, gồm HPSv3, ImageReward, PickScore, CLIP, DreamSim, DINO và LPIPS, thêm accuracy trên GenEval. Mốc SAT từ paper ở 1024 là ngưỡng chuyển sang 768, chỉ báo khi xác nhận thang điểm. Báo cáo phải ghi rõ độ phân giải 768 và cách chuyển model giữa CPU/GPU.
