# Thiết kế cho một A100 40 GB

Cấu hình `single_gpu_768` dùng FLUX.1-dev BF16, 768 × 768, 28 bước, một seed và bốn ảnh/prompt. SatisDive dùng Δ = 1.5. SGI dùng CLIP/DINO, λ = 1.5, M = 128 và giữ một nửa số candidate ở mỗi vòng cho tới K = 4. Hai phương pháp chạy lần lượt. Lượt chính gồm 30 prompt Pick-a-Pic và 30 prompt GenEval, tổng cộng 480 ảnh. Lượt pilot riêng tạo thêm 8 ảnh.

Mục tiêu phần cứng là một A100 40 GB, RAM hệ thống khuyến nghị 128 GB và ổ đĩa 200 GB (ít nhất 150 GB). Đây là thiết kế chưa được xác nhận bằng pilot GPU; không cam kết hết OOM hay thời gian chạy trước khi đo.

Pipeline dùng BF16 và quản lý trọng số theo từng pha, không lượng tử hóa:

1. Khởi tạo FLUX trên CPU. Worker HPSv3 khởi tạo trên GPU rồi chuyển về CPU trước khi đưa FLUX lên GPU.
2. Khi mã hóa prompt, chuyển transformer và VAE về CPU, đưa hai text encoder lên GPU. Mã hóa xong, đưa encoder về CPU rồi khôi phục transformer và VAE lên GPU. SGI dùng cùng cách mã hóa này.
3. Khi SatisDive yêu cầu gradient reward theo ảnh, chuyển transformer và VAE về CPU trước khi worker đưa HPSv3 lên GPU. Worker trả HPSv3 về CPU và giải phóng cache trước khi gửi kết quả. Sau đó khôi phục FLUX lên GPU.
4. Gom bốn ảnh của mỗi pass thành một yêu cầu reward. Worker vẫn xử lý từng ảnh theo thứ tự, nhưng chỉ chuyển FLUX/HPSv3 một lần cho mỗi pass thay vì bốn lần. Phần reward này nằm trong pass không giữ graph FLUX. Pass gradient FLUX được dựng lại sau khi HPSv3 đã rời GPU. Adapter từ chối offload nếu đầu vào reward còn yêu cầu gradient hoặc autograd vẫn bật, tránh chuyển trọng số giữa lúc graph còn sống.
5. Giữ gradient checkpointing của transformer, thêm checkpointing VAE và giải mã VAE từng ảnh. SGI vốn denoise từng candidate; M=128 được giữ nguyên, không sinh 128 ảnh cùng một batch.

Cách này tránh FLUX (~24 GB trọng số transformer BF16) cùng chiếm GPU với HPSv3. Activation, CUDA context và workspace vẫn cần bộ nhớ; pilot phải xác nhận đỉnh VRAM thực tế. Hai tiến trình cùng thấy một GPU. Các lần chuyển trọng số qua PCIe được tính vào runtime và có thể làm SatisDive chậm đáng kể. Metadata ghi `phase_swap_bf16_40gb`; code hash mới tạo run ID riêng để không dùng lẫn timing với phiên bản cũ.

Launcher đặt mặc định `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8` để tránh tạo thread pool theo toàn bộ CPU của host trong container có quota nhỏ. SGI dùng Python 3.11 và Gurobi 14.0.0 vì giấy phép đi kèm solver cũ đã hết hạn; môi trường thực tế và phiên bản solver được ghi trong metadata.

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

Nếu ổ đĩa container chỉ 32 GB nhưng `/dev/shm` đủ lớn và RAM cgroup còn dư, có thể đặt `HF_HOME=/dev/shm/tts-hf` cho cả download và generation. Lần triển khai A100 40 GB đã dùng khoảng 63 GB cho FLUX/Qwen/HPSv3. Kiểm tra `df -h /dev/shm`, giới hạn RAM cgroup, dung lượng còn lại và giữ outputs trên `/workspace`. Cache trong RAM có thể mất khi máy dừng/khởi động lại; đây là lựa chọn lưu tạm, không thay thế ổ đĩa bền vững.

## Pilot đo trên VM ngày 2026-11-25

Một prompt Pick-a-Pic (ID `000380`, nguồn mirror) cho mỗi method, bốn ảnh 768 mỗi nhóm. Chưa phải kết quả trên 30 + 30 prompt.

| Method | Runtime/nhóm, không gồm cold start | Đỉnh VRAM toàn GPU | Kết quả |
|---|---:|---:|---|
| SatisDive | 384,50 giây | 30.273 MiB (29,56 GiB) | 4 ảnh; reward/diversity gradient hữu hạn và khác 0 |
| SGI | 148,61 giây | 26.065 MiB (25,45 GiB) | 4 ảnh; candidate IDs 25, 55, 92, 110 |

SatisDive: 152 denoiser forwards, 48 latent autograd calls, 40 reward forwards/backwards. SGI: 340 denoiser forwards; solver chạm giới hạn thời gian 2 lần, có nghiệm khả thi. Gurobi 14.0.0 và Python 3.11 được dùng cho SGI do giấy phép đóng gói bản cũ hết hạn. Hai method chạy nối tiếp, cùng một GPU. Với một prompt pilot, ngoại suy sinh 480 ảnh khoảng 8,9 giờ; chưa bao gồm offline metrics, cold start và biến động giữa prompt. Các lần pilot lỗi hoặc bị dừng được giữ riêng, không tính là thành công.
