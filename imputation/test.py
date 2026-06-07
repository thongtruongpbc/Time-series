# import torch

# # Load the checkpoint
# checkpoint_path = "imputation/vector_db_poly/Crossformer_ECL_mask_0.125_custom_ftM_sl96_ll0_pl0_dm64_nh2_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/cached_states.pt"
# checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# print(checkpoint.shape)

import shutil

total, used, free = shutil.disk_usage(
    "imputation/vector_db_poly_tmp"
)  # Thay "/" bằng ổ đĩa bạn định lưu (vd: "D:")

print(f"Tổng dung lượng: {total / (1024**3):.2f} GB")
print(f"Đã dùng: {used / (1024**3):.2f} GB")
print(f"Còn trống: {free / (1024**3):.2f} GB")
