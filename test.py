import torch

a_org = torch.tensor(
    [
        [1, 2, 5, 7, 3, 5, 4, 9, 1, 4, 9, 0, 8, 1, 2, 8, 9],
        [2, 3, 1, 2, 3, 1, 2, 1, 3, 1, 2, 1, 3, 2, 3, 2, 3],
    ],
    dtype=torch.int32,
)
# a_new = torch.zeros_like(a_org)

# ブールマスクを作成し、a_org が 2 または 3 の位置を True に設定
mask = (a_org == 2) | (a_org == 3)

# 有効な値（2または3）の位置を見つける
valid_positions = torch.where(mask, torch.arange(a_org.size(1)), 0)

# 累積最大値を計算して、各位置に対応する最後の有効な位置を見つける
last_valid_positions = torch.cummax(valid_positions, dim=1)[0]

# 最後の有効な位置に対応する値を取得
a_new = torch.where(last_valid_positions == 0, torch.tensor(0), a_org.gather(1, last_valid_positions))

print(a_new)
