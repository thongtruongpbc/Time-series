#!/bin/bash

echo "🚀 Đang bắt đầu quá trình đồng bộ..."
cd ~/mnt/thongtx || exit

# 1. Khởi tạo Git nếu chưa có
if [ ! -d .git ]; then
    git init
fi

# 2. Cấu hình Remote
git remote add origin https://github.com/thongtruongpbc/Time-series.git 2>/dev/null
git remote set-url origin https://github.com/thongtruongpbc/Time-series.git

# 3. Xử lý triệt để lỗi "Embedded Git"
# Xóa file .git con
find . -mindepth 2 -name ".git" -type d -not -path "./.git/*" -exec rm -rf {} +

# QUAN TRỌNG: Xóa Index cũ để Git nhận diện lại các folder con là folder thường
git rm -r --cached . >/dev/null 2>&1

# 4. Add lại từ đầu (trừ datasets)
git add .
# Loại bỏ datasets khỏi commit này một cách thủ công để chắc chắn
git reset HEAD datasets/ 2>/dev/null

# 5. Commit
if ! git diff-index --quiet HEAD --; then
    git commit -m "Fix subtree & Auto-sync: $(date +'%Y-%m-%d %H:%M:%S')"
else
    echo "✨ Không có thay đổi mới."
fi

# 6. Duyệt qua các folder
for dir in */; do
    branch_name="${dir%/}"
    
    if [[ "$branch_name" == "node_modules" || "$branch_name" == ".git" || "$branch_name" == "datasets" ]]; then
        continue
    fi

    echo "------------------------------------------"
    echo "📁 Đang xử lý nhánh: $branch_name"
    
    # Ép buộc tạo lại subtree split để tránh dùng cache cũ lỗi
    TREE_ID=$(git subtree split --prefix="$branch_name")

    if [ -n "$TREE_ID" ]; then
        git push origin "$TREE_ID":refs/heads/"$branch_name" --force
        echo "✅ Đã push thành công: $branch_name"
    else
        echo "❌ Lỗi: Vẫn không thể tách subtree cho $branch_name."
    fi
done

echo "------------------------------------------"
echo "🎉 Hoàn thành!"