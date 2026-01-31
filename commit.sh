#!/bin/bash

echo "🚀 Đang bắt đầu quá trình đồng bộ các folder lên các nhánh riêng..."
cd ~/mnt/thongtx || exit

if [ ! -d .git ]; then
    git init
    echo "✅ Đã khởi tạo Git."
fi

if ! git remote | grep -q "origin"; then
    git remote add origin https://github.com/thongtruongpbc/Time-series.git
else
    git remote set-url origin https://github.com/thongtruongpbc/Time-series.git
fi

find . -mindepth 2 -name ".git" -type d -not -path "./.git/*" -exec rm -rf {} +

if [ -d "datasets" ]; then
    git rm -r --cached datasets/ 2>/dev/null
    echo "🚫 Đã bỏ qua thư mục datasets."
fi

# 5. Add và commit ở nhánh chính
git add .
if ! git diff-index --quiet HEAD --; then
    git commit -m "Auto-sync backup: $(date +'%Y-%m-%d %H:%M:%S')"
else
    echo "✨ Không có thay đổi mới ở local."
fi

# 6. Duyệt qua các folder để tách nhánh
for dir in */; do
    branch_name="${dir%/}"
    
    # Bỏ qua các folder không muốn push
    if [[ "$branch_name" == "node_modules" || "$branch_name" == ".git" || "$branch_name" == "datasets" ]]; then
        continue
    fi

    echo "------------------------------------------"
    echo "📁 Đang xử lý thư mục: $branch_name"
    
    # Tách nhánh bằng subtree và push
    # Lấy ID của commit tree sau khi split
    TREE_ID=$(git subtree split --prefix="$branch_name" 2>/dev/null)

    if [ -n "$TREE_ID" ]; then
        git push origin "$TREE_ID":refs/heads/"$branch_name" --force
        echo "✅ Đã push thành công lên nhánh: $branch_name"
    else
        echo "❌ Lỗi: Không thể tách subtree cho $branch_name (có thể folder trống hoặc lỗi git)"
    fi
done

echo "------------------------------------------"
echo "🎉 Hoàn thành tất cả!"