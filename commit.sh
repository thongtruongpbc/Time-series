#!/bin/bash

echo "🚀 Đang bắt đầu quá trình đồng bộ các folder lên các nhánh riêng..."
cd ~/mnt/thongtx
git init
git remote add origin https://github.com/thongtruongpbc/Time-series.git
git add .
# Kiểm tra xem có gì để commit không
if ! git diff-index --quiet HEAD --; then
    git commit -m "Auto-sync backup: $(date +'%Y-%m-%d %H:%M:%S')"
else
    echo "✨ Không có thay đổi mới ở local."
fi

for dir in */; do
    branch_name="${dir%/}"
    
    if [[ "$branch_name" == "node_modules" || "$branch_name" == ".git" ]]; then
        continue
    fi

    echo "------------------------------------------"
    echo "📁 Đang xử lý thư mục: $branch_name"
    
    # Sử dụng subtree split để tách folder thành một commit tree riêng và đẩy lên remote
    # Lệnh này sẽ tạo/cập nhật nhánh remote trực tiếp từ folder
    git push origin "$(git subtree split --prefix=$branch_name)":refs/heads/"$branch_name" --force

    if [ $? -eq 0 ]; then
        echo "✅ Đã push thành công lên nhánh: $branch_name"
    else
        echo "❌ Có lỗi xảy ra khi push folder: $branch_name"
    fi
done

echo "------------------------------------------"
echo "🎉 Hoàn thành tất cả!"