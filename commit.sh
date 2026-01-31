#!/bin/bash

echo "🚀 Đang đồng bộ bằng phương pháp Manual Push (Không dùng subtree)..."
cd ~/mnt/thongtx || exit

# 1. Khởi tạo Git & Remote
[ ! -d .git ] && git init
git remote add origin https://github.com/thongtruongpbc/Time-series.git 2>/dev/null
git remote set-url origin https://github.com/thongtruongpbc/Time-series.git

# 2. Xóa sạch vết tích Git con
find . -mindepth 2 -name ".git" -type d -not -path "./.git/*" -exec rm -rf {} +
git rm -r --cached . >/dev/null 2>&1

# 3. Add và Commit
git add .
git reset HEAD datasets/ 2>/dev/null
if ! git diff-index --quiet HEAD --; then
    git commit -m "Sync: $(date +'%Y-%m-%d %H:%M:%S')"
fi

# 4. Duyệt qua các folder
for dir in */; do
    branch_name="${dir%/}"
    
    if [[ "$branch_name" == "node_modules" || "$branch_name" == ".git" || "$branch_name" == "datasets" ]]; then
        continue
    fi

    echo "------------------------------------------"
    echo "📁 Đang xử lý: $branch_name"

    # CHIẾN THUẬT: Tạo một index tạm thời để tạo commit chỉ cho folder đó
    # Đây là cách Git thực hiện subtree ở cấp độ thấp
    export GIT_INDEX_FILE=".git/index.temp"
    rm -f "$GIT_INDEX_FILE"
    
    # Đọc nội dung folder vào index tạm
    SUBTREE_ID=$(git ls-tree -d HEAD "$branch_name" | awk '{print $3}')
    
    if [ -n "$SUBTREE_ID" ]; then
        # Tạo một commit object mới từ folder này
        NEW_COMMIT=$(echo "Push folder $branch_name" | git commit-tree "$SUBTREE_ID")
        
        # Push commit đó lên nhánh tương ứng
        git push origin "$NEW_COMMIT":refs/heads/"$branch_name" --force
        echo "✅ Đã push thành công: $branch_name"
    else
        echo "❌ Lỗi: Không thể tìm thấy dữ liệu cho $branch_name"
    fi

    rm -f "$GIT_INDEX_FILE"
    unset GIT_INDEX_FILE
done

echo "🎉 Xong!"