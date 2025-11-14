import os
import re

def safe_new_name(dirpath, new_name):
    base, ext = os.path.splitext(new_name)
    candidate = new_name
    i = 1
    while os.path.exists(os.path.join(dirpath, candidate)):
        candidate = f"{base}_{i}{ext}"
        i += 1
    return candidate

def normalize_name(name: str) -> str:
    new = name.lower()
    for ch in (' ', '(', ')'):
        new = new.replace(ch, '_')
    return new

def rename_all_recursive(root_folder: str, dry_run: bool = False):
    root_folder = os.path.abspath(root_folder)
    if not os.path.isdir(root_folder):
        raise ValueError(f"不是目录：{root_folder}")

    # 先重命名子目录和文件，再重命名当前目录自身（bottom-up）
    for dirpath, dirnames, filenames in os.walk(root_folder, topdown=False):
        # 1. 重命名文件
        for filename in filenames:
            old_path = os.path.join(dirpath, filename)
            new_name = normalize_name(filename)
            # if new_name == filename:
            #     continue
            new_name = re.sub(r'\d+(?=turn)', '', new_name)

            final_name = safe_new_name(dirpath, new_name)
            new_path = os.path.join(dirpath, final_name)
            print(f"文件: {old_path} -> {new_path}")
            if not dry_run:
                os.rename(old_path, new_path)

        for dirname in dirnames:
            old_path = os.path.join(dirpath, dirname)
            new_name = normalize_name(dirname)
            if new_name == dirname:
                continue
            final_name = safe_new_name(dirpath, new_name)
            new_path = os.path.join(dirpath, final_name)
            print(f"目录: {old_path} -> {new_path}")
            if not dry_run:
                os.rename(old_path, new_path)

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="递归重命名文件：大写->小写，空格/括号->下划线")
    p.add_argument("folder", help="目标文件夹（递归处理）")
    p.add_argument("--dry-run", action="store_true", help="仅显示将要做的改动，不实际重命名")
    args = p.parse_args()

    rename_all_recursive(args.folder, dry_run=args.dry_run)

