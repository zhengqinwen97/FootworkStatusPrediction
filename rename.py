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

def normalize_name(name, count) -> str:
    if '.zip' in name:
        return name

    if 'Obj' not in name:
        lr = 'left_0__right_0'
    elif 'Obj1' in name:
        lr = 'left_3__right_1'
    elif 'Obj2' in name:
        lr = 'left_3__right_1'
    elif 'Obj3' in name:
        lr = 'left_2__right_2'
    elif 'Obj4' in name:
        lr = 'left_2__right_2'
    elif 'Obj5' in name:
        lr = 'left_1__right_3'
    elif 'Obj6' in name:
        lr = 'left_1__right_3'
    elif 'Obj7' in name:
        lr = 'left_1__right_0'
    elif 'Obj8' in name:
        lr = 'left_1__right_0'
    else:
        raise RuntimeError(f"invalid name: {name}")

    if 'brisk' in name:
        motion_state = 'brisk_walking'
    elif 'slow' in name:
        motion_state = 'slow_walking'
    elif 'standing' in name:
        motion_state = 'standing'
    else:
        raise RuntimeError(f"invalid name: {name}")

    if 'right' in name:
        travel_state = 'turn_right'
    elif 'turn' in name:
        travel_state = 'turn_left'
    else:
        travel_state = 'nostatus'

    if travel_state != 'nostatus':
        trips_count = '1'
    else:
        trips_count = '0'

    rename = '__'.join([lr, motion_state, travel_state, trips_count]) + "__" + str(count) + '.json'
    return rename


def rename_all_recursive(root_folder: str, dry_run: bool = False):
    root_folder = os.path.abspath(root_folder)
    if not os.path.isdir(root_folder):
        raise ValueError(f"不是目录：{root_folder}")

    count = 5000
    for dirpath, dirnames, filenames in os.walk(root_folder, topdown=False):
        for filename in filenames:
            count += 1
            old_path = os.path.join(dirpath, filename)
            new_name = normalize_name(filename, count)
            new_name = re.sub(r'\d+(?=turn)', '', new_name)
            final_name = safe_new_name(dirpath, new_name)
            new_path = os.path.join(dirpath, final_name)
            print(f"文件: {old_path} -> {new_path}")
            if not dry_run:
                os.rename(old_path, new_path)


        # for dirname in dirnames:
        #     old_path = os.path.join(dirpath, dirname)
        #     # new_name = normalize_name(dirname)
        #     # if new_name == dirname:
        #     #     continue
        #     final_name = safe_new_name(dirpath, new_name)
        #     new_path = os.path.join(dirpath, final_name)
        #     print(f"目录: {old_path} -> {new_path}")
        #     if not dry_run:
        #         os.rename(old_path, new_path)

if __name__ == "__main__":
    # import argparse
    # p = argparse.ArgumentParser(description="递归重命名文件：大写->小写，空格/括号->下划线")
    # p.add_argument("folder", help="目标文件夹（递归处理）")
    # p.add_argument("--dry-run", action="store_true", help="仅显示将要做的改动，不实际重命名")
    # args = p.parse_args()

    # rename_all_recursive("datas/feet_datax", dry_run=False)
    rename_all_recursive("datas/feet_dataxx", dry_run=False)

