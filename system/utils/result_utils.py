
import h5py
import numpy as np
import os
import re

def average_data(algorithm="", dataset="", goal="",model_name="" ,times=10):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal,model_name, times)

    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    print("std for best accurancy:", np.std(max_accurancy))
    print("mean for best accurancy:", np.mean(max_accurancy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="",model_name="" ,times=10):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(dataset, algorithm, model_name, delete=False)))


    return test_acc


def find_latest_result_file(dataset, algorithm, model_name):
    """Tìm file mới nhất có format đúng"""
    result_path = "../results/"
    model_name = str(model_name)
    base_name = f"{dataset}_{algorithm}_{model_name}_test_0"  # Không có dấu _ thừa

    all_files = os.listdir(result_path)
    pattern = re.compile(rf"^{re.escape(base_name)}(\(\d+\))?\.h5$")  # Regex cho tên file đúng format
    matching_files = [f for f in all_files if pattern.match(f)]

    if not matching_files:
        print(f"⚠️ Không tìm thấy file phù hợp với pattern: {base_name}.h5 trong {result_path}")
        return None

    # Sắp xếp theo số trong ngoặc (x), nếu có
    matching_files.sort(key=lambda f: int(re.search(r"\((\d+)\)", f).group(1)) if re.search(r"\((\d+)\)", f) else 0)
    latest_file = matching_files[-1]  # File mới nhất
    return os.path.join(result_path, latest_file)

def read_data_then_delete(dataset, algorithm, model_name, delete=False):
    """Đọc dữ liệu từ file mới nhất, xóa nếu cần"""
    file_path = find_latest_result_file(dataset, algorithm, model_name)
    if not file_path:
        print("⚠️ Không thể đọc dữ liệu")
        return None

    print(f"📂 Đang đọc dữ liệu từ: {file_path}")

    # Đọc dữ liệu từ file HDF5
    try:
        with h5py.File(file_path, 'r') as hf:
            rs_test_acc = np.array(hf.get('rs_test_acc'))
            if rs_test_acc is None:
                print(f"⚠️ File {file_path} không chứa 'rs_test_acc'")
                return None
    except Exception as e:
        print(f"⚠️ Lỗi khi đọc file {file_path}: {e}")
        return None

    # Xóa file nếu `delete=True`
    if delete:
        os.remove(file_path)
        print(f"🗑️ Đã xóa file: {file_path}")

    return rs_test_acc