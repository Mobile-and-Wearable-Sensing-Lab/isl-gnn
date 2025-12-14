
def read_labels_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f]

    return labels