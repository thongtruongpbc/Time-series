import pickle


def pickle_dump(data, file_path):
    f_write = open(file_path, "wb")
    pickle.dump(data, f_write, True)


def pickle_load(file_path):
    f_read = open(file_path, "rb")
    data = pickle.load(f_read)

    return data


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x
