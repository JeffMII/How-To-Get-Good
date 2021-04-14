import sys, os, csv, re, math, copy
import numpy as npy
from mpmath import mpf, mp
from multiprocessing import Pool

attributes_data_count = 0
attributes_matrices_value_count = 0
attributes_matrices_entropy_count = 0
attributes_matrices_info_gain_count = 0
classes_data_count = 0
classes_matrices_value_count = 0
classes_matrices_entropy_count = 0
classes_matrices_info_gain_count = 0
classes_matrices_prediction_count = 0
do_path_depth = 0
print_depth = 0
debug = False
test = False

def __main():
    global test
    params = check_params(sys.argv)
    dataset = get_dataset(params[0], params[1])
    dataset = do_values(dataset)
    dataset = prepare_dataset(dataset)
    dataset = do_good(dataset)
    dataset = do_value_matrix(dataset)
    dataset = do_entropy_matrix(dataset)
    dataset = do_info_gain_matrix(dataset)
    dataset = do_tree_matrix(dataset)
    if test:
        yes = input("would you like to load a testfile for predictions? (y/n): ")
        while yes != 'n':    
            while yes != 'y' and yes != 'n':
                print("\tincorrect input:", yes)
                yes = input("would you like to load a testfile for predictions? (y/n): ")
            if yes == 'y':
                dataset = do_prediction_matrix(dataset)
                print("\tfinished predictions")
            yes = input("would you like to load another testfile for predictions? (y/n): ")

def check_params(params = None):
    if len(params) == 1:
        filename = input("enter dataset filename (e.g. data.csv): ")
        while re.search(r"\.csv$", filename) == None:
            print("\tincorrect dataset file type (must be .csv): ", filename)
            filename = input("enter dataset filename (e.g. data.csv): ")
        num_classes = input("enter number of classes in the dataset: ")
        while not num_classes.isnumeric() or int(num_classes) < 1:
            if not num_classes.isnumeric():
                print("\tincorrect input type for number of classes (must be an integer): ", num_classes)
            else:
                print("\tincorrect number of classes (must be 1 or more): ", num_classes)
            num_classes = input("enter number of classes in the dataset: ")
        p = [filename, int(num_classes)]
        return p
    elif len(params) == 2 and (params[1] == "--help" or params[1] == "-h"):
        print("to learn how-to-get-good.py, use the following:")
        print()
        print("\tpython3 how-to-get-good.py [.csv file] [number of classes]")
        print()
        print("\tor")
        print()
        print("\tfollow the prompts after running python3 how-to-get-good.py")
        return None
    elif len(params) == 3 and re.search(r"\.csv$", params[1]) != None and params[2].isnumeric():
        return [params[1], int(params[2])]
    else:
        print("for guidance on how-to-get-good.py, use the following:")
        print()
        print("\tpython3 how-to-get-good.py --help")
        print()
        print("\tor")
        print()
        print("\tpython3 how-to-get-good.py -h")
        return None

def get_dataset(filename, num_classes):
    dataset = {"attributes": { "names" : [], "data" : [], "values" : [], "matrices": {"value": [], "entropy": [], "info-gain": []}}, "classes": {"names" : [], "data" : [], "values" : [], "goodness" : [], "matrices": {"value": [], "entropy": [], "info-gain": [], "tree": [], "prediction": []}}, "nrecords": 0}
    good_read = False
    delimiters = [";", ",", "|", "\t"]
    while not good_read:
        with open(".\\input\\" + filename, "r") as csvfile:
            readcsv = csv.reader(csvfile, delimiter = delimiters.pop(0))
            row = next(readcsv)
            if len(row) == 1:
                csvfile.close()
                good_read = False
                continue
            dataset["attributes"]["names"] = row[0:len(row) - num_classes]
            dataset["classes"]["names"] = row[len(dataset["attributes"]["names"]):len(row)]
            dataset["attributes"]["data"] = [None] * len(dataset["attributes"]["names"])
            dataset["classes"]["data"] = [None] * len(dataset["classes"]["names"])
            for b in range(len(dataset["classes"]["names"])):
                break
            for row in readcsv:
                for a in range(len(dataset["attributes"]["names"])):
                    if dataset["attributes"]["data"][a] == None:
                        dataset["attributes"]["data"][a] = []
                    dataset["attributes"]["data"][a].append(row[a])
                for b in range(len(dataset["classes"]["names"])):
                    if dataset["classes"]["data"][b] == None:
                        dataset["classes"]["data"][b] = []
                    dataset["classes"]["data"][b].append(row[len(dataset["attributes"]["names"]) + b])
        dataset = update_nrecords(dataset)
        log_dataset_attributes(dataset, "init-set.txt")
        log_dataset_classes(dataset, "init-set.txt")
        return dataset

def update_nrecords(dataset):
    dataset["nrecords"] = len(dataset["classes"]["data"][0])
    return dataset

def log_dataset_attributes(dataset, filename = None):
    if not debug:
        return
    global attributes_data_count
    if filename == None:
        filename = str(attributes_data_count) + "-" + dataset["attributes"]["names"][0]
        attributes_data_count += 1
        if len(dataset["attributes"]["names"]) > 1:
            filename += "_" + dataset["attributes"]["names"][len(dataset["attributes"]["names"]) - 1]
        filename += "-n" + str(dataset["nrecords"]) + "-set.txt"
    path = create_directory("output\\attributes\\data")
    stdout = sys.stdout
    sys.stdout = open(path + "\\" + filename, "w")
    for a in range(len(dataset["attributes"]["names"])):
        for r in range(dataset["nrecords"]):
            print(dataset["attributes"]["names"][a], "|", dataset["attributes"]["data"][a][r])
    sys.stdout.close()
    sys.stdout = stdout

def log_dataset_classes(dataset, filename = None):
    if not debug:
        return
    global classes_data_count
    if filename == None:
        filename = str(classes_data_count) + "-" + dataset["classes"]["names"][0]
        classes_data_count += 1
        if len(dataset["classes"]["names"]) > 1:
            filename += "_" + dataset["classes"]["names"][len(dataset["classes"]["names"]) - 1]
        filename += "-n" + str(dataset["nrecords"]) + "-set.txt"
    path = create_directory("output\\classes\\data")
    stdout = sys.stdout
    sys.stdout = open(path + "\\" + filename, "w")
    for b in range(len(dataset["classes"]["names"])):
        for r in range(dataset["nrecords"]):
            print(dataset["classes"]["names"][b], "|", dataset["classes"]["data"][b][r])
    sys.stdout.close()
    sys.stdout = stdout

def do_values(dataset):
    dataset["attributes"]["values"] = [None] * len(dataset["attributes"]["names"])
    dataset["classes"]["values"] = [None] * len(dataset["classes"]["names"])
    for a in range(len(dataset["attributes"]["names"])):
        dataset["attributes"]["values"][a] = []
        for data in dataset["attributes"]["data"][a]:
            if data not in dataset["attributes"]["values"][a]:
                dataset["attributes"]["values"][a].append(data)
    is_numeric = True
    for b in range(len(dataset["classes"]["names"])):
        for data in dataset["classes"]["data"][b]:
            if dataset["classes"]["values"][b] == None:
                dataset["classes"]["values"][b] = []
            if data not in dataset["classes"]["values"][b]:
                if is_numeric and not data.isnumeric():
                    is_numeric = False
                dataset["classes"]["values"][b].append(data)
    if is_numeric:
        for b in range(len(dataset["classes"]["names"])):
            dataset["classes"]["values"][b].sort(key = float)
    log_attributes_values(dataset)
    log_classes_values(dataset)
    return dataset

def log_attributes_values(dataset, filename = None):
    if not debug:
        return
    if filename == None:
        filename = dataset["attributes"]["names"][0]
        if len(dataset["attributes"]["names"]) > 1:
            filename += "-" + dataset["attributes"]["names"][len(dataset["attributes"]["names"]) - 1]
        filename += "-values.txt"
    path = create_directory("output\\attributes\\values")
    stdout = sys.stdout
    sys.stdout = open(path + "\\" + filename, "w")
    for a in range(len(dataset["attributes"]["names"])):
        for x in range(len(dataset["attributes"]["values"][a])):
            print(dataset["attributes"]["names"][a], "|", dataset["attributes"]["values"][a][x])
    sys.stdout.close()
    sys.stdout = stdout

def log_classes_values(dataset, filename = None):
    if not debug:
        return
    if filename == None:
        filename = dataset["classes"]["names"][0]
        if len(dataset["classes"]["names"]) > 1:
            filename += "-" + dataset["classes"]["names"][len(dataset["classes"]["names"]) - 1]
        filename += "-values.txt"
    path = create_directory("output\\classes\\values")
    stdout = sys.stdout
    sys.stdout = open(path + "\\" + filename, "w")
    for b in range(len(dataset["classes"]["names"])):
        for y in range(len(dataset["classes"]["values"][b])):
            print(dataset["classes"]["names"][b], "|", dataset["classes"]["values"][b][y])
    sys.stdout.close()
    sys.stdout = stdout

def prepare_dataset(dataset):
    for a in range(len(dataset["attributes"]["names"])):
        if len(dataset["attributes"]["values"][a]) > 5:
            dataset["attributes"]["values"][a] = normalize(dataset["attributes"]["values"][a])
            for r in range(len(dataset["attributes"]["data"][a])):
                for x in range(len(dataset["attributes"]["values"][a])):
                    limits = dataset["attributes"]["values"][a][x].split("-")
                    if dataset["attributes"]["data"][a][r].isdecimal() and int(dataset["attributes"]["data"][a][r]) >= int(limits[0]) and int(dataset["attributes"]["data"][a][r]) <= int(limits[1]):
                        dataset["attributes"]["data"][a][r] = dataset["attributes"]["values"][a][x]
    for b in range(len(dataset["classes"]["names"])):
        if len(dataset["classes"]["values"][b]) > 5:
            dataset["classes"]["values"][b] = normalize(dataset["classes"]["values"][b])
            for r in range(len(dataset["classes"]["data"][b])):
                for y in range(len(dataset["classes"]["values"][b])):
                    limits = dataset["classes"]["values"][b][y].split("-")
                    if dataset["classes"]["data"][b][r].isdecimal() and int(dataset["classes"]["data"][b][r]) >= int(limits[0]) and int(dataset["classes"]["data"][b][r]) <= int(limits[1]):
                        dataset["classes"]["data"][b][r] = dataset["classes"]["values"][b][y]
    log_prepared_attributes(dataset)
    log_prepared_classes(dataset)
    return dataset

def log_prepared_attributes(dataset, filename = "prepared.txt"):
    if not debug:
        return
    path = create_directory("output\\attributes\\data")
    stdout = sys.stdout
    sys.stdout = open(path + "\\" + filename, "w")
    for a in range(len(dataset["attributes"]["names"])):
        for r in range(dataset["nrecords"]):
            print(dataset["attributes"]["names"][a], "|", dataset["attributes"]["data"][a][r])
    sys.stdout.close()
    sys.stdout = stdout

def log_prepared_classes(dataset, filename = "prepared.txt"):
    if not debug:
        return
    path = create_directory("output\\classes\\data")
    stdout = sys.stdout
    sys.stdout = open(path + "\\" + filename, "w")
    for b in range(len(dataset["classes"]["names"])):
        for r in range(dataset["nrecords"]):
            print(dataset["classes"]["names"][b], "|", dataset["classes"]["data"][b][r])
    sys.stdout.close()
    sys.stdout = stdout

def normalize(data):
    if npy.size(data) <= 5:
        return data
    min = int(get_min(data))
    if min % 5 != 0:
        min = min - (min % 5)
    max = int(get_max(data))
    if max % 5 != 0:
        max = max + (5 - (max % 5))
    rng = max - min
    div = []
    d = []
    for i in range(2, max + 1):
        if rng % i == 0 or rng % i == 1:
            div.append(i)
    div.sort()
    div = div[0:int(math.ceil(len(div) / 2))]
    while div[len(div) - 1] > 5:
        div = div[0:len(div) - 1]
    div = div[len(div) - 1]
    for i in range(div):
        if i == 0:
            d.append(str(int(max - rng * ((i + 1) / div) + 1)) + "-" + str(max))
        elif i == div - 1:
            d.append(str(min) + "-" + str(int(max - rng * (i / div))))
            break
        else:
            d.append(str(int(max - rng * ((i + 1) / div)) + 1) + "-" + str(int(max - rng * (i / div))))
    d.reverse()
    return d

def get_min(data):
    if npy.size(data) == 1:
        return int(data[0])
    min = int(data[0])
    for i in range(len(data)):
        if int(data[i]) < min:
            min = int(data[i])
    return min

def get_max(data):
    if npy.size(data) == 1:
        return int(data[0])
    max = int(data[0])
    for i in range(len(data)):
        if int(data[i]) > max:
            max = int(data[i])
    return max

def do_good(dataset):
    for b in range(len(dataset["classes"]["names"])):
        dataset["classes"]["goodness"].append([])
        print(dataset["classes"]["values"][b])
        for y in range(len(dataset["classes"]["values"][b])):
            goodness = math.tan((2 * (y / (len(dataset["classes"]["values"][b]) - 1))) - 1)
            dataset["classes"]["goodness"][b].append(goodness)
            print(dataset["classes"]["goodness"][b][y])
    return dataset

def do_value_matrix(dataset):
    dataset["attributes"]["matrices"]["value"] = [None] * len(dataset["attributes"]["names"])
    for a in range(len(dataset["attributes"]["names"])):
        dataset["attributes"]["matrices"]["value"][a] = [None] * len(dataset["classes"]["names"])
        for b in range(len(dataset["classes"]["names"])):
            dataset["attributes"]["matrices"]["value"][a][b] = [None] * len(dataset["attributes"]["values"][a])
            for x in range(len(dataset["attributes"]["values"][a])):
                dataset["attributes"]["matrices"]["value"][a][b][x] = [None] * len(dataset["classes"]["values"][b])
                for y in range(len(dataset["classes"]["values"][b])):
                    dataset["attributes"]["matrices"]["value"][a][b][x][y] = 0
                    for r in range(dataset["nrecords"]):
                        if dataset["attributes"]["data"][a][r] == dataset["attributes"]["values"][a][x] and dataset["classes"]["data"][b][r] == dataset["classes"]["values"][b][y]:
                            dataset["attributes"]["matrices"]["value"][a][b][x][y] += 1
    dataset["classes"]["matrices"]["value"] = [None] * len(dataset["classes"]["names"])
    for b in range(len(dataset["classes"]["names"])):
        dataset["classes"]["matrices"]["value"][b] = [None] * len(dataset["classes"]["values"][b])
        for y in range(len(dataset["classes"]["values"][b])):
            dataset["classes"]["matrices"]["value"][b][y] = 0
            for r in range(dataset["nrecords"]):
                if dataset["classes"]["data"][b][r] == dataset["classes"]["values"][b][y]:
                    dataset["classes"]["matrices"]["value"][b][y] += 1
    log_attributes_value_matrix(dataset)
    log_classes_value_matrix(dataset)
    return dataset

def log_attributes_value_matrix(dataset, filename = None):
    if not debug:
        return
    global attributes_matrices_value_count
    if filename == None:
        filename = str(attributes_matrices_value_count) + "-" + dataset["attributes"]["names"][0] + "_" + dataset["attributes"]["values"][0][0]
        attributes_matrices_value_count += 1
        if len(dataset["attributes"]["names"]) > 1:
            filename += "-" + dataset["attributes"]["names"][len(dataset["attributes"]["names"]) - 1]
            filename += "_" + dataset["attributes"]["values"][len(dataset["attributes"]["names"]) - 1][len(dataset["attributes"]["values"][len(dataset["attributes"]["names"]) - 1]) - 1]
        filename += "-value.txt"
    path = create_directory("output\\attributes\\matrices\\value")
    stdout = sys.stdout
    sys.stdout = open(path + "\\" + filename, "w")
    for b in range(len(dataset["classes"]["names"])):
        for y in range(len(dataset["classes"]["values"][b])):
            for a in range(len(dataset["attributes"]["names"])):
                for x in range(len(dataset["attributes"]["values"][a])):
                    if dataset["attributes"]["matrices"]["value"][a][b][x][y] > 0:
                        print(dataset["classes"]["names"][b], ":", dataset["classes"]["values"][b][y], "|", dataset["attributes"]["names"][a], ":", dataset["attributes"]["values"][a][x], "|", dataset["attributes"]["matrices"]["value"][a][b][x][y])
    sys.stdout.close()
    sys.stdout = stdout

def log_classes_value_matrix(dataset, filename = None):
    if not debug:
        return
    global classes_matrices_value_count
    if filename == None:
        filename = str(classes_matrices_value_count) + "-" + dataset["classes"]["names"][0] + "_" + dataset["classes"]["values"][0][0]
        classes_matrices_value_count += 1
        if len(dataset["classes"]["names"]) > 1:
            filename += "-" + dataset["classes"]["names"][len(dataset["classes"]["names"]) - 1]
            filename += "_" + dataset["classes"]["values"][len(dataset["classes"]["names"]) - 1][len(dataset["classes"]["values"][len(dataset["classes"]["names"]) - 1]) - 1]
        filename += "-value.txt"
    path = create_directory("output\\classes\\matrices\\value")
    stdout = sys.stdout
    sys.stdout = open(path + "\\" + filename, "w")
    for b in range(len(dataset["classes"]["names"])):
        for y in range(len(dataset["classes"]["values"][b])):
            print(dataset["classes"]["names"][b], ":", dataset["classes"]["values"][b][y], "|", dataset["classes"]["matrices"]["value"][b][y])
    sys.stdout.close()
    sys.stdout = stdout

def do_entropy_matrix(dataset):
    mp.dps = 55
    dataset["attributes"]["matrices"]["entropy"] = [None] * len(dataset["attributes"]["names"])
    for a in range(len(dataset["attributes"]["names"])):
        dataset["attributes"]["matrices"]["entropy"][a] = [None] * len(dataset["classes"]["names"])
        for b in range(len(dataset["classes"]["names"])):
            dataset["attributes"]["matrices"]["entropy"][a][b] = [None] * len(dataset["attributes"]["values"][a])
            for x in range(len(dataset["attributes"]["values"][a])):
                dataset["attributes"]["matrices"]["entropy"][a][b][x] = 0
                total = 0
                for y in range(len(dataset["classes"]["values"][b])):
                    total += dataset["attributes"]["matrices"]["value"][a][b][x][y]
                for y in range(len(dataset["classes"]["values"][b])):
                    if dataset["attributes"]["matrices"]["value"][a][b][x][y] == 0 or dataset["attributes"]["matrices"]["value"][a][b][x][y] == total:
                        continue
                    fraction = mpf(dataset["attributes"]["matrices"]["value"][a][b][x][y] / total)
                    try:
                        dataset["attributes"]["matrices"]["entropy"][a][b][x] -= mpf(fraction) * mpf(math.log(fraction, len(dataset["classes"]["values"][b])))
                    except ValueError:
                        dataset["attributes"]["matrices"]["entropy"][a][b][x] = 0
                        break
    dataset["classes"]["matrices"]["entropy"] = [None] * len(dataset["classes"]["names"])
    for b in range(len(dataset["classes"]["names"])):
        dataset["classes"]["matrices"]["entropy"][b] = 0
        total = 0
        for y in range(len(dataset["classes"]["values"][b])):
            total += dataset["classes"]["matrices"]["value"][b][y]
        for y in range(len(dataset["classes"]["values"][b])):
            if dataset["classes"]["matrices"]["value"][b][y] == 0 or dataset["classes"]["matrices"]["value"][b][y] == total:
                continue
            fraction = mpf(dataset["classes"]["matrices"]["value"][b][y] / total)
            try:
                entropy = mpf(dataset["classes"]["matrices"]["entropy"][b]) - mpf(fraction) * mpf(math.log(fraction, len(dataset["classes"]["values"][b])))
                dataset["classes"]["matrices"]["entropy"][b] = entropy
            except ValueError:
                dataset["classes"]["matrices"]["entropy"][b] = 0
                break
    log_attributes_entropy_matrix(dataset)
    log_classes_entropy_matrix(dataset)
    return dataset

def log_attributes_entropy_matrix(dataset, filename = None):
    if not debug:
        return
    global attributes_matrices_entropy_count
    if filename == None:
        filename = str(attributes_matrices_entropy_count) + "-" + dataset["attributes"]["names"][0] + "_" + dataset["attributes"]["values"][0][0]
        attributes_matrices_entropy_count += 1
        if len(dataset["attributes"]["names"]) > 1:
            filename += "-" + dataset["attributes"]["names"][len(dataset["attributes"]["names"]) - 1]
            filename += "_" + dataset["attributes"]["values"][len(dataset["attributes"]["names"]) - 1][len(dataset["attributes"]["values"][len(dataset["attributes"]["names"]) - 1]) - 1]
        filename += "-entropy.txt"
    path = create_directory("output\\attributes\\matrices\\entropy")
    stdout = sys.stdout
    sys.stdout = open(path + "\\" + filename, "w")
    for b in range(len(dataset["classes"]["names"])):
        for a in range(len(dataset["attributes"]["names"])):
            for x in range(len(dataset["attributes"]["values"][a])):
                print(dataset["classes"]["names"][b], "|", dataset["attributes"]["names"][a], ":", dataset["attributes"]["values"][a][x], "|", dataset["attributes"]["matrices"]["entropy"][a][b][x])
    sys.stdout.close()
    sys.stdout = stdout

def log_classes_entropy_matrix(dataset, filename = None):
    if not debug:
        return
    global classes_matrices_entropy_count
    if filename == None:
        filename = str(classes_matrices_entropy_count) + "-" + dataset["classes"]["names"][0]
        classes_matrices_entropy_count += 1
        if len(dataset["classes"]["names"]) > 1:
            filename += "-" + dataset["classes"]["names"][len(dataset["classes"]["names"]) - 1]
        filename += "-entropy.txt"
    path = create_directory("output\\classes\\matrices\\entropy")
    stdout = sys.stdout
    sys.stdout = open(path + "\\" + filename, "w")
    for b in range(len(dataset["classes"]["names"])):
        print(dataset["classes"]["names"][b], "|", dataset["classes"]["matrices"]["entropy"][b])
    sys.stdout.close()
    sys.stdout = stdout

def do_info_gain_matrix(dataset):
    dataset["attributes"]["matrices"]["info-gain"] = [None] * len(dataset["attributes"]["names"])
    dataset["classes"]["matrices"]["info-gain"] = [None] * len(dataset["attributes"]["names"])
    for a in range(len(dataset["attributes"]["names"])):
        dataset["attributes"]["matrices"]["info-gain"][a] = [None] * len(dataset["classes"]["names"])
        dataset["classes"]["matrices"]["info-gain"][a] = [None] * len(dataset["classes"]["names"])
        for b in range(len(dataset["classes"]["names"])):
            total_attribute_entropy = 0
            total_attribute_values = 0
            dataset["classes"]["matrices"]["info-gain"][a][b] = [None] * len(dataset["attributes"]["values"][a])
            for x in range(len(dataset["attributes"]["values"][a])):
                for y in range(len(dataset["classes"]["values"][b])):
                    total_attribute_values += dataset["attributes"]["matrices"]["value"][a][b][x][y]
            for x in range(len(dataset["attributes"]["values"][a])):
                for y in range(len(dataset["classes"]["values"][b])):
                    total_attribute_entropy += mpf(dataset["attributes"]["matrices"]["value"][a][b][x][y] / total_attribute_values) * mpf(dataset["attributes"]["matrices"]["entropy"][a][b][x])
            if dataset["classes"]["matrices"]["entropy"][b] > total_attribute_entropy:
                dataset["classes"]["matrices"]["info-gain"][a][b] = dataset["classes"]["matrices"]["entropy"][b] - total_attribute_entropy
            else:
                dataset["classes"]["matrices"]["info-gain"][a][b] = 0
            dataset["attributes"]["matrices"]["info-gain"][a][b] = [None] * len(dataset["attributes"]["names"])
            for i in range(len(dataset["attributes"]["names"])):
                dataset["attributes"]["matrices"]["info-gain"][a][b][i] = [None] * len(dataset["attributes"]["values"][i])
                for j in range(len(dataset["attributes"]["values"][i])):
                    if a == i:
                        dataset["attributes"]["matrices"]["info-gain"][a][b][i][j] = None
                        continue
                    dataset["attributes"]["matrices"]["info-gain"][a][b][i][j] = dataset["attributes"]["matrices"]["entropy"][i][b][j] - total_attribute_entropy
    log_attributes_info_gain_matrix(dataset)
    log_classes_info_gain_matrix(dataset)
    return dataset

def log_attributes_info_gain_matrix(dataset, filename = None):
    if not debug:
        return
    global attributes_matrices_info_gain_count
    if filename == None:
        filename = str(attributes_matrices_info_gain_count) + "-" + dataset["classes"]["names"][0] + "_" + dataset["attributes"]["names"][0] + "_" + dataset["attributes"]["values"][0][0]
        attributes_matrices_info_gain_count += 1
        if len(dataset["classes"]["names"]) > 1:
            filename += "-" + dataset["classes"]["names"][len(dataset["classes"]["names"]) - 1]
            filename += "_" + dataset["attributes"]["names"][len(dataset["attributes"]["names"]) - 1] + "_" + dataset["attributes"]["values"][len(dataset["attributes"]["names"]) - 1][len(dataset["attributes"]["values"][len(dataset["attributes"]["names"]) - 1]) - 1]
        filename += "-info-gain.txt"
    path = create_directory("output\\attributes\\matrices\\info-gain")
    stdout = sys.stdout
    sys.stdout = open(path + "\\" + filename, "w")
    for b in range(len(dataset["classes"]["names"])):
        for i in range(len(dataset["attributes"]["names"])):
            for j in range(len(dataset["attributes"]["values"][i])):
                for a in range(len(dataset["attributes"]["names"])):
                    print(dataset["classes"]["names"][b], "|", dataset["attributes"]["names"][i], ":", dataset["attributes"]["values"][i][j], "|", dataset["attributes"]["names"][a], "|", dataset["attributes"]["matrices"]["info-gain"][a][b][i][j])
    sys.stdout.close()
    sys.stdout = stdout

def log_classes_info_gain_matrix(dataset, filename = None):
    if not debug:
        return
    global classes_matrices_info_gain_count
    if filename == None:
        filename = str(classes_matrices_info_gain_count) + "-" + dataset["classes"]["names"][0] + "_" + dataset["attributes"]["names"][0]
        classes_matrices_info_gain_count += 1
        if len(dataset["classes"]["names"]) > 1:
            filename += "-" + dataset["classes"]["names"][len(dataset["classes"]["names"]) - 1]
            filename += "_" + dataset["attributes"]["names"][len(dataset["attributes"]["names"]) - 1]
        filename += "-info-gain.txt"
    path = create_directory("output\\classes\\matrices\\info-gain")
    stdout = sys.stdout
    sys.stdout = open(path + "\\" + filename, "w")
    for b in range(len(dataset["classes"]["names"])):
        for a in range(len(dataset["attributes"]["names"])):
            print(dataset["classes"]["names"][b], "|", dataset["attributes"]["names"][a], "|", dataset["classes"]["matrices"]["info-gain"][a][b])
    sys.stdout.close()
    sys.stdout = stdout

def do_tree_matrix(dataset):
    global do_path_depth
    dataset["classes"]["matrices"]["tree"] = []
    for b in range(len(dataset["classes"]["names"])):
        head = TreeNode(dataset["attributes"]["names"][0], dataset["attributes"]["values"][0], dataset["classes"]["matrices"]["info-gain"][0][b], 0)
        for a in range(1, len(dataset["attributes"]["names"])):
            temp = TreeNode(dataset["attributes"]["names"][a], dataset["attributes"]["values"][a], dataset["classes"]["matrices"]["info-gain"][a][b], a)
            if head.info_gain < temp.info_gain:
                head = temp
        tree = Tree(head, dataset["classes"]["names"][b])
        print(do_path_depth, "head")
        dataset["classes"]["matrices"]["tree"].append(do_path(dataset, tree, b))
    for b in range(len(dataset["classes"]["names"])):
        dataset["classes"]["matrices"]["tree"][b].head.get_good()
    log_classes_tree_matrix(dataset)
    return dataset  

def do_path(dataset, tree, cls_index):
    global do_path_depth
    do_path_depth += 1
    b = cls_index
    a = tree.current.index
    for x in range(len(tree.current.paths)):
        newset = get_new_set(dataset, tree, x)
        newset = do_value_matrix(newset)
        newset = do_entropy_matrix(newset)
        if newset["attributes"]["matrices"]["entropy"][a][b][x] == 0:
            h = 0
            for y in range(1, len(newset["classes"]["values"][b])):
                if newset["attributes"]["matrices"]["value"][a][b][x][y] > newset["attributes"]["matrices"]["value"][a][b][x][h]:
                    h = y
            tree.current.paths[x].connect(TreeLeaf(newset["classes"]["values"][b][h], newset["classes"]["goodness"][b][h]))
            print(do_path_depth, "    leaf")
        else:
            newset = do_info_gain_matrix(newset)
            node = TreeNode(newset["attributes"]["names"][0], newset["attributes"]["values"][0], newset["attributes"]["matrices"]["info-gain"][0][b][a][x], 0)
            for i in range(1, len(newset["attributes"]["names"])):
                if newset["attributes"]["matrices"]["info-gain"][i][b][a][x] == None:
                    continue
                temp = TreeNode(newset["attributes"]["names"][i], newset["attributes"]["values"][i], newset["attributes"]["matrices"]["info-gain"][i][b][a][x], i)
                if node.info_gain == None or node.info_gain < temp.info_gain:
                    node = temp
            current = tree.current
            tree.current = tree.current.paths[x].connect(node)
            print(do_path_depth, "  node")
            tree = do_path(newset, tree, b)
            tree.current = current
    do_path_depth -= 1
    return tree
    
def log_classes_tree_matrix(dataset, filename = None):
    path = create_directory("output\\classes\\matrices\\tree")
    stdout = sys.stdout
    for t in range(len(dataset["classes"]["matrices"]["tree"])):
        filename = dataset["classes"]["names"][t] + "-tree.txt"
        sys.stdout = open(path + "\\" + filename, "w")
        dataset["classes"]["matrices"]["tree"][t].print_tree()
        sys.stdout.close()
    sys.stdout = stdout

def do_prediction_matrix(dataset):
    filename = input("enter testset filename (e.g. test.csv): ")
    while re.search(r"\.csv$", filename) == None:
        print("\tincorrect dataset file type (must be .csv):", filename)
        filename = input("enter testset filename (e.g. test.csv): ")
    testset = get_dataset(filename, len(dataset["classes"]["names"]))
    if len(testset["attributes"]["names"]) != len(dataset["attributes"]["names"]) or len(testset["classes"]["names"]) != len(dataset["classes"]["names"]):
        print("\tnumber of attributes or classes do not match testset | dataset:")
        print("\t\tattribute count", "-", len(testset["attributes"]["names"]), "|", len(dataset["attributes"]["names"]))
        print("\t\tclass count", "-", len(testset["classes"]["names"]), "|", len(dataset["classes"]["names"]))
        return None     
    for a in range(len(testset["attributes"]["names"])):
        if testset["attributes"]["names"][a] != dataset["attributes"]["names"][a]:
            print("\tattribute names do not match testset | dataset:", testset["attributes"]["names"][a], "|", dataset["attributes"]["names"][a])
            return None
    for b in range(len(testset["classes"]["names"])):
        if testset["classes"]["names"][b] != dataset["classes"]["names"][b]:
            print("\tclass names do not match testset | dataset:", testset["classes"]["names"][b], "|", dataset["classes"]["names"][b])
            return None
    dataset["classes"]["matrices"]["prediction"] = []
    prediction = ""
    for a in range(len(testset["attributes"]["names"])):
        prediction += testset["attributes"]["names"][a] + ";"
    prediction += testset["classes"]["names"][0]
    for b in range(1, len(testset["classes"]["names"])):
        prediction += ";" + testset["classes"]["names"][b]
    dataset["classes"]["matrices"]["prediction"].append(prediction)
    for r in range(testset["nrecords"]):
        prediction = ""
        for a in range(len(testset["attributes"]["names"])):
            prediction += testset["attributes"]["data"][a][r] + ";"
        prediction += str(dataset["classes"]["matrices"]["tree"][0].evaluate_data(testset, r))
        for b in range(1, len(testset["classes"]["names"])):
            prediction += ";" + str(dataset["classes"]["matrices"]["tree"][b].evaluate_data(testset, r))
        dataset["classes"]["matrices"]["prediction"].append(prediction)
    log_classes_prediction_matrix(dataset)
    return dataset

def log_classes_prediction_matrix(dataset):
    global classes_matrices_prediction_count
    classes_matrices_prediction_count += 1
    path = create_directory("output\\classes\\matrices\\prediction")
    filename = str(classes_matrices_prediction_count) + "-prediction.csv"
    stdout = sys.stdout
    sys.stdout = open(path + "\\" + filename, "w")
    for r in range(len(dataset["classes"]["matrices"]["prediction"])):
        print(dataset["classes"]["matrices"]["prediction"][r])
    sys.stdout.close()
    sys.stdout = stdout

def get_new_set(dataset, tree, val_index):
    newset = copy.deepcopy(dataset)
    v = val_index
    a = tree.current.index
    popping = True
    if "-" in tree.current.paths[v].value:
        rng = tree.current.paths[v].value.split("-")
        while popping:
            popping = False
            for r in range(newset["nrecords"]):
                if newset["attributes"]["data"][a][r] not in range(int(rng[0]), int(rng[1]) + 1):
                    for i in range(len(newset["attributes"]["data"])):
                        newset["attributes"]["data"][i].pop(r)
                    for j in range(len(newset["classes"]["data"])):
                        newset["classes"]["data"][j].pop(r)
                    newset = update_nrecords(newset)
                    popping = True
                    break
    else:
        while popping:
            popping = False
            for r in range(newset["nrecords"]):
                if newset["attributes"]["data"][a][r] != tree.current.paths[v].value:
                    for i in range(len(newset["attributes"]["data"])):
                        newset["attributes"]["data"][i].pop(r)
                    for j in range(len(newset["classes"]["data"])):
                        newset["classes"]["data"][j].pop(r)
                    newset = update_nrecords(newset)
                    popping = True
                    break
    log_dataset_attributes(newset)
    return newset

def create_directory(directory):
    cwd = os.path.dirname(os.path.abspath(__file__))
    dirs = directory.split("\\")
    dir = ""
    path = ""
    for d in range(len(dirs)):
        dir += "\\" + dirs[d]
        path = cwd + dir
        try:
            os.mkdir(path)
        except FileExistsError:
            continue
    return path

class Tree(object):
    def __init__(self, head, classifier):
        self.max_depth = 0
        self.min_depth = 0
        self.depth = 0
        self.head = head
        self.classifier = classifier
        self.current = self.head
    
    def get_previous(self):
        if self.current.parent_path == None:
            self.current = self.head
        else:
            self.current = self.current.parent_path.parent
        return self.current

    def get_paths(self):
        return self.current.paths

    def get_children(self):
        children = []
        for x in range(len(self.current.paths)):
            children.append(self.current.paths[x].child)
        return children

    def num_leaves(self):
        current = self.current
        self.current = self.head
        num = self.count_leaves()
        self.current = current
        return num

    def count_leaves(self):
        if type(self.current) is TreeLeaf:
            return 1
        else:
            count = 0
            current = self.current
            for p in range(len(self.current.paths)):
                self.current = self.current.paths[p].child
                count += self.count_leaves()
                self.current = current
            return count

    def evaluate_data(self, dataset, r):
        current = self.current
        self.current = self.head
        result = self.evaluate_record(dataset, r)
        self.current = current
        return result
    
    def evaluate_record(self, dataset, r):
        if type(self.current) is TreeLeaf:
            return self.current.value
        else:
            current = self.current
            for i in range(len(dataset["attributes"]["names"])):
                if dataset["attributes"]["names"][i] == self.current.name:
                    for j in range(len(self.current.values)):
                        if "-" in self.current.values[j]:
                            rng = self.current.values[j].split("-")
                            if int(dataset["attributes"]["data"][i][r]) in range(int(rng[0]), int(rng[1]) + 1):
                                self.current = self.current.paths[j].child
                                result = self.evaluate_record(dataset, r)
                                self.current = current
                                return result
                        else:
                            if dataset["attributes"]["data"][i][r] == self.current.values[j]:
                                self.current = self.current.paths[j].child
                                result = self.evaluate_record(dataset, r)
                                self.current = current
                                return result

    def print_tree(self):
        current = self.current
        self.current = self.head
        self.__pt__()
        self.print_depth()
        self.current = current

    def print_depth(self):
        print("max depth:", self.max_depth)
        print("min depth:", self.min_depth)
    
    def __pt__(self):
        indent = "  " * self.depth
        if type(self.current) is TreeLeaf:
            self.min_depth = self.depth if self.min_depth == 0 else self.min_depth
            self.max_depth = self.depth if self.depth > self.max_depth else self.max_depth
            self.min_depth = self.depth if self.depth < self.min_depth else self.min_depth
            print(self.depth, indent, "leaf", "|", self.classifier, "|", self.current.parent_path.parent.name, ":", self.current.parent_path.value, "|", self.current.value)
        else:
            children = self.get_children()
            prev = ""
            if self.current.parent_path == None:
                prev = "head | " + self.classifier
            else:
                prev = "node | " + self.classifier + " | " + str(self.current.parent_path.parent.name) + " : " + str(self.current.parent_path.value)
            paths = self.current.paths[0].value + " [ " + str(round(self.current.paths[0].child.goodness, 4)) + " ] "
            for p in range(1, len(self.current.paths)):
                paths += " : " + str(self.current.paths[p].value) + " [ " + str(round(self.current.paths[p].child.goodness, 4)) + " ] "
            print(self.depth, indent, prev, "|", self.current.name, "|", paths)
            for c in range(len(children)):
                current = self.current
                self.current = children[c]
                self.depth += 1
                self.__pt__()
                self.current = current
        self.depth -= 1 if self.depth > 0 else 0

class TreeNode(object):
    def __init__(self, name, values, info_gain, index, parent_path = None):
        self.name = name
        self.values = values
        self.info_gain = info_gain
        self.index = index
        self.parent_path = parent_path
        self.paths = []
        self.goodness = 0
        for v in range(len(values)):
            self.paths.append(TreePath(self, values[v], v))

    def get_good(self):
        for p in range(len(self.paths)):
            self.goodness += self.paths[p].child.get_good() / len(self.paths)
        return self.goodness

class TreePath(object):
    def __init__(self, parent, value, index):
        self.parent = parent
        self.value = value
        self.index = index
    def connect(self, child):
        self.child = child
        self.child.parent_path = self
        return self.child

class TreeLeaf(object):
    def __init__(self, value, goodness):
        self.value = value
        self.goodness = goodness
        self.parent_path = None
    def get_good(self):
        return self.goodness

__main()