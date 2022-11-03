import csv
import pandas as pd


def data_importer(training_path, test_path):
    with open(training_path, "r") as f:
        csvreader = csv.reader(f, delimiter="\n")
        columns = next(csvreader)[0].split(",")
        data = []
        for row in csvreader:
            sample = row[0]
            comma_idx = [i for i, ltr in enumerate(sample) if ltr == ',']
            temp_list = [sample[:(comma_idx[0])], sample[(comma_idx[0]+1):comma_idx[-7]], sample[(comma_idx[-7]+1):comma_idx[-6]], sample[(comma_idx[-6]+1):comma_idx[-5]], sample[(
                comma_idx[-5]+1):comma_idx[-4]], sample[(comma_idx[-4]+1):comma_idx[-3]], sample[(comma_idx[-3]+1):comma_idx[-2]], sample[(comma_idx[-2]+1):comma_idx[-1]], sample[(comma_idx[-1]+1):]]
            data.append(temp_list)
    training_data = pd.DataFrame(data, columns=columns)

    with open(test_path, "r") as f:
        csvreader = csv.reader(f, delimiter="\n")
        columns = next(csvreader)[0].split(",")
        data = []
        for row in csvreader:
            sample = row[0]
            comma_idx = [i for i, ltr in enumerate(sample) if ltr == ',']
            temp_list = [sample[:(comma_idx[0])], sample[(comma_idx[0]+1):]]
            data.append(temp_list)
    test_data = pd.DataFrame(data, columns=columns)

    print(f"Shape of Training Data is {training_data.shape}")
    print(f'Columns in Training Data: {training_data.columns.values}')
    print(f"Shape of Testing Data is {test_data.shape}")
    print(f'Columns in Testing Data: {test_data.columns.values}')

    del csvreader, columns, data, sample, comma_idx, temp_list, training_path, test_path

    return training_data, test_data
