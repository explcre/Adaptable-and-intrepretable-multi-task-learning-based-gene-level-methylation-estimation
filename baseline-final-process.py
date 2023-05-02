import csv
import math

# Calculate the weighted average accuracy and standard deviation
def calculate_weighted_metrics(acc_list, std_list, weights):
    weighted_acc = sum([acc * weight for acc, weight in zip(acc_list, weights)]) / sum(weights)
    weighted_std = math.sqrt(sum([(std ** 2) * weight for std, weight in zip(std_list, weights)]) / sum(weights))
    return weighted_acc, weighted_std

# Read the data from the local CSV file
with open('output.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')

    header = next(reader)
    header.extend(['accuracy', 'std'])

    data = {row[0]: row[1:] for row in reader}
    weights = [53, 88, 79, 49, 160, 118]

# Process the data
for key in data:
    method, site = key.split('site')
    site = site.strip()

    acc_list = [float(data[key][i]) for i in range(0, len(data[key]), 2)]
    std_list = [float(data[key][i + 1]) if data[key][i + 1] != 'nan' else 0 for i in range(0, len(data[key]), 2)]

    weighted_acc, weighted_std = calculate_weighted_metrics(acc_list, std_list, weights)
    data[key].extend([weighted_acc, weighted_std])

# Merge the data
merged_data = {}
for key, values in data.items():
    method, site = key.split('site')
    site = site.strip()

    if method not in merged_data:
        merged_data[method] = [None] * 26

    index = header.index(f'{site}_acc') - 1
    merged_data[method][index:index + 4] = values

# Write the merged data to a new CSV file
with open('merged_output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(header)
    for method, values in merged_data.items():
        writer.writerow([method] + values)
