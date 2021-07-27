import re

from pathlib import Path
from collections import defaultdict, OrderedDict

import matplotlib.pyplot as plt

import pandas as pd

MIN_TRIP_LENGTH = 1000

COARSE_LABELS = {
    0: 'Null',
    1: 'Still',
    2: 'Walking',
    3: 'Run',
    4: 'Bike',
    5: 'Car',
    6: 'Bus',
    7: 'Train',
    8: 'Subway',
}

UNWANTED_LABELS = [ 'Null' ]

MOTION_COLUMNS = [
    'Time',
    'Acceleration X',
    'Acceleration Y',
    'Acceleration Z',
    'Gyroscope X',
    'Gyroscope Y',
    'Gyroscope Z',
    'Magnetometer X',
    'Magnetometer Y',
    'Magnetometer Z',
    'Orientation w',
    'Orientation x',
    'Orientation y',
    'Orientation z',
    'Gravity X',
    'Gravity Y',
    'Gravity Z',
    'Linear acceleration X',
    'Linear acceleration Y',
    'Linear acceleration Z',
    'Pressure',
    'Altitude',
    'Temperature',
]

BOXPLOT_GROUPS = {
    'Acceleration': [
        'Acceleration X',
        'Acceleration Y',
        'Acceleration Z',
    ],
    'Gyroscrope': [
        'Gyroscope X',
        'Gyroscope Y',
        'Gyroscope Z',
    ],
    'Magnetometer': [
        'Magnetometer X',
        'Magnetometer Y',
        'Magnetometer Z',
    ],
    'Orientation': [
        'Orientation w',
        'Orientation x',
        'Orientation y',
        'Orientation z',
    ],
    'Gravity': [
        'Gravity X',
        'Gravity Y',
        'Gravity Z',
    ],
    'Linear Acceleration': [
        'Linear acceleration X',
        'Linear acceleration Y',
        'Linear acceleration Z',
    ],
    'Other': [
        'Pressure',
        'Altitude',
        'Temperature',
    ],
}

LABEL_COLUMNS = [
    'Time',
    'Coarse Label',
    'Fine Label',
    'Road Label',
    'Traffic Label',
    'Tunnels Label',
    'Social Label',
    'Food Label',
]

pd.set_option("display.precision", 2)

release_dir = Path('shl/release')
records_dir = release_dir / 'User1'
record_dirs = [r for r in records_dir.iterdir() if r.is_dir()]

def load_motion_data(record_dir: Path) -> pd.DataFrame:
    motion_data = pd.read_csv(
        record_dir / 'Hips_Motion.txt',
        sep=' ',
        header=None,
        names=MOTION_COLUMNS,
        low_memory=False
    )
    return motion_data


def load_labels_data(record_dir: Path) -> pd.DataFrame:
    labels_data = pd.read_csv(
        record_dir / 'Label.txt',
        sep=' ',
        header=None,
        names=LABEL_COLUMNS,
        low_memory=False
    )
    labels_data.replace({
        'Coarse Label': COARSE_LABELS,
    }, inplace=True)
    return labels_data


def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_').lower()
    return re.sub(r'(?u)[^-\w.]', '', s)


joined_data = None

for record_dir in record_dirs:
    try:
        motion_data = load_motion_data(record_dir)
        labels_data = load_labels_data(record_dir)
    except FileNotFoundError:
        print(f'Missing file for dir {record_dir}')
        continue

    # In the documentation, it is said that every line
    # in Hips_Motion.txt corresponds to the exact same line
    # in Label.txt, therefore we can just merge them together
    try:
        data = pd.merge(motion_data, labels_data)
    except ValueError:
        print(f'Data under dir {record_dir} has erroneous format')
        continue

    # Drop rows that contain any NaN
    data.dropna(how='any', inplace=True)

    if joined_data is None:
        joined_data = data
    else:
        joined_data = joined_data.append(data)

    print(f'Loaded dataframe under dir {record_dir}')

# Drop columns that are all NaN
joined_data.dropna(axis='columns', how='all', inplace=True)

joined_data_description = joined_data.describe()
print(joined_data_description)

# Write the data description as a LateX table
with open('shl_data_description.tex', 'w') as f:
    f.write(joined_data_description.to_latex(
        float_format="{:0.2f}".format,
        decimal=',',
        caption='Datenattribute des SHL-Datensatzes.',
        label='tab:shl-analyse'
    ))

# Build a trip column that increments every time
# the label changes, for grouping
joined_data['Trip'] = joined_data['Coarse Label'].ne(
    joined_data['Coarse Label'].shift()
).cumsum()

trips_by_label = defaultdict(list)
for trip_id, data in joined_data.groupby('Trip'):
    length, _ = data.shape
    if length < MIN_TRIP_LENGTH:
        print(f'Warning: Dropped trip {trip_id} because it was too short! (Height {length})')
        continue
    label = data['Coarse Label'].values[0]
    # Drop unwanted labels
    if label in UNWANTED_LABELS:
        continue
    trips_by_label[label].append(data)

# Create boxplots for each sensor
# to show differences between labels
for sensor_group, sensors in BOXPLOT_GROUPS.items():
    plt.clf()
    fig, axs = plt.subplots(1, len(sensors))
    fig.set_size_inches(6 * len(sensors), 6)

    for i, sensor in enumerate(sensors):
        diagram_dict = OrderedDict()
        for label, trips in trips_by_label.items():
            # Concatenate individual trips to one single trip
            trips_data = pd.concat(trips)
            sensor_data = trips_data[sensor].to_numpy()
            diagram_dict[label] = sensor_data
        diagram_labels, diagram_data = [*zip(*diagram_dict.items())]

        bp = axs[i].boxplot(diagram_data, showfliers=False)
        axs[i].set_xticks(range(1, len(diagram_labels) + 1))
        axs[i].set_xticklabels(diagram_labels)
        axs[i].set_title(sensor)

        for element in [
            'boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps'
        ]:
            plt.setp(bp[element], color='black')

    plt.savefig(
        f'{get_valid_filename(sensor_group)}.pdf',
        dpi=1200,
        bbox_inches='tight'
    )
    print(f'Computed statistics for sensor group {sensor_group}')


# Create bar plot for each label
labels = []
quantities = []
for label, trips in trips_by_label.items():
    trips_data = pd.concat(trips)
    labels.append(label)
    height, _ = trips_data.shape
    quantities.append(height)
plt.clf()
fig, ax = plt.subplots()
ax.bar(labels, quantities, color='black')
ax.ticklabel_format(style='plain', axis='y')
plt.savefig(
    f'shl-label-quantities.pdf',
    dpi=1200,
    bbox_inches='tight'
)

# Create a boxplot to show how long the trips are
# for each label
diagram_dict = OrderedDict()
for label, trips in trips_by_label.items():
    lengths = []
    for trip in trips:
        height, _ = trip.shape
        lengths.append(height)
    diagram_dict[label] = lengths
plt.clf()
fig, ax = plt.subplots()
diagram_labels, diagram_data = [*zip(*diagram_dict.items())]
bp = ax.boxplot(diagram_data, showfliers=False)
ax.set_xticks(range(1, len(diagram_labels) + 1))
ax.set_xticklabels(diagram_labels)
ax.set_title(sensor)

for element in [
    'boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps'
]:
    plt.setp(bp[element], color='black')

plt.savefig(
    f'shl-trip-lengths.pdf',
    dpi=1200,
    bbox_inches='tight'
)

