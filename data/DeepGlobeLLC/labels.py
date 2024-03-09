from collections import namedtuple

# Define the Label named tuple
Label = namedtuple('Label', ['name', 'id', 'color','ignoreInEval'])

# Define the list of labels
labels = [
    Label('urban_land',       3,  (0, 255, 255),False),
    Label('agriculture_land', 6,  (255, 255, 0),False),
    Label('rangeland',        5,  (255, 0, 255),False),
    Label('forest_land',      2,  (0, 255, 0),False),
    Label('water',            1,  (0, 0, 255),False),
    Label('barren_land',      7,  (255, 255, 255),False),
    Label('unknown',          0,  (0, 0, 0),True)
]

def get_train_ids():
    train_ids = []
    for i in labels:
        if not i.ignoreInEval:
            train_ids.append(i.id)
    return train_ids