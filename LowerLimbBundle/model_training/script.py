import json

# generate all case numbers 1–56, except 13
cases = [i for i in range(1, 57) if i != 13]

training = []
num_folds = 3

for idx, case in enumerate(cases):
    img = f"image_{case:02d}.nii.gz"
    lbl = f"label_{case:02d}.nii.gz"
    fold = idx % num_folds
    training.append({
        "fold": fold,
        "image": img,
        "label": lbl
    })

datalist = {"training": training, "testing": []}

with open("task.json", "w") as f:
    json.dump(datalist, f, indent=2)