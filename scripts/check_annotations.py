from pathlib import Path

# Before training, you can double check the annotations (labels) in the prepared training data.
# If an annotation is too short, it can cause the training to crash.
LABELS = Path("data/yolo_dataset/labels/train")
errors = []

for txt_file in LABELS.glob("*.txt"):
    lines = txt_file.read_text().strip().splitlines()
    
    if not lines:
        errors.append(f"EMPTY: {txt_file.name}")
        continue
    
    for line in lines:
        parts = line.strip().split()
        
        # Annotations must have a class_id and at least 6 coordinates (3 points minimum).
        if len(parts) < 7:
            errors.append(f"TOO SHORT: {txt_file.name} — {line}")
            continue
        
        coords = [float(x) for x in parts[1:]]
        
        # All coordinates must be between 0 and 1.
        if any(c < 0 or c > 1 for c in coords):
            errors.append(f"OUT OF BOUNDS: {txt_file.name} — {line}")

if errors:
    print(f"{len(errors)} Problems found:\n")
    # Show the first 20 bad annotations.
    for e in errors[:20]:
        print(e)
else:
    print("All annotations look clean, ready to train!")