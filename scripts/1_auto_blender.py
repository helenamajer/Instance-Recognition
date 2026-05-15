# script to generate annotated dataset
import bpy
import json
import math
import numpy as np
import cv2
from mathutils import Vector
import random
import os

# Paths #

# path to .OBJ file *removing hardcoded file paths soon*
file_path = " "
output_path = " "
part_name = " "
part_number = " "
# increment this per part (0, 1, 2, and so on)
class_id = 0

# Render Settings #

# 1-10 for testing, ~250 for final dataset
num_renders = 250
min_radius = 7.5
max_radius = 9.5

os.makedirs(output_path, exist_ok=True)

# Clear scene 
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import .OBJ 
before = set(bpy.context.scene.objects)
bpy.ops.wm.obj_import(filepath=file_path)
bpy.context.view_layer.update()

after = set(bpy.context.scene.objects)
new_objects = list(after - before)

bpy.ops.object.select_all(action='DESELECT')
for o in new_objects:
    o.select_set(True)
if new_objects:
    bpy.context.view_layer.objects.active = new_objects[0]

# Normalize - bring object to origin 
obj = new_objects[0]
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
obj.location = (0, 0, 0)

# Create and assign material if object has none.
if len(obj.material_slots) == 0:
    mat = bpy.data.materials.new(name="PartMaterial")
    obj.data.materials.append(mat)

# grab the material and set up Principled BSDF nodes
mat = obj.material_slots[0].material
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

principled = nodes.new("ShaderNodeBsdfPrincipled")
output = nodes.new("ShaderNodeOutputMaterial")
links.new(principled.outputs["BSDF"], output.inputs["Surface"])

# store reference for colour randomisation in render loop
part_material = principled

# Add camera to scene
cam_data = bpy.data.cameras.new("Camera")
cam = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam)
bpy.context.scene.camera = cam

# Add light to scene
light_data = bpy.data.lights.new(name="Light", type='AREA')
light = bpy.data.objects.new(name="Light", object_data=light_data)
bpy.context.collection.objects.link(light)

# Render settings 
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 64
scene.render.image_settings.file_format = 'PNG'
scene.render.resolution_x = 720
scene.render.resolution_y = 720
# transparent background
scene.render.film_transparent = True
scene.render.image_settings.color_mode = 'RGBA'      

# Segmentation Mask #

# Strategy: render RGB normally,
# then swap to a flat white material and black background to capture objects silhouette.
# render again to get the mask, then swap back to normal render (randomized lighting and angle).
mask_mat = bpy.data.materials.new(name="MaskMaterial")
# enable node based shade editing.
mask_mat.use_nodes = True
# Clear Blender's default nodes (settings).
nodes = mask_mat.node_tree.nodes
links = mask_mat.node_tree.links
nodes.clear()
# Add an emission shader node - ignores all scene lighting and emits light of its own.
emission = nodes.new("ShaderNodeEmission")
# Sets emission color to solid white.
emission.inputs["Color"].default_value = (1, 1, 1, 1)
emission.inputs["Strength"].default_value = 1.0
output = nodes.new("ShaderNodeOutputMaterial")
# Material output so the white emission is what gets rendered.
links.new(emission.outputs["Emission"], output.inputs["Surface"])

# Store the original materials to restore them after the mask render.
original_materials = [slot.material for slot in obj.material_slots]

# Helpers 
def look_at(cam, target):
    direction = target - cam.location
    rotation_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rotation_quat.to_euler()

def random_camera_position():
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(math.radians(20), math.radians(80))
    r = random.uniform(min_radius, max_radius)
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    return Vector((x, y, z))

# Trace the outline of the part to acurrately capture the silhouette
def mask_to_yolo_polygon(mask_path, img_w, img_h):
    # Convert a binary mask PNG to a normalized YOLO segmentation polygon.
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    # convert greyscale image to purely black and white.
    # any pixel > 127 becomes 255 (white), anything less than 127 becomes 0 (black).
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # findContours traces the outline of all white regions in the masked image.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Out of all the traces in the images, the largest trace (by area) is the part outline.
    # so holes and small white specs are ingnored.
    contour = max(contours, key=cv2.contourArea)
    # simplify so we don't write thousands of points
    epsilon = 0.002 * cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, epsilon, True)

    # This block makes up the .txt annotation of each image.
    # first the class ID, then the altering x y coordinates of the polygon (trace).
    # YOLO requires all coordinates to be normalized between 0 and 1 relative to the image dimensions.
    points = simplified.reshape(-1, 2)
    normalized = [(x / img_w, y / img_h) for x, y in points]
    flat = [val for point in normalized for val in point]
    coords = " ".join(f"{v:.6f}" for v in flat)
    return f"{class_id} {coords}"

base_name = os.path.splitext(os.path.basename(file_path))[0]

# Render loop 
for i in range(num_renders):
    frame_str = f"{i:04d}"

    # Camera and light randomization.
    cam.location = random_camera_position()
    look_at(cam, obj.location)
    light.location = (
        random.uniform(-5, 5),
        random.uniform(-5, 5),
        random.uniform(2, 8),
    )
    light.data.energy = random.uniform(300, 1500)

    # Color randomization
    part_material.inputs["Base Color"].default_value = (
        random.uniform(0.0, 1.0), # R
        random.uniform(0.0, 1.0), # G
        random.uniform(0.0, 1.0), # B
        1.0 # A
    )

    # Render 1: normal greyscale image.
    scene.render.filepath = os.path.join(output_path, f"{base_name}_{frame_str}.png")
    scene.cycles.samples = 64
    bpy.ops.render.render(write_still=True)

    # Render 2: segmentation mask - flat white mask, black background.
    # Swap to mask material.
    for slot in obj.material_slots:
        slot.material = mask_mat

    mask_path = os.path.join(output_path, f"{base_name}_{frame_str}_mask.png")
    scene.render.filepath = mask_path
    scene.cycles.samples = 1
    bpy.ops.render.render(write_still=True)

    # Restore original materials.
    for slot, mat in zip(obj.material_slots, original_materials):
        slot.material = mat

    # Reset sample count for next RGB render.
    scene.cycles.samples = 64

    # Convert mask to YOLO annotation (.txt file for eacg image).
    yolo_line = mask_to_yolo_polygon(
        mask_path,
        scene.render.resolution_x,
        scene.render.resolution_y,
    )

    txt_path = os.path.join(output_path, f"{base_name}_{frame_str}.txt")
    if yolo_line:
        with open(txt_path, "w") as f:
            f.write(yolo_line + "\n")
    else:
        print(f"[WARNING] No contour found for frame {i} — skipping annotation.")

    # Delete intermediate mask file.
    if os.path.exists(mask_path):
        os.remove(mask_path)

    print(f"[{i+1}/{num_renders}] {base_name}_{frame_str}.png + .txt done")

# Write one metadata.json for the whole part folder 
metadata = {
    "name": part_name,
    "part_number": part_number,
    "class_id": class_id,
}
with open(os.path.join(output_path, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Dataset generation complete — {num_renders} renders written to {output_path}")