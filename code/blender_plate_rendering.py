import math
import os
import sys

import bpy

argv = sys.argv
argv = argv[argv.index("--") + 1 :]  # Get arguments after "--"
folder_path = argv[0][:-5]
filepath = os.path.join(folder_path, "teaser.ply")

# set up camera and scene settings
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.object.camera_add()
camera = bpy.context.object
camera.data.clip_start = 0.0001
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.use_adaptive_sampling = True
scene.camera = camera
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.render.image_settings.file_format = "PNG"
scene.render.resolution_x = 2048
scene.render.resolution_y = 2048
scene.cycles.samples = 256
scene.cycles.seed = 42
scene.cycles.use_denoising = True
scene.render.film_transparent = False

# white world background
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg_node = world.node_tree.nodes["Background"]
bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
bg_node.inputs["Strength"].default_value = 1.0

# load plate mesh
bpy.ops.wm.ply_import(filepath=filepath)
bpy.ops.object.shade_flat()
mesh_obj = bpy.context.selected_objects[0]
mesh_obj.scale.y = -1
bpy.ops.object.transform_apply(scale=True)
plate_material = bpy.data.materials.new(name="PlateMaterial")
plate_material.use_nodes = True
nodes = plate_material.node_tree.nodes
nodes.clear()
principled_bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
fac = 1.5
principled_bsdf.inputs["Base Color"].default_value = (0.03 * fac, 0.08 * fac, 0.2 * fac, 1)
principled_bsdf.inputs["Roughness"].default_value = 0.5
principled_bsdf.inputs["Specular IOR Level"].default_value = 0.0
output = nodes.new(type="ShaderNodeOutputMaterial")
links = plate_material.node_tree.links
links.new(principled_bsdf.outputs["BSDF"], output.inputs["Surface"])
mesh_obj.data.materials.append(plate_material)

# create light
light_data = bpy.data.lights.new(name="Sun", type="SUN")
light_object = bpy.data.objects.new(name="Sun", object_data=light_data)
bpy.context.collection.objects.link(light_object)
light_object.location = (0.5, 0.5, 10)
light_object.rotation_euler = (math.radians(15), 0, math.radians(10))
light_data.energy = 1.0
light_data.angle = math.radians(30)


def render_view(view_name, camera_location, camera_rotation):
    camera.location = camera_location
    camera.rotation_euler = camera_rotation
    scene.render.filepath = os.path.join(folder_path, view_name + ".png")
    bpy.ops.render.render(write_still=True)


camera.data.type = "ORTHO"
camera.data.ortho_scale = 1.05
z_cam = 1.5
render_view("render_top_view", (0.5, -0.5, z_cam), (0, 0, 0))
camera.data.type = "PERSP"
for i in range(7):
    z_cam /= 1.5
    render_view(
        "render_closeup_" + str(i),
        (0.5, -0.7, z_cam),
        (math.radians(14), 0, math.radians(14)),
    )
