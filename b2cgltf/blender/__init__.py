"""b2c subjects in Blender 5.3+ (native Gaussian splats): the add-on's code.

posing.py and correctives.py are numpy only (tests run them without Blender); nodes.py, importer.py, runtime.py and
addon.py need bpy. tools/build_blender_addon.py packages this with the rest of b2cgltf as a Blender extension; the
extension's own __init__ calls register(__package__).
"""


def register(package: str | None = None):
    from . import addon
    addon.register(package)


def unregister():
    from . import addon
    addon.unregister()
